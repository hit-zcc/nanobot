"""Claude OAuth provider — uses Claude Max subscription via OAuth PKCE flow."""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
import time
import webbrowser
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.base import LLMResponse

# Claude Code's public OAuth client (extracted from Claude Code source)
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_API_BASE = "https://api.anthropic.com"
CLAUDE_SCOPES = "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
REDIRECT_URI = "http://localhost:0/callback"

TOKEN_FILENAME = "claude-oauth.json"
_EXPIRY_SKEW_SECONDS = 120  # refresh 2 min before expiry
_DEFAULT_EXPIRES_IN = 28800  # 8 hours


def _token_path() -> Path:
    """Return the path to the stored Claude OAuth token."""
    config_dir = Path.home() / ".nanobot" / "auth"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / TOKEN_FILENAME


def _claude_code_credentials_path() -> Path:
    """Return the path to Claude Code's headless OAuth credentials file."""
    return Path.home() / ".claude" / ".credentials.json"


KEYCHAIN_SERVICE = "Claude Code-credentials"

# Where a borrowed Claude Code token came from, so a refresh can be written
# back to the same store instead of orphaning it.
SOURCE_KEYCHAIN = "claude_code_keychain"
SOURCE_CC_FILE = "claude_code_file"


def _read_claude_code_keychain() -> str | None:
    """Return the raw Claude Code credentials JSON from the macOS Keychain."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _keychain_account() -> str | None:
    """Return the account name of the Claude Code keychain item."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if '"acct"<blob>=' in line:
            _, _, value = line.partition('"acct"<blob>=')
            value = value.strip()
            return value[1:-1] if value.startswith('"') and value.endswith('"') else None
    return None


def _write_claude_code_keychain(raw: str) -> bool:
    """Replace the Claude Code keychain item's payload with *raw*."""
    account = _keychain_account()
    if account is None:
        return False
    try:
        result = subprocess.run(
            ["security", "add-generic-password", "-U",
             "-s", KEYCHAIN_SERVICE, "-a", account, "-w", raw],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0


def _merge_claude_code_credentials(raw: str, token_data: dict[str, Any]) -> str | None:
    """Fold refreshed tokens into Claude Code's blob, preserving its other keys.

    Claude Code keeps fields nanobot doesn't model (``subscriptionType``,
    ``scopes``, ``rateLimitTier``, …); those must survive a write-back or the
    ``claude`` CLI loses state.
    """
    try:
        blob = json.loads(raw)
    except Exception:
        return None
    oauth = blob.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    oauth["accessToken"] = token_data["access_token"]
    if token_data.get("refresh_token"):
        oauth["refreshToken"] = token_data["refresh_token"]
    # Claude Code stores expiresAt in milliseconds.
    oauth["expiresAt"] = int(token_data["expires_at"] * 1000)
    return json.dumps(blob)


def _persist_to_claude_code(token_data: dict[str, Any], source: str) -> bool:
    """Write refreshed tokens back to whichever Claude Code store they came from."""
    if source == SOURCE_KEYCHAIN:
        raw = _read_claude_code_keychain()
        merged = _merge_claude_code_credentials(raw, token_data) if raw else None
        return _write_claude_code_keychain(merged) if merged else False

    if source == SOURCE_CC_FILE:
        path = _claude_code_credentials_path()
        try:
            merged = _merge_claude_code_credentials(path.read_text(), token_data)
            if not merged:
                return False
            path.write_text(merged)
            return True
        except Exception:
            return False

    return False


def _parse_claude_code_credentials(raw: str, source: str) -> dict[str, Any] | None:
    """Parse Claude Code's credentials blob into our token dict shape."""
    try:
        oauth = (json.loads(raw).get("claudeAiOauth") or {})
    except Exception:
        return None
    access_token = oauth.get("accessToken")
    if not access_token:
        return None
    expires_at = oauth.get("expiresAt")
    # Claude Code stores expiresAt in milliseconds.
    return {
        "access_token": access_token,
        "refresh_token": oauth.get("refreshToken", ""),
        "expires_at": float(expires_at) / 1000.0 if expires_at else 0.0,
        "token_type": "Bearer",
        "source": source,
    }


def _load_claude_code_token() -> dict[str, Any] | None:
    """Load OAuth token from Claude Code's own credentials, if present.

    Claude Code stores ``{"claudeAiOauth": {"accessToken", "refreshToken",
    "expiresAt", ...}}`` with ``expiresAt`` in milliseconds — in
    ``~/.claude/.credentials.json`` on headless installs, and in the login
    Keychain on macOS. Sharing it means users who already ran ``claude`` login
    don't need a separate ``nanobot provider login claude-oauth``; refreshes
    are written back so both stay on the same token chain.
    """
    path = _claude_code_credentials_path()
    if path.exists():
        try:
            if token := _parse_claude_code_credentials(path.read_text(), SOURCE_CC_FILE):
                return token
        except Exception:
            pass
    raw = _read_claude_code_keychain()
    return _parse_claude_code_credentials(raw, SOURCE_KEYCHAIN) if raw else None


def _load_token() -> dict[str, Any] | None:
    """Load persisted OAuth token, falling back to Claude Code's credentials.

    A stored token whose refresh token has been revoked is worse than no token
    at all, so an expired-and-unrefreshable local token yields to Claude Code's
    credentials when those are still live.
    """
    fallback = None
    path = _token_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("access_token"):
                if time.time() < data.get("expires_at", 0) - _EXPIRY_SKEW_SECONDS:
                    return data
                fallback = data
        except Exception:
            pass

    claude_code = _load_claude_code_token()
    if claude_code and time.time() < claude_code.get("expires_at", 0) - _EXPIRY_SKEW_SECONDS:
        return claude_code
    # Nothing is currently valid — hand back whatever can still be refreshed.
    return fallback or claude_code


def _save_token(data: dict[str, Any]) -> None:
    """Persist OAuth token to disk."""
    _token_path().write_text(json.dumps(data, indent=2))


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    import base64
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def get_claude_oauth_login_status() -> dict[str, Any] | None:
    """Return the persisted Claude OAuth token if available."""
    return _load_token()


def _refresh_access_token(refresh_token: str, source: str | None = None) -> dict[str, Any]:
    """Exchange a refresh token for a new access token.

    *source* says which store the refresh token was read from. Anthropic
    rotates refresh tokens, so a token borrowed from Claude Code must be
    written back to Claude Code's own store — otherwise the rotation would
    silently invalidate the copy the ``claude`` CLI still holds.
    """
    timeout = httpx.Timeout(20.0, connect=20.0)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        resp = client.post(
            CLAUDE_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLAUDE_CLIENT_ID,
                "scope": CLAUDE_SCOPES,
            },
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            detail = resp.text[:500]
            try:
                detail = resp.json()
            except Exception:
                pass
            raise RuntimeError(f"Token refresh failed ({resp.status_code}): {detail}")
        payload = resp.json()

    now = time.time()
    expires_in = int(payload.get("expires_in", _DEFAULT_EXPIRES_IN))
    token_data = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", refresh_token),
        "expires_at": now + expires_in,
        "token_type": payload.get("token_type", "Bearer"),
    }

    if source in (SOURCE_KEYCHAIN, SOURCE_CC_FILE):
        if _persist_to_claude_code(token_data, source):
            return {**token_data, "source": source}
        # Write-back failed: keep the rotated token in nanobot's own store so
        # this process stays usable, and warn that `claude` may need a re-login.
        logger.warning(
            "Refreshed a Claude Code OAuth token but could not write it back to "
            "{}; the `claude` CLI may need to log in again.", source,
        )

    _save_token(token_data)
    return token_data


def login_claude_oauth(
    print_fn: Callable[[str], None] | None = None,
    prompt_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run OAuth PKCE flow for Claude via a local HTTP callback server."""
    import socket
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlencode, urlparse, parse_qs

    printer = print_fn or print

    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    # Find a free port for the callback server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    redirect_uri = f"http://localhost:{port}/callback"
    auth_code_result: dict[str, str | None] = {"code": None, "error": None}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if params.get("error"):
                auth_code_result["error"] = params["error"][0]
                body = b"Authentication failed. You can close this window."
            elif params.get("code"):
                returned_state = params.get("state", [None])[0]
                if returned_state != state:
                    auth_code_result["error"] = "state_mismatch"
                    body = b"State mismatch. Authentication failed."
                else:
                    auth_code_result["code"] = params["code"][0]
                    body = b"Authentication successful! You can close this window."
            else:
                auth_code_result["error"] = "no_code"
                body = b"No authorization code received."

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # Suppress HTTP server logs

    server = HTTPServer(("127.0.0.1", port), CallbackHandler)

    auth_params = urlencode({
        "response_type": "code",
        "client_id": CLAUDE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": CLAUDE_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{CLAUDE_AUTHORIZE_URL}?{auth_params}"

    printer(f"Opening browser for Claude authentication...")
    printer(f"If the browser doesn't open, visit:\n{auth_url}\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Wait for the callback (timeout 5 min)
    server.timeout = 300
    server_thread = threading.Thread(target=lambda: server.handle_request())
    server_thread.start()
    server_thread.join(timeout=310)
    server.server_close()

    if auth_code_result["error"]:
        raise RuntimeError(f"Claude OAuth failed: {auth_code_result['error']}")
    if not auth_code_result["code"]:
        raise RuntimeError("Claude OAuth timed out. No authorization code received.")

    # Exchange code for tokens (matches Claude Code's exact format)
    printer("Exchanging authorization code for tokens...")
    token_body = {
        "grant_type": "authorization_code",
        "code": auth_code_result["code"],
        "redirect_uri": redirect_uri,
        "client_id": CLAUDE_CLIENT_ID,
        "code_verifier": code_verifier,
        "state": state,
    }

    timeout = httpx.Timeout(15.0, connect=15.0)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        resp = client.post(
            CLAUDE_TOKEN_URL,
            json=token_body,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            detail = resp.text[:500]
            try:
                detail = resp.json()
            except Exception:
                pass
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {detail}")
        payload = resp.json()

    now = time.time()
    expires_in = int(payload.get("expires_in", _DEFAULT_EXPIRES_IN))
    token_data = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", ""),
        "expires_at": now + expires_in,
        "token_type": payload.get("token_type", "Bearer"),
    }
    _save_token(token_data)
    printer(f"Token saved to {_token_path()}")
    return token_data


class ClaudeOAuthProvider(AnthropicProvider):
    """Anthropic provider that authenticates via Claude Max OAuth subscription.

    Mimics Claude Code's authentication: OAuth token sent as Bearer auth
    with identifying headers (x-app, User-Agent, session ID).
    """

    # api.anthropic.com only accepts Pro/Max OAuth tokens when the first
    # system block is exactly this Claude Code identity string.
    _CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

    def _system_prefix(self) -> str | None:
        return self._CLAUDE_CODE_SYSTEM_PREFIX

    def __init__(self, default_model: str = "claude-sonnet-4-20250514"):
        # Strip claude-oauth/ prefix from model name
        if default_model.startswith("claude-oauth/") or default_model.startswith("claude_oauth/"):
            default_model = default_model.split("/", 1)[1]

        # Load token early so we can set authToken on the SDK client
        token_data = _load_token()
        initial_token = token_data["access_token"] if token_data else "placeholder"

        from anthropic import AsyncAnthropic

        self._session_id = secrets.token_hex(16)
        # Create client matching Claude Code's auth pattern
        self._client = AsyncAnthropic(
            auth_token=initial_token,
            base_url=CLAUDE_API_BASE,
            default_headers={
                "anthropic-beta": "oauth-2025-04-20",
                "x-app": "cli",
                "User-Agent": "claude-code/2.1.90 nanobot",
                "X-Claude-Code-Session-Id": self._session_id,
            },
        )

        # Skip AnthropicProvider.__init__'s client creation, just set attrs
        from nanobot.providers.base import LLMProvider
        LLMProvider.__init__(self, api_key=None, api_base=CLAUDE_API_BASE)
        self.default_model = default_model
        self.extra_headers = {}

        self._oauth_token: str | None = initial_token if token_data else None
        self._oauth_expires_at: float = token_data.get("expires_at", 0) if token_data else 0.0

    def _ensure_token(self) -> str:
        """Get a valid OAuth access token, refreshing if needed."""
        now = time.time()

        # Return cached token if still valid
        if self._oauth_token and now < self._oauth_expires_at - _EXPIRY_SKEW_SECONDS:
            return self._oauth_token

        token_data = _load_token()
        if not token_data:
            raise RuntimeError(
                "Claude OAuth not logged in. Run: nanobot provider login claude-oauth"
            )

        # Check if stored token is still valid
        expires_at = token_data.get("expires_at", 0)
        if now < expires_at - _EXPIRY_SKEW_SECONDS:
            self._oauth_token = token_data["access_token"]
            self._oauth_expires_at = expires_at
            return self._oauth_token

        # Try to refresh. A token borrowed from Claude Code is refreshed in
        # place: the rotated pair is written back to Claude Code's own store so
        # nanobot and the `claude` CLI stay on one shared token chain.
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                "Claude OAuth token expired and no refresh token available. "
                "Run: nanobot provider login claude-oauth"
            )

        logger.debug("Refreshing Claude OAuth token...")
        try:
            new_data = _refresh_access_token(refresh_token, token_data.get("source"))
            self._oauth_token = new_data["access_token"]
            self._oauth_expires_at = new_data["expires_at"]
            return self._oauth_token
        except Exception as e:
            raise RuntimeError(
                f"Failed to refresh Claude OAuth token: {e}. "
                "Run: nanobot provider login claude-oauth"
            ) from e

    @staticmethod
    def _strip_prefix(model: str) -> str:
        for prefix in ("claude-oauth/", "claude_oauth/", "anthropic/"):
            if model.startswith(prefix):
                return model[len(prefix):]
        return model

    def _inject_token(self) -> None:
        """Inject the current OAuth token into the Anthropic client."""
        token = self._ensure_token()
        self._client.auth_token = token

    async def _call_with_retry(self, method, **kwargs):
        """Call parent method with retry on 429 rate limit."""
        import asyncio
        max_retries = 5
        for attempt in range(max_retries):
            self._inject_token()
            result = await method(**kwargs)
            # Check if the response is a 429 rate limit error
            if result.finish_reason == "error" and result.content and "429" in result.content:
                wait = min(2 ** attempt * 5, 60)  # 5s, 10s, 20s, 40s, 60s
                logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
                continue
            return result
        return result  # Return last result even if still 429

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._call_with_retry(
            super().chat,
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        return await self._call_with_retry(
            super().chat_stream,
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )
