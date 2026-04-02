"""Claude OAuth provider — uses Claude Max subscription via OAuth PKCE flow."""

from __future__ import annotations

import hashlib
import json
import secrets
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


def _load_token() -> dict[str, Any] | None:
    """Load persisted OAuth token from disk."""
    path = _token_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("access_token"):
            return data
    except Exception:
        pass
    return None


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


def _refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access token."""
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

        # Try to refresh
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                "Claude OAuth token expired and no refresh token available. "
                "Run: nanobot provider login claude-oauth"
            )

        logger.debug("Refreshing Claude OAuth token...")
        try:
            new_data = _refresh_access_token(refresh_token)
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
