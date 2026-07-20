# Codex OAuth Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the nanobot gateway to `openai-codex/gpt-5.6-sol` while reusing the official Codex CLI ChatGPT login from `~/.codex/auth.json`.

**Architecture:** A focused credential manager reads short-lived request credentials from Codex state and delegates refresh to the official `codex app-server` `account/read` RPC. `OpenAICodexProvider` keeps its current Responses/SSE/tool adapter, retries one 401 after a coordinated refresh, and never owns or copies a refresh token.

**Tech Stack:** Python 3.11+, asyncio subprocesses, JSON-RPC over JSONL, httpx streaming, Pydantic configuration, Typer CLI, pytest/pytest-asyncio.

## Global Constraints

- Reuse `$CODEX_HOME/auth.json` when `CODEX_HOME` is set, otherwise `~/.codex/auth.json`.
- Require Codex `auth_mode` to be `chatgpt`; do not silently use an API-key login.
- Never copy or log OpenAI access tokens, refresh tokens, authorization headers, or raw `auth.json` content.
- The official Codex CLI remains the only writer and refresh-token owner.
- Do not disable TLS verification.
- Refresh at most once for a rejected access token and retry a failed request at most once.
- Default model: `openai-codex/gpt-5.6-sol`.
- Default reasoning effort in the deployed gateway: `medium`.
- Preserve Claude configuration and credentials for configuration-only rollback.
- Do not stage or overwrite unrelated existing worktree changes.

## File Map

- Create `nanobot/providers/codex_credentials.py`: Codex auth-file parsing, JWT expiry inspection, refresh serialization, and app-server JSON-RPC client.
- Create `tests/providers/test_codex_credentials.py`: credential parsing, expiry, refresh protocol, concurrency, and redaction tests.
- Modify `nanobot/providers/openai_codex_provider.py`: credential-manager integration, 401 refresh/retry, current default model, and strict TLS.
- Create `tests/providers/test_openai_codex_provider.py`: provider request retry, error, streaming, and current-model behavior.
- Modify `nanobot/cli/commands.py`: make `provider login openai-codex` inspect/delegate to the official Codex CLI instead of `oauth-cli-kit`.
- Modify `tests/cli/test_commands.py`: Codex login/status behavior and current model assertions.
- Modify `pyproject.toml`: remove `oauth-cli-kit` after its last import is removed.
- Modify `README.md`: document official Codex login reuse and `gpt-5.6-sol`.
- Modify `~/.nanobot/config.json` during rollout only: select `openai_codex`, `gpt-5.6-sol`, and `medium` reasoning.

---

### Task 1: Codex Credential Manager

**Files:**
- Create: `nanobot/providers/codex_credentials.py`
- Create: `tests/providers/test_codex_credentials.py`

**Interfaces:**
- Produces: `CodexCredentials(access_token: str, account_id: str)`.
- Produces: `CodexCredentialError(RuntimeError)` with credential-safe messages.
- Produces: `CodexCredentialManager.get_credentials(*, force_refresh: bool = False, rejected_access_token: str | None = None) -> CodexCredentials`.
- Produces: `CodexCredentialManager.status() -> CodexCredentials` for synchronous CLI validation without refresh.
- Depends on: official `codex app-server` JSONL protocol: `initialize`, `initialized`, then `account/read` with `refreshToken: true`.

- [ ] **Step 1: Write failing path and parsing tests**

```python
# tests/providers/test_codex_credentials.py
import json
from pathlib import Path

import pytest

from nanobot.providers.codex_credentials import (
    CodexCredentialError,
    CodexCredentialManager,
)


def _write_auth(home: Path, *, mode="chatgpt", access="access-1", account="acct-1"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "auth_mode": mode,
        "tokens": {
            "access_token": access,
            "account_id": account,
            "refresh_token": "must-never-appear",
        },
    }))


def test_status_reads_chatgpt_credentials_from_explicit_codex_home(tmp_path):
    _write_auth(tmp_path)
    credentials = CodexCredentialManager(codex_home=tmp_path).status()
    assert credentials.access_token == "access-1"
    assert credentials.account_id == "acct-1"


def test_status_uses_codex_home_environment(monkeypatch, tmp_path):
    _write_auth(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert CodexCredentialManager().status().account_id == "acct-1"


@pytest.mark.parametrize("payload, message", [
    ({}, "Run: codex login"),
    ({"auth_mode": "apikey", "tokens": {}}, "ChatGPT"),
    ({"auth_mode": "chatgpt", "tokens": {}}, "incomplete"),
])
def test_status_rejects_unusable_auth_without_leaking_payload(tmp_path, payload, message):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "auth.json").write_text(json.dumps(payload))
    with pytest.raises(CodexCredentialError, match=message) as exc:
        CodexCredentialManager(codex_home=tmp_path).status()
    assert "refresh_token" not in str(exc.value)
```

- [ ] **Step 2: Run the parsing tests and verify they fail**

Run: `pytest -q tests/providers/test_codex_credentials.py`

Expected: collection fails with `ModuleNotFoundError: nanobot.providers.codex_credentials`.

- [ ] **Step 3: Implement credential types, path resolution, and safe parsing**

```python
# nanobot/providers/codex_credentials.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


class CodexCredentialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CodexCredentials:
    access_token: str
    account_id: str


class CodexCredentialManager:
    def __init__(
        self,
        codex_home: Path | None = None,
        codex_executable: str | None = None,
        refresh_skew_seconds: int = 120,
        rpc_timeout_seconds: float = 20.0,
    ):
        configured = os.environ.get("CODEX_HOME")
        self.codex_home = Path(codex_home or configured or (Path.home() / ".codex")).expanduser()
        self.auth_path = self.codex_home / "auth.json"
        self.codex_executable = codex_executable
        self.refresh_skew_seconds = refresh_skew_seconds
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self._refresh_lock = asyncio.Lock()

    def status(self) -> CodexCredentials:
        try:
            payload = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CodexCredentialError("Codex ChatGPT login not found. Run: codex login") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexCredentialError("Codex auth state is unreadable. Run: codex login") from exc
        if payload.get("auth_mode") != "chatgpt":
            raise CodexCredentialError("Codex must be logged in with ChatGPT. Run: codex login")
        tokens = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
        access = tokens.get("access_token")
        account = tokens.get("account_id")
        if not isinstance(access, str) or not access or not isinstance(account, str) or not account:
            raise CodexCredentialError("Codex ChatGPT credentials are incomplete. Run: codex login")
        return CodexCredentials(access_token=access, account_id=account)
```

- [ ] **Step 4: Run parsing tests and verify they pass**

Run: `pytest -q tests/providers/test_codex_credentials.py`

Expected: all parsing/path tests pass.

- [ ] **Step 5: Add failing expiry, app-server, and refresh-coalescing tests**

```python
# append to tests/providers/test_codex_credentials.py
import asyncio
import base64


def _jwt(exp: int) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.mark.asyncio
async def test_get_credentials_does_not_refresh_fresh_token(monkeypatch, tmp_path):
    _write_auth(tmp_path, access=_jwt(4_000_000_000))
    manager = CodexCredentialManager(codex_home=tmp_path)
    async def unexpected_refresh():
        raise AssertionError("refresh should not run")
    monkeypatch.setattr(manager, "_refresh_with_codex", unexpected_refresh)
    assert (await manager.get_credentials()).account_id == "acct-1"


@pytest.mark.asyncio
async def test_force_refresh_uses_app_server_once_for_rejected_token(monkeypatch, tmp_path):
    _write_auth(tmp_path, access="rejected")
    manager = CodexCredentialManager(codex_home=tmp_path)
    calls = 0
    async def refresh():
        nonlocal calls
        calls += 1
        _write_auth(tmp_path, access="fresh")
    monkeypatch.setattr(manager, "_refresh_with_codex", refresh)
    first, second = await asyncio.gather(
        manager.get_credentials(force_refresh=True, rejected_access_token="rejected"),
        manager.get_credentials(force_refresh=True, rejected_access_token="rejected"),
    )
    assert first.access_token == second.access_token == "fresh"
    assert calls == 1


class _FakeStdin:
    def __init__(self, lines):
        self.lines = lines
    def write(self, data: bytes):
        self.lines.append(data.decode().strip())
    async def drain(self):
        return None
    def close(self):
        return None


class _FakeStdout:
    def __init__(self, responses):
        self.responses = [json.dumps(item).encode() + b"\n" for item in responses]
    async def readline(self):
        return self.responses.pop(0) if self.responses else b""


class FakeAppServerProcess:
    def __init__(self, responses):
        self.stdin_lines = []
        self.stdin = _FakeStdin(self.stdin_lines)
        self.stdout = _FakeStdout(responses)
        self.stderr = _FakeStdout([])
        self.returncode = None
    async def create(self, *args, **kwargs):
        return self
    async def wait(self):
        self.returncode = 0
        return 0
    def terminate(self):
        self.returncode = 0
    def kill(self):
        self.returncode = -9


@pytest.mark.asyncio
async def test_refresh_app_server_protocol(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([
        {"id": 1, "result": {"codexHome": str(tmp_path)}},
        {"id": 2, "result": {"account": {"type": "chatgpt"}}},
    ])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)
    await manager._refresh_with_codex()
    sent = [json.loads(line) for line in fake.stdin_lines]
    assert sent[0]["method"] == "initialize"
    assert sent[1] == {"method": "initialized", "params": {}}
    assert sent[2]["method"] == "account/read"
    assert sent[2]["params"] == {"refreshToken": True}
```

- [ ] **Step 6: Run refresh tests and verify they fail**

Run: `pytest -q tests/providers/test_codex_credentials.py`

Expected: failures show `get_credentials` and `_refresh_with_codex` are missing.

- [ ] **Step 7: Implement expiry inspection and coordinated app-server refresh**

```python
# add inside CodexCredentialManager
    @staticmethod
    def _expires_at(access_token: str) -> int | None:
        try:
            encoded = access_token.split(".")[1]
            encoded += "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(encoded))
            exp = payload.get("exp")
            return int(exp) if exp is not None else None
        except (IndexError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _expires_soon(self, credentials: CodexCredentials) -> bool:
        expires_at = self._expires_at(credentials.access_token)
        return expires_at is not None and expires_at <= time.time() + self.refresh_skew_seconds

    async def get_credentials(
        self,
        *,
        force_refresh: bool = False,
        rejected_access_token: str | None = None,
    ) -> CodexCredentials:
        current = self.status()
        if not force_refresh and not self._expires_soon(current):
            return current
        async with self._refresh_lock:
            current = self.status()
            if force_refresh and rejected_access_token and current.access_token != rejected_access_token:
                return current
            if not force_refresh and not self._expires_soon(current):
                return current
            await self._refresh_with_codex()
            refreshed = self.status()
            if force_refresh and rejected_access_token and refreshed.access_token == rejected_access_token:
                raise CodexCredentialError("Codex token refresh did not produce new credentials. Run: codex login")
            return refreshed

    async def _refresh_with_codex(self) -> None:
        executable = self.codex_executable or shutil.which("codex")
        if not executable:
            raise CodexCredentialError("Official Codex CLI is not installed or not on PATH")
        proc = await asyncio.create_subprocess_exec(
            executable, "app-server", "--listen", "stdio://",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "CODEX_HOME": str(self.codex_home)},
        )
        try:
            await self._send(proc, {"method": "initialize", "id": 1, "params": {
                "clientInfo": {"name": "nanobot", "title": "nanobot", "version": "0.1"},
            }})
            await self._read_response(proc, 1)
            await self._send(proc, {"method": "initialized", "params": {}})
            await self._send(proc, {"method": "account/read", "id": 2, "params": {"refreshToken": True}})
            response = await self._read_response(proc, 2)
            account = (response.get("result") or {}).get("account") or {}
            if account.get("type") != "chatgpt":
                raise CodexCredentialError("Codex is not logged in with ChatGPT. Run: codex login")
        finally:
            if proc.stdin:
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.terminate()
                await proc.wait()

    async def _send(self, proc, message: dict) -> None:
        if proc.stdin is None:
            raise CodexCredentialError("Codex app-server stdin is unavailable")
        proc.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
        await proc.stdin.drain()

    async def _read_response(self, proc, request_id: int) -> dict:
        if proc.stdout is None:
            raise CodexCredentialError("Codex app-server stdout is unavailable")
        while True:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=self.rpc_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise CodexCredentialError("Codex app-server authentication request timed out") from exc
            if not line:
                raise CodexCredentialError("Codex app-server stopped during authentication refresh")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                raise CodexCredentialError("Codex app-server rejected the authentication request")
            return message
```

- [ ] **Step 8: Run credential tests**

Run: `pytest -q tests/providers/test_codex_credentials.py`

Expected: all credential tests pass, including one refresh for concurrent rejected-token callers.

- [ ] **Step 9: Commit the credential manager**

```bash
git add nanobot/providers/codex_credentials.py tests/providers/test_codex_credentials.py
git commit -m "feat(codex): reuse official CLI credentials"
```

---

### Task 2: Provider Refresh/Retry and Current Model

**Files:**
- Modify: `nanobot/providers/openai_codex_provider.py`
- Create: `tests/providers/test_openai_codex_provider.py`
- Modify: `tests/cli/test_commands.py`

**Interfaces:**
- Consumes: `CodexCredentialManager.get_credentials(force_refresh=False, rejected_access_token=None)`.
- Produces: `OpenAICodexProvider(default_model="openai-codex/gpt-5.6-sol", credential_manager=None)`.
- Produces: `_CodexHTTPError.status_code` for provider-local HTTP decisions.

- [ ] **Step 1: Write failing default-model and 401-retry tests**

```python
# tests/providers/test_openai_codex_provider.py
import pytest

from nanobot.providers.codex_credentials import CodexCredentials
from nanobot.providers.openai_codex_provider import (
    OpenAICodexProvider,
    _CodexHTTPError,
)


class FakeCredentials:
    def __init__(self):
        self.calls = []
    async def get_credentials(self, **kwargs):
        self.calls.append(kwargs)
        token = "fresh" if kwargs.get("force_refresh") else "rejected"
        return CodexCredentials(token, "acct")


def test_default_model_is_current_sol_model():
    assert OpenAICodexProvider().get_default_model() == "openai-codex/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_401_refreshes_rejected_token_and_retries_once(monkeypatch):
    credentials = FakeCredentials()
    provider = OpenAICodexProvider(credential_manager=credentials)
    seen_headers = []
    async def request(url, headers, body, on_content_delta=None):
        seen_headers.append(headers)
        if len(seen_headers) == 1:
            raise _CodexHTTPError(401, "authentication failed")
        return "ok", [], "stop"
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert credentials.calls == [
        {},
        {"force_refresh": True, "rejected_access_token": "rejected"},
    ]
    assert seen_headers[0]["Authorization"] == "Bearer rejected"
    assert seen_headers[1]["Authorization"] == "Bearer fresh"
```

Also add tests asserting 429 does not refresh, a second 401 stops after two requests,
and `_request_codex` creates `httpx.AsyncClient` with TLS verification enabled.

- [ ] **Step 2: Run provider tests and verify they fail**

Run: `pytest -q tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py -k 'codex'`

Expected: failures show the old default, old manager-free constructor, and missing `_CodexHTTPError`.

- [ ] **Step 3: Integrate the credential manager and one-shot 401 recovery**

```python
# core shape in nanobot/providers/openai_codex_provider.py
class _CodexHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class OpenAICodexProvider(LLMProvider):
    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.6-sol",
        credential_manager: CodexCredentialManager | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.credentials = credential_manager or CodexCredentialManager()

    async def _request_with_auth(self, body, on_content_delta=None):
        credentials = await self.credentials.get_credentials()
        for attempt in range(2):
            try:
                return await _request_codex(
                    DEFAULT_CODEX_URL,
                    _build_headers(credentials.account_id, credentials.access_token),
                    body,
                    on_content_delta=on_content_delta,
                )
            except _CodexHTTPError as exc:
                if exc.status_code != 401 or attempt == 1:
                    raise
                credentials = await self.credentials.get_credentials(
                    force_refresh=True,
                    rejected_access_token=credentials.access_token,
                )
        raise AssertionError("unreachable")
```

Change `_request_codex` to omit the `verify` argument, always use
`httpx.AsyncClient(timeout=60.0)`, and raise `_CodexHTTPError(status, safe_message)`.
Remove the certificate-error retry with `verify=False`. Keep 429 mapped to the
existing quota-specific friendly error. Ensure caught exceptions become
`LLMResponse(finish_reason="error")` without authorization headers or raw tokens.

- [ ] **Step 4: Run provider-focused tests**

Run: `pytest -q tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py -k 'codex'`

Expected: all selected tests pass.

- [ ] **Step 5: Run existing message/tool conversion tests**

Run: `pytest -q tests/cli/test_commands.py tests/providers/test_providers_init.py`

Expected: all tests pass; existing prefix stripping and lazy imports remain intact.

- [ ] **Step 6: Commit provider changes**

```bash
git add nanobot/providers/openai_codex_provider.py tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py
git commit -m "fix(codex): refresh CLI auth on unauthorized responses"
```

---

### Task 3: Official Codex CLI Login UX and Dependency Cleanup

**Files:**
- Modify: `nanobot/cli/commands.py`
- Modify: `tests/cli/test_commands.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `CodexCredentialManager.status()`.
- Produces: `nanobot provider login openai-codex` as a status/delegation command for official Codex login.

- [ ] **Step 1: Write failing CLI tests**

```python
# append to tests/cli/test_commands.py
def test_openai_codex_login_reports_existing_chatgpt_login(monkeypatch):
    monkeypatch.setattr(
        "nanobot.providers.codex_credentials.CodexCredentialManager.status",
        lambda self: CodexCredentials("hidden-token", "acct-123"),
    )
    result = runner.invoke(app, ["provider", "login", "openai-codex"])
    assert result.exit_code == 0
    assert "Authenticated with OpenAI Codex" in result.output
    assert "hidden-token" not in result.output


def test_openai_codex_login_points_to_official_cli_when_missing(monkeypatch):
    def missing(self):
        raise CodexCredentialError("Codex ChatGPT login not found. Run: codex login")
    monkeypatch.setattr(
        "nanobot.providers.codex_credentials.CodexCredentialManager.status", missing,
    )
    result = runner.invoke(app, ["provider", "login", "openai-codex"])
    assert result.exit_code == 1
    assert "codex login" in result.output
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: `pytest -q tests/cli/test_commands.py -k 'openai_codex_login'`

Expected: tests fail because the handler still imports `oauth_cli_kit`.

- [ ] **Step 3: Replace the independent OAuth flow with official-login inspection**

```python
@_register_login("openai_codex")
def _login_openai_codex() -> None:
    from nanobot.providers.codex_credentials import (
        CodexCredentialError,
        CodexCredentialManager,
    )
    try:
        credentials = CodexCredentialManager().status()
    except CodexCredentialError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        "[green]✓ Authenticated with OpenAI Codex via official Codex CLI[/green]  "
        f"[dim]{credentials.account_id}[/dim]"
    )
```

Do not display `credentials.access_token`. Keep actual interactive sign-in owned by
the official `codex login` command so nanobot never receives a refresh token.

- [ ] **Step 4: Remove the obsolete package dependency**

Delete `"oauth-cli-kit>=0.1.3,<1.0.0",` from `pyproject.toml`, then confirm no imports remain:

Run: `rg -n "oauth_cli_kit|oauth-cli-kit" nanobot pyproject.toml`

Expected: no output.

- [ ] **Step 5: Run CLI and import tests**

Run: `pytest -q tests/cli/test_commands.py tests/providers/test_providers_init.py`

Expected: all tests pass.

- [ ] **Step 6: Commit CLI/dependency changes**

```bash
git add nanobot/cli/commands.py tests/cli/test_commands.py pyproject.toml
git commit -m "refactor(codex): delegate login to official CLI"
```

---

### Task 4: Documentation and Configuration Defaults

**Files:**
- Modify: `README.md`
- Modify: `nanobot/providers/openai_codex_provider.py` only if a default assertion was missed in Task 2.
- Modify: `tests/cli/test_commands.py`

**Interfaces:**
- Documents: official `codex login` prerequisite and reuse of `~/.codex/auth.json`.
- Documents: `provider=openai_codex`, `model=openai-codex/gpt-5.6-sol`, `reasoningEffort=medium`.

- [ ] **Step 1: Add a failing documentation regression test**

```python
def test_readme_codex_example_uses_current_model_and_official_login():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "codex login" in readme
    assert '"model": "openai-codex/gpt-5.6-sol"' in readme
    assert '"provider": "openai_codex"' in readme
    assert '"reasoningEffort": "medium"' in readme
    assert "openai-codex/gpt-5.1-codex" not in readme
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run: `pytest -q tests/cli/test_commands.py -k 'readme_codex_example'`

Expected: failure because README still uses `gpt-5.1-codex`.

- [ ] **Step 3: Update README Codex OAuth instructions**

Replace the existing Codex section with this operational sequence:

```bash
codex login
codex login status
nanobot provider login openai-codex
```

Use this configuration example:

```json
{
  "agents": {
    "defaults": {
      "model": "openai-codex/gpt-5.6-sol",
      "provider": "openai_codex",
      "reasoningEffort": "medium"
    }
  }
}
```

State that credentials remain in `~/.codex/auth.json`, are refreshed by the official
Codex CLI, and are not copied into nanobot configuration.

- [ ] **Step 4: Run documentation and focused tests**

Run: `pytest -q tests/cli/test_commands.py -k 'codex or readme'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/cli/test_commands.py
git commit -m "docs(codex): use current subscription model"
```

---

### Task 5: Full Verification and Deployment Switch

**Files:**
- Modify outside repository: `~/.nanobot/config.json`
- Preserve backup: `~/.nanobot/config.json.codex-backup-20260720`
- Runtime log: `~/.nanobot/gateway.log`

**Interfaces:**
- Consumes: completed credential manager/provider/CLI implementation.
- Produces: running gateway using `openai-codex/gpt-5.6-sol` at reasoning `medium`.

- [ ] **Step 1: Run format and focused test gates**

Run:

```bash
ruff check nanobot/providers/codex_credentials.py nanobot/providers/openai_codex_provider.py nanobot/cli/commands.py tests/providers/test_codex_credentials.py tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py
pytest -q tests/providers/test_codex_credentials.py tests/providers/test_openai_codex_provider.py tests/providers/test_providers_init.py tests/cli/test_commands.py
```

Expected: Ruff exits 0 and all focused tests pass.

- [ ] **Step 2: Run broader provider and agent regressions**

Run: `pytest -q tests/providers tests/agent tests/channels/test_channel_manager_delta_coalescing.py tests/channels/test_feishu_streaming.py`

Expected: all tests pass. Any pre-existing unrelated failure is recorded with its exact test name and rerun independently before proceeding.

- [ ] **Step 3: Verify official login and perform a direct smoke test**

Run:

```bash
codex login status
nanobot provider login openai-codex
nanobot agent -m "Reply with exactly: CODEX_OK"
```

Expected: ChatGPT login is reported, no token is printed, and the final response contains `CODEX_OK`.

- [ ] **Step 4: Back up and update deployed configuration**

Create the backup only if it does not already exist. Update only these fields in
`~/.nanobot/config.json`:

```json
"model": "openai-codex/gpt-5.6-sol",
"provider": "openai_codex",
"reasoningEffort": "medium"
```

After editing, validate without printing secrets:

```bash
jq '.agents.defaults | {model, provider, reasoningEffort}' ~/.nanobot/config.json
```

Expected: exactly the three values above.

- [ ] **Step 5: Install and restart with the existing deployment workflow**

Run: `~/.nanobot/workspace/scripts/restart-gateway.sh`

Expected: editable install succeeds, the previous gateway exits, and a new gateway PID is reported.

- [ ] **Step 6: Verify gateway startup without exposing message content or credentials**

Run:

```bash
tail -n 200 ~/.nanobot/gateway.log | rg "Starting nanobot|Channels enabled|Heartbeat|ERROR|Traceback|Codex"
```

Expected: current startup, both configured Feishu channels, and heartbeat initialization appear with no new traceback/authentication error.

- [ ] **Step 7: Run one Feishu tool-call smoke test**

Send a benign request through Feishu that requires one read-only tool, such as asking
the bot to report the current workspace directory. Confirm the streaming card
completes, the final answer is delivered, and no duplicate response is sent.

- [ ] **Step 8: Roll back on deployment failure**

If Steps 3, 5, 6, or 7 fail because of Codex authentication/protocol compatibility,
restore only the prior `agents.defaults` values from the backup and rerun the restart
script. Do not modify `~/.codex/auth.json` and do not remove Claude credentials.

- [ ] **Step 9: Commit any verification-only test adjustment, otherwise leave no new commit**

If verification required a legitimate repository test correction, stage only its
named files and commit with `test(codex): cover deployed OAuth flow`. Otherwise,
report the prior task commits and leave the user's unrelated worktree changes untouched.
