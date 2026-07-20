from __future__ import annotations

import asyncio
import base64
import binascii
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodexCredentialError("Codex auth state is unreadable. Run: codex login") from exc
        if not isinstance(payload, dict):
            raise CodexCredentialError("Codex auth state is unreadable. Run: codex login")
        if payload.get("auth_mode") != "chatgpt":
            raise CodexCredentialError("Codex must be logged in with ChatGPT. Run: codex login")
        tokens = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
        access = tokens.get("access_token")
        account = tokens.get("account_id")
        if not isinstance(access, str) or not access or not isinstance(account, str) or not account:
            raise CodexCredentialError("Codex ChatGPT credentials are incomplete. Run: codex login")
        return CodexCredentials(access_token=access, account_id=account)

    @staticmethod
    def _expires_at(access_token: str) -> int | None:
        try:
            encoded = access_token.split(".")[1]
            encoded += "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(encoded))
            if not isinstance(payload, dict):
                return None
            exp = payload.get("exp")
            return int(exp) if exp is not None else None
        except (
            binascii.Error,
            IndexError,
            OverflowError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return None

    def _expires_soon(self, credentials: CodexCredentials) -> bool:
        expires_at = self._expires_at(credentials.access_token)
        return expires_at is not None and expires_at <= time.time() + self.refresh_skew_seconds

    def certify_cli_available(self) -> str:
        executable = shutil.which(self.codex_executable or "codex")
        if not executable:
            raise CodexCredentialError("Official Codex CLI is not installed or not on PATH")
        return executable

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
            if (
                force_refresh
                and rejected_access_token
                and refreshed.access_token == rejected_access_token
            ):
                raise CodexCredentialError(
                    "Codex token refresh did not produce new credentials. Run: codex login"
                )
            return refreshed

    async def _refresh_with_codex(self) -> None:
        executable = self.codex_executable or shutil.which("codex")
        if not executable:
            raise CodexCredentialError("Official Codex CLI is not installed or not on PATH")
        try:
            proc = await asyncio.create_subprocess_exec(
                executable,
                "app-server",
                "--listen",
                "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ, "CODEX_HOME": str(self.codex_home)},
            )
        except (OSError, ValueError):
            raise CodexCredentialError(
                "Unable to start official Codex CLI. Verify it is installed and executable"
            ) from None
        try:
            await self._send(
                proc,
                {
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "clientInfo": {
                            "name": "nanobot",
                            "title": "nanobot",
                            "version": "0.1",
                        },
                    },
                },
            )
            await self._read_response(proc, 1)
            await self._send(proc, {"method": "initialized", "params": {}})
            await self._send(
                proc,
                {"method": "account/read", "id": 2, "params": {"refreshToken": True}},
            )
            response = await self._read_response(proc, 2)
            result = response["result"]
            account = result.get("account") or {}
            if not isinstance(account, dict):
                raise CodexCredentialError("Codex app-server returned a malformed response")
            if account.get("type") != "chatgpt":
                raise CodexCredentialError("Codex is not logged in with ChatGPT. Run: codex login")
        finally:
            if proc.stdin:
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        raise CodexCredentialError(
                            "Codex app-server did not stop after authentication refresh"
                        ) from None

    async def _send(self, proc, message: dict) -> None:
        if proc.stdin is None:
            raise CodexCredentialError("Codex app-server stdin is unavailable")
        proc.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
        await proc.stdin.drain()

    async def _read_response(self, proc, request_id: int) -> dict:
        if proc.stdout is None:
            raise CodexCredentialError("Codex app-server stdout is unavailable")
        deadline = time.monotonic() + self.rpc_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexCredentialError(
                    "Codex app-server authentication request timed out"
                )
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise CodexCredentialError(
                    "Codex app-server authentication request timed out"
                ) from exc
            if not line:
                raise CodexCredentialError(
                    "Codex app-server stopped during authentication refresh"
                )
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise CodexCredentialError("Codex app-server returned a malformed response") from None
            if not isinstance(message, dict):
                raise CodexCredentialError("Codex app-server returned a malformed response")
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise CodexCredentialError("Codex app-server rejected the authentication request")
            if not isinstance(message.get("result"), dict):
                raise CodexCredentialError("Codex app-server returned a malformed response")
            return message
