import asyncio
import base64
import json
from pathlib import Path

import pytest

from nanobot.providers.codex_credentials import (
    CodexCredentialError,
    CodexCredentialManager,
)


def _write_auth(home: Path, *, mode="chatgpt", access="access-1", account="acct-1"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": mode,
                "tokens": {
                    "access_token": access,
                    "account_id": account,
                    "refresh_token": "must-never-appear",
                },
            }
        )
    )


def test_status_reads_chatgpt_credentials_from_explicit_codex_home(tmp_path):
    _write_auth(tmp_path)
    credentials = CodexCredentialManager(codex_home=tmp_path).status()
    assert credentials.access_token == "access-1"
    assert credentials.account_id == "acct-1"


def test_status_uses_codex_home_environment(monkeypatch, tmp_path):
    _write_auth(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert CodexCredentialManager().status().account_id == "acct-1"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "Run: codex login"),
        ({"auth_mode": "apikey", "tokens": {}}, "ChatGPT"),
        ({"auth_mode": "chatgpt", "tokens": {}}, "incomplete"),
    ],
)
def test_status_rejects_unusable_auth_without_leaking_payload(tmp_path, payload, message):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "auth.json").write_text(json.dumps(payload))
    with pytest.raises(CodexCredentialError, match=message) as exc:
        CodexCredentialManager(codex_home=tmp_path).status()
    assert "refresh_token" not in str(exc.value)


@pytest.mark.parametrize("payload", [[], "chatgpt", 1, None])
def test_status_rejects_non_object_auth_json(tmp_path, payload):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "auth.json").write_text(json.dumps(payload))

    with pytest.raises(CodexCredentialError, match="unreadable"):
        CodexCredentialManager(codex_home=tmp_path).status()


def _jwt(exp: int) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.mark.parametrize(
    "access_token",
    [
        "not-a-jwt",
        "header.a.signature",
        "header._w.signature",
        "header.W10.signature",
        "header.Im5vdC1hbi1vYmplY3Qi.signature",
    ],
)
def test_expires_at_returns_none_for_malformed_jwt(access_token):
    assert CodexCredentialManager._expires_at(access_token) is None


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


class _StubbornAppServerProcess(FakeAppServerProcess):
    def __init__(self, responses):
        super().__init__(responses)
        self.wait_calls = 0
        self.terminated = False
        self.killed = False

    async def wait(self):
        self.wait_calls += 1
        if self.wait_calls <= 2:
            raise asyncio.TimeoutError
        self.returncode = -9
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


@pytest.mark.asyncio
async def test_refresh_app_server_protocol(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess(
        [
            {"id": 1, "result": {"codexHome": str(tmp_path)}},
            {"id": 2, "result": {"account": {"type": "chatgpt"}}},
        ]
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)
    await manager._refresh_with_codex()
    sent = [json.loads(line) for line in fake.stdin_lines]
    assert sent[0]["method"] == "initialize"
    assert sent[1] == {"method": "initialized", "params": {}}
    assert sent[2]["method"] == "account/read"
    assert sent[2]["params"] == {"refreshToken": True}


@pytest.mark.asyncio
async def test_refresh_wraps_subprocess_launch_failure(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")

    async def fail_launch(*args, **kwargs):
        raise OSError("secret-launch-detail")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_launch)
    with pytest.raises(CodexCredentialError, match="installed and executable") as exc:
        await manager._refresh_with_codex()
    assert "secret-launch-detail" not in str(exc.value)


@pytest.mark.asyncio
async def test_refresh_kills_process_when_terminate_does_not_stop_it(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = _StubbornAppServerProcess(
        [
            {"id": 1, "result": {"codexHome": str(tmp_path)}},
            {"id": 2, "result": {"account": {"type": "chatgpt"}}},
        ]
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    await manager._refresh_with_codex()

    assert fake.terminated is True
    assert fake.killed is True
    assert fake.wait_calls == 3
