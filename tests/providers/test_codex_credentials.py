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


def test_status_uses_default_codex_home(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_auth(tmp_path / ".codex")

    assert CodexCredentialManager().status().account_id == "acct-1"


def test_status_rejects_missing_auth_file(tmp_path):
    with pytest.raises(CodexCredentialError, match="codex login"):
        CodexCredentialManager(codex_home=tmp_path).status()


def test_status_rejects_malformed_auth_json_without_leaking_payload(tmp_path):
    secret = "secret-refresh-token"
    (tmp_path / "auth.json").write_text(f'{{"refresh_token":"{secret}"')

    with pytest.raises(CodexCredentialError, match="unreadable") as exc:
        CodexCredentialManager(codex_home=tmp_path).status()

    assert secret not in str(exc.value)


def test_status_rejects_invalid_utf8_auth_without_leaking_bytes(tmp_path):
    (tmp_path / "auth.json").write_bytes(b'\xffsecret-auth-bytes')

    with pytest.raises(CodexCredentialError, match="unreadable") as exc:
        CodexCredentialManager(codex_home=tmp_path).status()

    assert "secret-auth-bytes" not in str(exc.value)


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


def test_expires_at_returns_none_when_exp_overflows_integer_conversion():
    body = base64.urlsafe_b64encode(json.dumps({"exp": 1e1000}).encode()).decode().rstrip("=")

    assert CodexCredentialManager._expires_at(f"header.{body}.signature") is None


@pytest.mark.asyncio
async def test_get_credentials_does_not_refresh_fresh_token(monkeypatch, tmp_path):
    _write_auth(tmp_path, access=_jwt(4_000_000_000))
    manager = CodexCredentialManager(codex_home=tmp_path)

    async def unexpected_refresh():
        raise AssertionError("refresh should not run")

    monkeypatch.setattr(manager, "_refresh_with_codex", unexpected_refresh)
    assert (await manager.get_credentials()).account_id == "acct-1"


@pytest.mark.asyncio
async def test_get_credentials_refreshes_expired_token(monkeypatch, tmp_path):
    _write_auth(tmp_path, access=_jwt(1))
    manager = CodexCredentialManager(codex_home=tmp_path)

    async def refresh():
        _write_auth(tmp_path, access=_jwt(4_000_000_000))

    monkeypatch.setattr(manager, "_refresh_with_codex", refresh)

    assert (await manager.get_credentials()).access_token == _jwt(4_000_000_000)


@pytest.mark.asyncio
async def test_force_refresh_rejects_unchanged_rejected_token(monkeypatch, tmp_path):
    _write_auth(tmp_path, access="rejected-secret")
    manager = CodexCredentialManager(codex_home=tmp_path)

    async def refresh():
        return None

    monkeypatch.setattr(manager, "_refresh_with_codex", refresh)

    with pytest.raises(CodexCredentialError, match="did not produce new credentials") as exc:
        await manager.get_credentials(
            force_refresh=True, rejected_access_token="rejected-secret"
        )

    assert "rejected-secret" not in str(exc.value)


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
        self.closed = False

    def write(self, data: bytes):
        self.lines.append(data.decode().strip())

    async def drain(self):
        return None

    def close(self):
        self.closed = True


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
    launch_kwargs = {}

    async def create(*args, **kwargs):
        launch_kwargs.update(kwargs)
        return await fake.create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    await manager._refresh_with_codex()
    sent = [json.loads(line) for line in fake.stdin_lines]
    assert sent[0]["method"] == "initialize"
    assert sent[1] == {"method": "initialized", "params": {}}
    assert sent[2]["method"] == "account/read"
    assert sent[2]["params"] == {"refreshToken": True}
    assert launch_kwargs["stderr"] is asyncio.subprocess.DEVNULL


def test_certify_cli_available_rejects_unresolvable_executable(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda executable: None)
    manager = CodexCredentialManager(codex_home=tmp_path)

    with pytest.raises(CodexCredentialError, match="installed.*PATH"):
        manager.certify_cli_available()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [42, "response", [], None])
async def test_refresh_rejects_scalar_rpc_response(monkeypatch, tmp_path, response):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([response])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="malformed"):
        await manager._refresh_with_codex()


@pytest.mark.asyncio
async def test_refresh_rejects_scalar_rpc_result(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess(
        [{"id": 1, "result": {}}, {"id": 2, "result": "secret-result"}]
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="malformed") as exc:
        await manager._refresh_with_codex()

    assert "secret-result" not in str(exc.value)


@pytest.mark.asyncio
async def test_refresh_sanitizes_rpc_error(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess(
        [{"id": 1, "error": {"message": "secret-server-detail"}}]
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="rejected") as exc:
        await manager._refresh_with_codex()

    assert "secret-server-detail" not in str(exc.value)


class _BlockingStdout:
    def __init__(self):
        self.started = asyncio.Event()

    async def readline(self):
        self.started.set()
        await asyncio.Event().wait()


class _RawStdout:
    def __init__(self, lines):
        self.lines = list(lines)

    async def readline(self):
        return self.lines.pop(0) if self.lines else b""


@pytest.mark.asyncio
async def test_refresh_rejects_malformed_json_rpc_message(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([])
    fake.stdout = _RawStdout([b'{"id":1,"result":"secret"\n'])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="malformed") as exc:
        await manager._refresh_with_codex()

    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_refresh_rejects_invalid_utf8_rpc_message(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([])
    fake.stdout = _RawStdout([b"\xffsecret-rpc-bytes\n"])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="malformed") as exc:
        await manager._refresh_with_codex()

    assert "secret-rpc-bytes" not in str(exc.value)


class _FakeMonotonicClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _AdvancingStdout:
    def __init__(self, clock):
        self.clock = clock
        self.next_id = 100
        self.remaining = 2

    async def readline(self):
        if not self.remaining:
            return b""
        self.remaining -= 1
        self.clock.now += 0.6
        self.next_id += 1
        return json.dumps({"id": self.next_id, "result": {}}).encode() + b"\n"


@pytest.mark.asyncio
async def test_refresh_uses_one_deadline_across_unmatched_messages(monkeypatch, tmp_path):
    manager = CodexCredentialManager(
        codex_home=tmp_path, codex_executable="codex-test", rpc_timeout_seconds=1.0
    )
    clock = _FakeMonotonicClock()
    fake = FakeAppServerProcess([])
    fake.stdout = _AdvancingStdout(clock)
    monkeypatch.setattr("nanobot.providers.codex_credentials.time.monotonic", clock.monotonic)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="timed out"):
        await manager._refresh_with_codex()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, False, 0, "", []])
async def test_refresh_rejects_response_with_falsey_error_member(
    monkeypatch, tmp_path, error
):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([{"id": 1, "error": error, "result": {}}])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="rejected"):
        await manager._refresh_with_codex()


@pytest.mark.asyncio
async def test_refresh_reports_early_app_server_exit(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="stopped"):
        await manager._refresh_with_codex()


@pytest.mark.asyncio
async def test_refresh_times_out_and_cleans_up_process(monkeypatch, tmp_path):
    manager = CodexCredentialManager(
        codex_home=tmp_path, codex_executable="codex-test", rpc_timeout_seconds=0.01
    )
    fake = FakeAppServerProcess([])
    fake.stdout = _BlockingStdout()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    with pytest.raises(CodexCredentialError, match="timed out"):
        await manager._refresh_with_codex()

    assert fake.stdin.closed is True
    assert fake.returncode == 0


@pytest.mark.asyncio
async def test_refresh_cancellation_cleans_up_process(monkeypatch, tmp_path):
    manager = CodexCredentialManager(codex_home=tmp_path, codex_executable="codex-test")
    fake = FakeAppServerProcess([])
    fake.stdout = _BlockingStdout()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake.create)

    task = asyncio.create_task(manager._refresh_with_codex())
    await fake.stdout.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake.stdin.closed is True
    assert fake.returncode == 0


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
