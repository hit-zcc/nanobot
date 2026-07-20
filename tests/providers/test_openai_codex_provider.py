import pytest

from nanobot.providers.codex_credentials import CodexCredentials
from nanobot.providers.openai_codex_provider import (
    OpenAICodexProvider,
    _CodexHTTPError,
    _request_codex,
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
async def test_nested_forced_tool_choice_is_sent_in_flat_responses_shape(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    seen_body = None

    async def request(url, headers, body, on_content_delta=None):
        nonlocal seen_body
        seen_body = body
        return "ok", [], "stop"

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    await provider.chat(
        messages=[{"role": "user", "content": "save it"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "save_memory"}},
    )

    assert seen_body["tool_choice"] == {"type": "function", "name": "save_memory"}


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


@pytest.mark.asyncio
async def test_429_does_not_refresh_credentials(monkeypatch):
    credentials = FakeCredentials()
    provider = OpenAICodexProvider(credential_manager=credentials)
    request_count = 0

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        raise _CodexHTTPError(429, "quota exceeded")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling Codex: quota exceeded"
    assert request_count == 1
    assert credentials.calls == [{}]


@pytest.mark.asyncio
async def test_second_401_stops_after_two_requests(monkeypatch):
    credentials = FakeCredentials()
    provider = OpenAICodexProvider(credential_manager=credentials)
    request_count = 0

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        raise _CodexHTTPError(401, "authentication failed")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert request_count == 2
    assert credentials.calls == [
        {},
        {"force_refresh": True, "rejected_access_token": "rejected"},
    ]


@pytest.mark.asyncio
async def test_request_codex_uses_default_tls_verification(monkeypatch):
    seen_kwargs = {}

    class StopRequestError(RuntimeError):
        pass

    class FakeClient:
        async def __aenter__(self):
            raise StopRequestError

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    def make_client(**kwargs):
        seen_kwargs.update(kwargs)
        return FakeClient()

    monkeypatch.setattr("nanobot.providers.openai_codex_provider.httpx.AsyncClient", make_client)

    with pytest.raises(StopRequestError):
        await _request_codex("https://example.test", {}, {})

    assert seen_kwargs == {"timeout": 60.0}
