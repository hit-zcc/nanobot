import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.providers.base import ToolCallRequest
from nanobot.providers.codex_credentials import CodexCredentialError, CodexCredentials
from nanobot.providers.openai_codex_provider import (
    _CONTINUATION_CACHE_LIMIT,
    OpenAICodexProvider,
    _CodexHTTPError,
    _consume_sse,
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
async def test_fast_service_tier_sends_priority(monkeypatch):
    provider = OpenAICodexProvider(
        credential_manager=FakeCredentials(),
        service_tier="fast",
    )
    seen_body = None

    async def request(url, headers, body, on_content_delta=None):
        nonlocal seen_body
        seen_body = body
        return "ok", [], "stop", []

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert seen_body["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_omitted_service_tier_sends_no_field(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    seen_body = None

    async def request(url, headers, body, on_content_delta=None):
        nonlocal seen_body
        seen_body = body
        return "ok", [], "stop", []

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert "service_tier" not in seen_body


@pytest.mark.asyncio
async def test_unknown_service_tier_fails_before_transport(monkeypatch):
    provider = OpenAICodexProvider(
        credential_manager=FakeCredentials(),
        service_tier="turbo",
    )
    transport_called = False

    async def request(url, headers, body, on_content_delta=None):
        nonlocal transport_called
        transport_called = True
        return "ok", [], "stop", []

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling Codex: request failed"
    assert transport_called is False


@pytest.mark.asyncio
async def test_encrypted_reasoning_is_replayed_before_tool_continuation(monkeypatch):
    reasoning_1 = {
        "id": "rs_1",
        "type": "reasoning",
        "encrypted_content": "synthetic-encrypted-reasoning-1",
        "summary": [],
    }
    reasoning_2 = {
        "id": "rs_2",
        "type": "reasoning",
        "encrypted_content": "synthetic-encrypted-reasoning-2",
        "summary": [],
    }
    function_call_1 = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "list_dir",
        "arguments": '{"path":"."}',
        "status": "completed",
    }
    function_call_2 = {
        "id": "fc_2",
        "type": "function_call",
        "call_id": "call_2",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
        "status": "completed",
    }
    first_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "list_dir",
                "arguments": "",
            },
        },
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_2",
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": "",
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": reasoning_2,
        },
        {
            "type": "response.output_item.done",
            "output_index": 3,
            "item": function_call_2,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": reasoning_1,
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": function_call_1,
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    ]
    second_events = [
        {"type": "response.output_text.delta", "delta": "done"},
        {"type": "response.completed", "response": {"status": "completed"}},
    ]
    request_bodies = []

    class FakeSSE:
        def __init__(self, events):
            self.events = events

        async def aiter_lines(self):
            for event in self.events:
                yield f"data: {json.dumps(event)}"
                yield ""

    async def request(url, headers, body, on_content_delta=None):
        request_bodies.append(body)
        events = first_events if len(request_bodies) == 1 else second_events
        return await _consume_sse(FakeSSE(events), on_content_delta)

    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=lambda name, arguments: f"{name} result")

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "inspect"}],
            tools=tools,
            model="openai-codex/gpt-5.6-sol",
            max_iterations=2,
        )
    )

    assert result.final_content == "done"
    assert request_bodies[1]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
        reasoning_1,
        function_call_1,
        reasoning_2,
        function_call_2,
        {"type": "function_call_output", "call_id": "call_1", "output": "list_dir result"},
        {"type": "function_call_output", "call_id": "call_2", "output": "read_file result"},
    ]
    serialized_messages = json.dumps(result.messages)
    assert "synthetic-encrypted-reasoning" not in serialized_messages
    assert all("thinking_blocks" not in message for message in result.messages)
    assert "synthetic-encrypted-reasoning" not in repr(result)


@pytest.mark.asyncio
async def test_failed_status_keeps_continuation_for_next_request_then_evicts(monkeypatch):
    marker = "retry-only-encrypted-reasoning"
    reasoning = {"id": "rs_retry", "type": "reasoning", "encrypted_content": marker}
    function_call = {
        "id": "fc_retry",
        "type": "function_call",
        "call_id": "call_retry",
        "name": "list_dir",
        "arguments": "{}",
        "status": "completed",
    }
    tool_call = ToolCallRequest(
        id="call_retry|fc_retry",
        name="list_dir",
        arguments={},
    )
    request_bodies = []

    async def request(url, headers, body, on_content_delta=None):
        request_bodies.append(body)
        if len(request_bodies) == 1:
            return "", [tool_call], "stop", [reasoning, function_call]
        if len(request_bodies) == 2:
            return "", [], "error", []
        return "done", [], "stop", []

    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    initial_messages = [{"role": "user", "content": "inspect"}]
    continuation_messages = [
        *initial_messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call.to_openai_tool_call()],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": "tool result",
        },
    ]

    first = await provider.chat(initial_messages)
    failed = await provider.chat(continuation_messages)
    retried = await provider.chat(continuation_messages)
    after_success = await provider.chat(continuation_messages)

    assert first.thinking_blocks is None
    assert marker not in repr(first)
    assert failed.finish_reason == "error"
    assert retried.content == after_success.content == "done"
    assert marker in json.dumps(request_bodies[1]["input"])
    assert marker in json.dumps(request_bodies[2]["input"])
    assert marker not in json.dumps(request_bodies[3]["input"])


@pytest.mark.asyncio
async def test_continuation_cache_is_bounded(monkeypatch):
    request_count = 0

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        call_id = f"call_{request_count}"
        item_id = f"fc_{request_count}"
        return (
            "",
            [ToolCallRequest(id=f"{call_id}|{item_id}", name="tool", arguments={})],
            "stop",
            [
                {
                    "id": f"rs_{request_count}",
                    "type": "reasoning",
                    "encrypted_content": f"encrypted-{request_count}",
                },
                {
                    "id": item_id,
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "tool",
                    "arguments": "{}",
                },
            ],
        )

    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    for index in range(_CONTINUATION_CACHE_LIMIT + 1):
        await provider.chat([{"role": "user", "content": f"request {index}"}])

    assert len(provider._continuations) == _CONTINUATION_CACHE_LIMIT


@pytest.mark.asyncio
async def test_nested_forced_tool_choice_is_sent_in_flat_responses_shape(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    seen_body = None

    async def request(url, headers, body, on_content_delta=None):
        nonlocal seen_body
        seen_body = body
        return "ok", [], "stop", []

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
        return "ok", [], "stop", []

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
async def test_unexpected_transport_error_does_not_expose_secret_or_raw_detail(monkeypatch):
    secret = "synthetic-secret-header-value"
    raw_detail = "illegal header protocol detail"
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())

    async def request(url, headers, body, on_content_delta=None):
        raise RuntimeError(f"{raw_detail}: Bearer {secret}")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling Codex: request failed"
    assert secret not in result.content
    assert raw_detail not in result.content


@pytest.mark.asyncio
async def test_credential_error_message_is_not_reflected():
    secret = "credential-error-synthetic-secret"

    class MaliciousCredentials:
        async def get_credentials(self, **kwargs):
            raise CodexCredentialError(f"credential detail contains {secret}")

    provider = OpenAICodexProvider(credential_manager=MaliciousCredentials())

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling Codex: Codex authentication failed. Run: codex login"
    assert secret not in result.content


@pytest.mark.asyncio
async def test_429_does_not_refresh_credentials(monkeypatch):
    credentials = FakeCredentials()
    provider = OpenAICodexProvider(credential_manager=credentials)
    request_count = 0
    secret = "http-error-synthetic-secret"

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        raise _CodexHTTPError(429, f"untrusted response contains {secret}")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == (
        "Error calling Codex: ChatGPT usage quota exceeded or rate limit triggered. "
        "Please try again later."
    )
    assert secret not in result.content
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
