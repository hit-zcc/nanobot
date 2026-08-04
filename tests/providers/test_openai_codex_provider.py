import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.providers.base import ToolCallRequest
from nanobot.providers.codex_credentials import CodexCredentialError, CodexCredentials
from nanobot.providers.openai_codex_provider import (
    _CONTINUATION_CACHE_LIMIT,
    _CONTINUATION_CACHE_TTL_SECONDS,
    OpenAICodexProvider,
    _CodexHTTPError,
    _CodexResponseError,
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
async def test_continuation_cache_survives_ten_minute_tool_run(monkeypatch):
    assert _CONTINUATION_CACHE_TTL_SECONDS == 900.0
    now = 100.0
    marker = "long-running-tool-encrypted-reasoning"
    tool_call = ToolCallRequest(id="call_long|fc_long", name="exec", arguments={})
    request_bodies = []

    async def request(url, headers, body, on_content_delta=None):
        request_bodies.append(body)
        if len(request_bodies) == 1:
            return (
                "",
                [tool_call],
                "stop",
                [
                    {"id": "rs_long", "type": "reasoning", "encrypted_content": marker},
                    {
                        "id": "fc_long",
                        "type": "function_call",
                        "call_id": "call_long",
                        "name": "exec",
                        "arguments": "{}",
                    },
                ],
            )
        return "done", [], "stop", []

    monkeypatch.setattr(
        "nanobot.providers.openai_codex_provider.time.monotonic",
        lambda: now,
    )
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    initial_messages = [{"role": "user", "content": "inspect"}]

    await provider.chat(initial_messages)
    now += 600.0
    await provider.chat(
        [
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
    )

    assert marker in json.dumps(request_bodies[1]["input"])


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
async def test_timeout_is_safely_classified_and_retried(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    request_count = 0
    delays = []

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise httpx.ReadTimeout("synthetic-secret-timeout-detail")
        return "ok", [], "stop", []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", fake_sleep)

    result = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert request_count == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_response_failure_is_safely_logged_and_retried(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())
    request_count = 0
    delays = []
    warnings = []
    secret = "synthetic-secret-response-message"

    class FakeSSE:
        async def aiter_lines(self):
            event = {
                "type": "response.failed",
                "response": {
                    "id": "resp_safe_123",
                    "error": {
                        "code": "server_error",
                        "message": secret,
                    },
                },
            }
            yield f"data: {json.dumps(event)}"
            yield ""

    async def request(url, headers, body, on_content_delta=None):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return await _consume_sse(FakeSSE(), on_content_delta)
        return "ok", [], "stop", []

    async def fake_sleep(delay):
        delays.append(delay)

    def capture_warning(message, *args):
        warnings.append((message, args))

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    monkeypatch.setattr("nanobot.providers.openai_codex_provider.logger.warning", capture_warning)
    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", fake_sleep)

    result = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert request_count == 2
    assert delays == [1]
    assert warnings[0] == (
        "Codex response failed (event_type={}, code={}, request_id={})",
        ("response.failed", "server_error", "resp_safe_123"),
    )
    assert warnings[1] == (
        "LLM transient error (attempt {}/{}), retrying in {}s: {}",
        (
            1,
            3,
            1,
            "error calling codex: response failed",
        ),
    )
    assert secret not in repr(warnings)


@pytest.mark.asyncio
async def test_response_failure_discards_unsafe_diagnostic_fields():
    secret = "Bearer synthetic-secret"

    class FakeSSE:
        async def aiter_lines(self):
            event = {
                "type": "error",
                "request_id": secret,
                "error": {"code": secret, "message": secret},
            }
            yield f"data: {json.dumps(event)}"
            yield ""

    with pytest.raises(_CodexResponseError) as raised:
        await _consume_sse(FakeSSE())

    assert raised.value.event_type == "error"
    assert raised.value.code is None
    assert raised.value.request_id is None
    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_transport_error_is_safely_classified(monkeypatch):
    provider = OpenAICodexProvider(credential_manager=FakeCredentials())

    async def request(url, headers, body, on_content_delta=None):
        raise httpx.ConnectError("synthetic-secret-connection-detail")

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)

    result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling Codex: connection failed"
    assert "synthetic-secret" not in result.content


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

    timeout = seen_kwargs["timeout"]
    assert timeout.connect == 20.0
    assert timeout.read == 300.0
    assert timeout.write == 60.0
    assert timeout.pool == 20.0
