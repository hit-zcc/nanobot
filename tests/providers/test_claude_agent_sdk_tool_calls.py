"""One-shot tool calls on the Claude Agent SDK backend.

``ClaudeAgentSDKProvider.chat`` used to answer *any* request carrying tools with
``LLMResponse(content="Claude Agent SDK tool calls require ClaudeAgentSDKRunner.",
finish_reason="error")``.  Three callers deliberately bypass the agent runner and
ask the provider for a single forced tool call — heartbeat ``_decide``,
``MemoryStore.consolidate`` and ``evaluator.evaluate_response`` — so switching to
this backend broke all three.  In production that showed up as::

    Memory consolidation: LLM did not call save_memory (finish_reason=error,
    content_len=57, content_preview=Claude Agent SDK tool calls require ...)
    Memory consolidation degraded: raw-archived 132 messages
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider

_SAVE_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Persist the consolidation result.",
            "parameters": {
                "type": "object",
                "properties": {"history_entry": {"type": "string"}},
                "required": ["history_entry"],
            },
        },
    }
]

_FORCED = {"type": "function", "function": {"name": "save_memory"}}


@pytest.fixture
def fake_sdk(monkeypatch):
    """Stand in for the real claude_agent_sdk so no CLI/subscription is touched."""
    module = types.ModuleType("claude_agent_sdk")
    state: dict[str, Any] = {"reply": "", "options": None, "prompt": None}

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class AssistantMessage:
        def __init__(self, content) -> None:
            self.content = content

    class ResultMessage:
        def __init__(self, result: str) -> None:
            self.result = result
            self.usage = {"input_tokens": 11, "output_tokens": 7}
            self.is_error = False
            self.errors: list[str] = []
            self.subtype = ""

    def ClaudeAgentOptions(**kwargs):  # noqa: N802 - mirrors the SDK's name
        return types.SimpleNamespace(**kwargs)

    def query(*, prompt, options):
        state["prompt"], state["options"] = prompt, options

        async def _stream():
            yield AssistantMessage([TextBlock(state["reply"])])
            yield ResultMessage(state["reply"])

        return _stream()

    module.TextBlock = TextBlock
    module.AssistantMessage = AssistantMessage
    module.ResultMessage = ResultMessage
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return state


def _provider(tmp_path) -> ClaudeAgentSDKProvider:
    return ClaudeAgentSDKProvider("claude-agent-sdk/claude-opus-5", tmp_path)


# --- parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"name": "save_memory", "arguments": {"history_entry": "hello"}}',
        '```json\n{"name": "save_memory", "arguments": {"history_entry": "hello"}}\n```',
        'Sure!\n{"name": "save_memory", "arguments": {"history_entry": "hello"}}\nDone.',
        '{"name": "save_memory", "arguments": "{\\"history_entry\\": \\"hello\\"}"}',
        '{"history_entry": "hello"}',  # bare arguments, single candidate tool
    ],
)
def test_parse_tool_call_survives_the_usual_reply_shapes(reply):
    specs = ClaudeAgentSDKProvider._tool_specs(_SAVE_MEMORY_TOOLS, _FORCED)

    call = ClaudeAgentSDKProvider.parse_tool_call(reply, specs)

    assert call is not None
    assert call.name == "save_memory"
    assert call.arguments == {"history_entry": "hello"}


def test_parse_tool_call_rejects_a_reply_with_no_json():
    specs = ClaudeAgentSDKProvider._tool_specs(_SAVE_MEMORY_TOOLS, _FORCED)
    assert ClaudeAgentSDKProvider.parse_tool_call("I cannot do that.", specs) is None


def test_tool_choice_none_disables_emulation():
    assert ClaudeAgentSDKProvider._tool_specs(_SAVE_MEMORY_TOOLS, "none") == []


def test_forced_tool_choice_narrows_to_that_tool():
    tools = _SAVE_MEMORY_TOOLS + [
        {"type": "function", "function": {"name": "other", "parameters": {}}}
    ]
    specs = ClaudeAgentSDKProvider._tool_specs(tools, _FORCED)
    assert [s["name"] for s in specs] == ["save_memory"]


# --- chat() ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_with_tools_returns_a_tool_call_instead_of_an_error(fake_sdk, tmp_path):
    fake_sdk["reply"] = '{"name": "save_memory", "arguments": {"history_entry": "[2026-09-01] x"}}'

    response = await _provider(tmp_path).chat(
        messages=[{"role": "system", "content": "You are a memory consolidation agent."},
                  {"role": "user", "content": "consolidate this"}],
        tools=_SAVE_MEMORY_TOOLS,
        tool_choice=_FORCED,
    )

    assert response.finish_reason == "tool_calls"
    assert response.has_tool_calls
    assert response.tool_calls[0].name == "save_memory"
    assert response.tool_calls[0].arguments == {"history_entry": "[2026-09-01] x"}
    assert "Claude Agent SDK tool calls require" not in (response.content or "")
    # The schema has to reach the model, or it can only guess the shape.
    assert "save_memory" in fake_sdk["options"].system_prompt
    assert "history_entry" in fake_sdk["options"].system_prompt


@pytest.mark.asyncio
async def test_chat_without_tools_is_unchanged(fake_sdk, tmp_path):
    fake_sdk["reply"] = "plain answer"

    response = await _provider(tmp_path).chat(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.finish_reason == "stop"
    assert response.content == "plain answer"
    assert not response.has_tool_calls
    assert "Required output format" not in fake_sdk["options"].system_prompt


@pytest.mark.asyncio
async def test_unparsable_reply_degrades_to_text_not_to_a_fake_error(fake_sdk, tmp_path):
    fake_sdk["reply"] = "I refuse."

    response = await _provider(tmp_path).chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=_SAVE_MEMORY_TOOLS,
        tool_choice=_FORCED,
    )

    assert not response.has_tool_calls
    assert response.finish_reason == "stop"
    assert response.content == "I refuse."


# --- the actual production symptom ---------------------------------------


@pytest.mark.asyncio
async def test_memory_consolidation_no_longer_raw_archives_on_this_backend(fake_sdk, tmp_path):
    from nanobot.agent.memory import MemoryStore

    fake_sdk["reply"] = (
        '{"name": "save_memory", "arguments": {'
        '"history_entry": "[2026-09-01 18:06] talked about heartbeat holders", '
        '"new_facts": "- gateway PID 22179 started 16:11", '
        '"obsolete_facts": []}}'
    )
    store = MemoryStore(tmp_path)
    provider = _provider(tmp_path)

    ok = await store.consolidate(
        [{"role": "user", "content": "hi", "timestamp": "2026-09-01T18:06:00"}],
        provider,
        "claude-agent-sdk/claude-opus-5",
    )

    assert ok is True
    history = store.history_file.read_text(encoding="utf-8")
    assert "talked about heartbeat holders" in history
    assert "[RAW]" not in history  # the degraded path
    assert "gateway PID 22179" in store.read_long_term()
