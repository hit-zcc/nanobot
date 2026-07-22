"""Tests for the shared agent runner and its integration contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop


@pytest.mark.asyncio
async def test_runner_preserves_reasoning_fields_and_tool_results():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                reasoning_content="hidden reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                usage={"prompt_tokens": 5, "completion_tokens": 3},
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    assert result.final_content == "done"
    assert result.tools_used == ["list_dir"]
    assert result.tool_events == [
        {"name": "list_dir", "status": "ok", "detail": "tool result"}
    ]

    assistant_messages = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
    assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "tool result"
        for msg in captured_second_call
    )


@pytest.mark.asyncio
async def test_runner_calls_hooks_in_order():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    call_count = {"n": 0}
    events: list[tuple] = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    class RecordingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            events.append(("before_iteration", context.iteration))

        async def before_execute_tools(self, context: AgentHookContext) -> None:
            events.append((
                "before_execute_tools",
                context.iteration,
                [tc.name for tc in context.tool_calls],
            ))

        async def after_iteration(self, context: AgentHookContext) -> None:
            events.append((
                "after_iteration",
                context.iteration,
                context.final_content,
                list(context.tool_results),
                list(context.tool_events),
                context.stop_reason,
            ))

        def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
            events.append(("finalize_content", context.iteration, content))
            return content.upper() if content else content

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        hook=RecordingHook(),
    ))

    assert result.final_content == "DONE"
    assert events == [
        ("before_iteration", 0),
        ("before_execute_tools", 0, ["list_dir"]),
        (
            "after_iteration",
            0,
            None,
            ["tool result"],
            [{"name": "list_dir", "status": "ok", "detail": "tool result"}],
            None,
        ),
        ("before_iteration", 1),
        ("finalize_content", 1, "done"),
        ("after_iteration", 1, "DONE", [], [], "completed"),
    ]


@pytest.mark.asyncio
async def test_runner_streaming_hook_receives_deltas_and_end_signal():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    streamed: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("he")
        await on_content_delta("llo")
        return LLMResponse(content="hello", tool_calls=[], usage={})

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            streamed.append(delta)

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        hook=StreamingHook(),
    ))

    assert result.final_content == "hello"
    assert streamed == ["he", "llo"]
    assert endings == [False]
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_returns_max_iterations_fallback():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="still working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
    ))

    assert result.stop_reason == "max_iterations"
    assert result.final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_runner_surfaces_refusal_message_when_content_empty():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[],
        finish_reason="refusal",
        usage={},
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        refusal_message="blocked, try again differently",
    ))

    assert result.stop_reason == "refusal"
    assert result.final_content == "blocked, try again differently"


@pytest.mark.asyncio
async def test_runner_refusal_auto_downgrades_to_fallback_model():
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    models_used: list[str] = []
    notices: list[str] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, model, **kwargs):
        models_used.append(model)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(content=None, tool_calls=[], finish_reason="refusal", usage={})
        return LLMResponse(content="recovered answer", tool_calls=[], finish_reason="stop", usage={})

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    class NoticeHook(AgentHook):
        async def on_notice(self, context: AgentHookContext, message: str) -> None:
            notices.append(message)

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="primary-model",
        max_iterations=3,
        hook=NoticeHook(),
        refusal_fallback_model="fallback-model",
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "recovered answer"
    assert models_used == ["primary-model", "fallback-model"]
    assert len(notices) == 1 and "fallback-model" in notices[0]


@pytest.mark.asyncio
async def test_runner_refusal_surfaces_message_when_fallback_also_refuses():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    models_used: list[str] = []

    async def chat_with_retry(*, model, **kwargs):
        models_used.append(model)
        return LLMResponse(content=None, tool_calls=[], finish_reason="refusal", usage={})

    provider = MagicMock()
    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="primary-model",
        max_iterations=3,
        refusal_message="blocked for good",
        refusal_fallback_model="fallback-model",
    ))

    assert result.stop_reason == "refusal"
    assert result.final_content == "blocked for good"
    # Downgrades exactly once, then gives up rather than looping forever.
    assert models_used == ["primary-model", "fallback-model"]


@pytest.mark.asyncio
async def test_runner_refusal_keeps_partial_content_when_present():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="here is the partial answer before it stopped",
        tool_calls=[],
        finish_reason="refusal",
        usage={},
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=3,
        refusal_message="blocked",
    ))

    assert result.stop_reason == "refusal"
    assert result.final_content == "here is the partial answer before it stopped"


@pytest.mark.asyncio
async def test_runner_heartbeats_while_a_slow_tool_runs(monkeypatch):
    from nanobot.agent.hook import AgentHook
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_FIRST", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_MAX", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_GROWTH", 1.0)

    beats: list[tuple[float, list[str]]] = []

    class _Hook(AgentHook):
        async def on_tool_heartbeat(self, context, *, elapsed, pending):
            beats.append((elapsed, list(pending)))

    provider = MagicMock()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry

    async def slow_tool(*args, **kwargs):
        await asyncio.sleep(0.2)
        return "ok"

    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = slow_tool

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "go"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        hook=_Hook(),
    ))

    assert result.final_content == "done"
    assert beats, "a slow tool must emit liveness notices"
    assert all(pending == ["exec"] for _, pending in beats)
    assert beats[0][0] > 0


@pytest.mark.asyncio
async def test_runner_stays_silent_for_fast_tools():
    """The grace period must keep quick tools from announcing themselves."""
    from nanobot.agent.hook import AgentHook
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    beats: list[float] = []

    class _Hook(AgentHook):
        async def on_tool_heartbeat(self, context, *, elapsed, pending):
            beats.append(elapsed)

    provider = MagicMock()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="instant")

    result = await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "go"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        hook=_Hook(),
    ))

    assert result.final_content == "done"
    assert beats == []


@pytest.mark.asyncio
async def test_loop_tool_heartbeat_reaches_progress_channel(tmp_path, monkeypatch):
    """Heartbeats must ride the plain progress channel, not the tool_hint one."""
    from nanobot.agent.runner import AgentRunner

    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_FIRST", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_MAX", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_GROWTH", 1.0)

    loop = _make_loop(tmp_path)
    progress: list[tuple[str, bool]] = []
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={"command": "sleep"})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    loop.provider.chat_with_retry = chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])

    async def slow_tool(*args, **kwargs):
        await asyncio.sleep(0.2)
        return "ok"

    loop.tools.execute = slow_tool

    async def on_progress(msg, *, tool_hint=False):
        progress.append((msg, tool_hint))

    final_content, _, _ = await loop._run_agent_loop([], on_progress=on_progress)

    assert final_content == "done"
    beats = [m for m, hint in progress if not hint and "已用时" in m]
    assert beats, f"expected a heartbeat on the plain progress channel, got {progress}"
    assert "执行命令" in beats[0]


def test_tool_progress_formatting(tmp_path):
    loop = _make_loop(tmp_path)

    assert loop._format_duration(20) == "20 秒"
    assert loop._format_duration(263) == "4 分 23 秒"
    assert loop._format_duration(3661) == "1 小时 1 分 1 秒"
    assert loop._format_tool_names(["exec", "read_file"]) == ["执行命令", "读取文件"]
    assert loop._format_tool_names(["custom_tool"]) == ["`custom_tool`"]


@pytest.mark.asyncio
async def test_loop_tool_progress_has_one_id_and_completes(tmp_path, monkeypatch):
    from nanobot.agent.runner import AgentRunner

    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_FIRST", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_MAX", 0.01)
    monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_GROWTH", 1.0)

    loop = _make_loop(tmp_path)
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    loop.provider.chat_with_retry = chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])

    async def slow_tool(*args, **kwargs):
        await asyncio.sleep(0.05)
        return "ok"

    loop.tools.execute = slow_tool
    events: list[tuple[str, str, bool]] = []

    async def on_tool_progress(content: str, *, progress_id: str, done: bool) -> None:
        events.append((content, progress_id, done))

    final_content, _, _ = await loop._run_agent_loop(
        [], on_tool_progress=on_tool_progress,
    )

    assert final_content == "done"
    assert events[0][2] is False
    assert events[-1][2] is True
    assert events[0][1]
    assert {progress_id for _, progress_id, _ in events} == {events[0][1]}
    assert events[0][0].startswith("⏳ 正在执行命令 · 已用时")
    assert events[-1][0].startswith("✅ 执行命令完成 · 共用时")


def test_loop_refusal_fallback_model_derivation(tmp_path):
    loop = _make_loop(tmp_path)
    loop.model = "claude-oauth/claude-fable-5"
    assert loop._refusal_fallback_model() == "claude-oauth/claude-opus-4-8"
    loop.model = "claude-fable-5"
    assert loop._refusal_fallback_model() == "claude-opus-4-8"
    # Already on Opus 4.8 → no fallback.
    loop.model = "claude-oauth/claude-opus-4-8"
    assert loop._refusal_fallback_model() is None
    # Non-Claude provider → no fallback.
    loop.model = "azure/gpt-4o"
    assert loop._refusal_fallback_model() is None


@pytest.mark.asyncio
async def test_loop_refusal_auto_downgrades_and_notifies(tmp_path):
    loop = _make_loop(tmp_path)
    loop.model = "claude-oauth/claude-fable-5"
    models_used: list[str] = []
    notices: list[str] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, model, **kwargs):
        models_used.append(model)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(content=None, tool_calls=[], finish_reason="refusal", usage={})
        return LLMResponse(content="recovered", tool_calls=[], finish_reason="stop", usage={})

    loop.provider.chat_with_retry = chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])

    async def on_progress(msg, *, tool_hint=False):
        notices.append(msg)

    final_content, _, _ = await loop._run_agent_loop([], on_progress=on_progress)

    assert final_content == "recovered"
    assert models_used == ["claude-oauth/claude-fable-5", "claude-oauth/claude-opus-4-8"]
    assert any("claude-opus-4-8" in n for n in notices)


@pytest.mark.asyncio
async def test_loop_refusal_returns_chinese_message(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[],
        finish_reason="refusal",
        usage={},
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])

    final_content, _, _ = await loop._run_agent_loop([])

    assert "安全限制" in final_content
    assert final_content != "I've completed processing but have no response to give."


@pytest.mark.asyncio
async def test_runner_returns_structured_tool_error():
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=RuntimeError("boom"))

    runner = AgentRunner(provider)

    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert result.error == "Error: RuntimeError: boom"
    assert result.tool_events == [
        {"name": "list_dir", "status": "error", "detail": "boom"}
    ]


@pytest.mark.asyncio
async def test_loop_max_iterations_message_stays_stable(tmp_path):
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    loop.max_iterations = 2

    final_content, _, _ = await loop._run_agent_loop([])

    assert final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )


@pytest.mark.asyncio
async def test_loop_stream_filter_handles_think_only_prefix_without_crashing(tmp_path):
    loop = _make_loop(tmp_path)
    deltas: list[str] = []
    endings: list[bool] = []

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("<think>hidden")
        await on_content_delta("</think>Hello")
        return LLMResponse(content="<think>hidden</think>Hello", tool_calls=[], usage={})

    loop.provider.chat_stream_with_retry = chat_stream_with_retry

    async def on_stream(delta: str) -> None:
        deltas.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        endings.append(resuming)

    final_content, _, _ = await loop._run_agent_loop(
        [],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert final_content == "Hello"
    assert deltas == ["Hello"]
    assert endings == [False]


@pytest.mark.asyncio
async def test_subagent_max_iterations_announces_failure(tmp_path, monkeypatch):
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
    ))
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus, max_iterations=2)
    mgr._announce_result = AsyncMock()

    async def fake_execute(self, name, arguments):
        return "tool result"

    monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

    await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

    mgr._announce_result.assert_awaited_once()
    args = mgr._announce_result.await_args.args
    assert args[3] == (
        "Task did not complete before reaching the maximum number of "
        "tool call iterations (2)."
    )
    assert args[5] == "error"


@pytest.mark.asyncio
async def test_runner_skips_tool_calls_truncated_by_max_tokens():
    """finish_reason=length + tool_calls => do NOT execute (args may be
    silently truncated, e.g. write_file losing `content`); feed an error
    back so the model retries with smaller payloads."""
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    provider = MagicMock()
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Truncated mid-stream: content argument was dropped by
            # partial-JSON parsing, finish_reason is "length".
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="call_trunc", name="write_file",
                    arguments={"path": "big.md"},
                )],
                finish_reason="length",
                usage={"prompt_tokens": 5, "completion_tokens": 8192},
            )
        captured_second_call[:] = messages
        return LLMResponse(content="recovered", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="should not run")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "write a huge file"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
    ))

    # The truncated call must never reach tool execution.
    tools.execute.assert_not_awaited()
    assert result.final_content == "recovered"
    assert result.tools_used == []

    # The model got the assistant tool_call message plus an error tool result.
    tool_msgs = [m for m in captured_second_call if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_trunc"
    assert "max_tokens" in tool_msgs[0]["content"]
    assert "NOT executed" in tool_msgs[0]["content"]
