"""A cancelled turn (/stop) must still persist the work already completed.

Regression: `_dispatch` used to swallow `CancelledError` after only logging, so
every message produced during the cancelled turn was dropped. The next user
message then reloaded the pre-turn history and the agent "started over".
"""

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


class TestSealDanglingToolCalls:
    def test_orphan_tool_call_gets_synthetic_result(self, tmp_path):
        loop = _make_loop(tmp_path)
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "exec"}},
                    {"id": "c2", "type": "function", "function": {"name": "read_file"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "ok"},
        ]

        sealed = loop._seal_dangling_tool_calls(messages)

        # c1 already answered, only c2 gets a synthetic result appended.
        assert len(sealed) == len(messages) + 1
        extra = sealed[-1]
        assert extra["role"] == "tool"
        assert extra["tool_call_id"] == "c2"
        assert extra["name"] == "read_file"
        assert "interrupted" in extra["content"]

        # Every declared tool call now has exactly one result -> legal history.
        declared = {
            tc["id"]
            for m in sealed if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
        }
        answered = [m["tool_call_id"] for m in sealed if m.get("role") == "tool"]
        assert declared == set(answered)
        assert len(answered) == len(set(answered))

    def test_fully_answered_history_is_untouched(self, tmp_path):
        loop = _make_loop(tmp_path)
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exec"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "ok"},
        ]
        assert loop._seal_dangling_tool_calls(messages) == messages


class TestCancelledTurnPersisted:
    @pytest.mark.asyncio
    async def test_stop_midtool_saves_partial_turn(self, tmp_path):
        """End-to-end: cancel while a tool runs, then verify the session kept it."""
        from nanobot.bus.events import InboundMessage
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)

        session = Session(key="test:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        saved: list[int] = []
        loop.sessions.save = MagicMock(side_effect=lambda s: saved.append(len(s.messages)))
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
        loop.context.build_messages = MagicMock(return_value=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do task"},
        ])
        loop.commands.dispatch = AsyncMock(return_value=None)
        loop.subagents.format_session_context = MagicMock(return_value=None)

        tool_started = asyncio.Event()

        async def chat_with_retry(**kwargs):
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={"command": "x"})],
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        async def never_finishing_tool(name, args):
            tool_started.set()
            await asyncio.sleep(60)

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(side_effect=never_finishing_tool)

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="do task")
        task = asyncio.create_task(loop._process_message(msg))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The in-flight turn survived the cancel.
        assert saved, "session was never saved after cancellation"
        roles = [m["role"] for m in session.messages]
        assert "assistant" in roles, f"assistant turn lost: {roles}"
        assert "tool" in roles, f"interrupted tool not sealed: {roles}"

        # Saved history is legal: the tool call has a matching result.
        declared = {
            tc["id"]
            for m in session.messages if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
        }
        answered = {m["tool_call_id"] for m in session.messages if m.get("role") == "tool"}
        assert declared == answered

    @pytest.mark.asyncio
    async def test_cancel_before_any_output_saves_nothing(self, tmp_path):
        """No completed work -> no partial write (avoid a lone user message)."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.save = MagicMock()

        loop._save_cancelled_turn(session, [{"role": "system"}, {"role": "user"}], 2, "test:c1")

        loop.sessions.save.assert_not_called()
        assert session.messages == []

    @pytest.mark.asyncio
    async def test_save_failure_does_not_mask_cancellation(self, tmp_path):
        """A broken save must not raise over the CancelledError being propagated."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.save = MagicMock(side_effect=RuntimeError("disk on fire"))

        loop._save_cancelled_turn(
            session,
            [{"role": "system"}, {"role": "user"}, {"role": "assistant", "content": "partial"}],
            2,
            "test:c1",
        )  # must not raise


class TestCancelledTurnClosesStream:
    """A stopped turn must also close whatever surface it was streaming into.

    Regression: `_dispatch` re-raised the cancellation without ever ending the
    stream. Feishu keeps one card per turn and only lets go of it on a stream
    end, so a stopped turn left its card open forever — and every later answer
    in that chat was written into it instead of a card of its own. Once Feishu
    timed that card's stream out, those answers stopped arriving at all.
    """

    @pytest.mark.asyncio
    async def test_cancelled_turn_emits_a_final_stream_end(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop = _make_loop(tmp_path)
        loop._process_message = AsyncMock(side_effect=asyncio.CancelledError)

        msg = InboundMessage(
            channel="feishu", sender_id="u1", chat_id="c1", content="do task",
            metadata={"_wants_stream": True},
        )
        with pytest.raises(asyncio.CancelledError):
            await loop._dispatch(msg)

        published = []
        while not loop.bus.outbound.empty():
            published.append(loop.bus.outbound.get_nowait())
        ends = [m for m in published if m.metadata.get("_stream_end")]
        assert ends, "the cancelled turn never closed its stream"
        assert ends[-1].metadata["_resuming"] is False, "a stopped turn is not resuming"

    @pytest.mark.asyncio
    async def test_a_failed_close_does_not_mask_the_cancellation(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop = _make_loop(tmp_path)
        loop._process_message = AsyncMock(side_effect=asyncio.CancelledError)
        loop.bus.publish_outbound = AsyncMock(side_effect=RuntimeError("bus is gone"))

        msg = InboundMessage(
            channel="feishu", sender_id="u1", chat_id="c1", content="do task",
            metadata={"_wants_stream": True},
        )
        with pytest.raises(asyncio.CancelledError):
            await loop._dispatch(msg)


class TestRunnerMessageSink:
    @pytest.mark.asyncio
    async def test_sink_receives_live_messages(self):
        from nanobot.agent.runner import AgentRunner, AgentRunSpec

        provider = MagicMock()
        sink: list[dict] = []
        seen_midrun: list[int] = []

        async def chat_with_retry(**kwargs):
            seen_midrun.append(len(sink))
            return LLMResponse(content="done", tool_calls=[], usage={})

        provider.chat_with_retry = chat_with_retry
        tools = MagicMock()
        tools.get_definitions.return_value = []

        result = await AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            model="m",
            max_iterations=3,
            message_sink=sink,
        ))

        # Sink was populated before the run returned, and is the same list.
        assert seen_midrun == [1]
        assert result.messages is sink
        assert [m["role"] for m in sink] == ["user", "assistant"]
