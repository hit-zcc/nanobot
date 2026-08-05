"""Progress must survive a process that dies without unwinding.

`os.execv` (restart), SIGKILL and crashes run no `finally`, so a turn saved only
at its end is lost whole — including the messages describing the work just done.
Checkpointing at iteration boundaries bounds the loss to the current step.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)
    return loop


def _wire(loop, session):
    loop.sessions.get_or_create = MagicMock(return_value=session)
    loop.sessions.save = MagicMock()
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    loop.commands.dispatch = AsyncMock(return_value=None)
    loop.subagents.format_session_context = MagicMock(return_value=None)


def _msg(content="go"):
    return InboundMessage(channel="test", sender_id="u1", chat_id="c1", content=content)


class TestCheckpointCursor:
    def test_advances_and_is_idempotent(self, tmp_path):
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.save = MagicMock()

        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "step one"},
        ]

        cursor = loop._checkpoint_turn(session, messages, 2, "test:c1")
        assert cursor == 3
        assert len(session.messages) == 1

        # Nothing new -> no duplicate write.
        again = loop._checkpoint_turn(session, messages, cursor, "test:c1")
        assert again == 3
        assert len(session.messages) == 1

    def test_failure_keeps_cursor_so_work_is_retried(self, tmp_path):
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.save = MagicMock(side_effect=OSError("disk full"))

        messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "x"}]
        cursor = loop._checkpoint_turn(session, messages, 1, "test:c1")

        assert cursor == 1, "cursor advanced despite a failed save"


class TestMidRunPersistence:
    @pytest.mark.asyncio
    async def test_work_is_on_disk_before_the_turn_ends(self, tmp_path):
        """Simulates a restart: inspect the session while the turn still runs."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        _wire(loop, session)

        seen_midrun: dict[str, list] = {}
        state = {"n": 0}
        gate = asyncio.Event()

        async def chat_with_retry(**kwargs):
            state["n"] += 1
            if state["n"] == 1:
                return LLMResponse(
                    content="calling a tool",
                    tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                    usage={},
                )
            # Second call: the first iteration is fully behind us. Snapshot what
            # a freshly restarted process would load from disk right now.
            seen_midrun["roles"] = [m["role"] for m in session.messages]
            seen_midrun["contents"] = [str(m.get("content")) for m in session.messages]
            await gate.wait()
            return LLMResponse(content="all done", tool_calls=[], usage={})

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(return_value="tool output")

        task = asyncio.create_task(loop._process_message(_msg("do it")))
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.wait_for(task, timeout=5)

        assert "assistant" in seen_midrun["roles"], "first iteration was not persisted mid-run"
        assert "tool" in seen_midrun["roles"]
        assert any("tool output" in c for c in seen_midrun["contents"])

    @pytest.mark.asyncio
    async def test_no_duplicate_messages_after_completion(self, tmp_path):
        """Checkpoints plus the final save must not double-write the turn."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        _wire(loop, session)

        state = {"n": 0}

        async def chat_with_retry(**kwargs):
            state["n"] += 1
            if state["n"] <= 2:
                return LLMResponse(
                    content=f"step {state['n']}",
                    tool_calls=[ToolCallRequest(id=f"c{state['n']}", name="exec", arguments={})],
                    usage={},
                )
            return LLMResponse(content="final answer", tool_calls=[], usage={})

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(return_value="out")

        await asyncio.wait_for(loop._process_message(_msg("go")), timeout=5)

        contents = [str(m.get("content")) for m in session.messages]
        for expected in ("step 1", "step 2", "final answer"):
            assert contents.count(expected) == 1, f"{expected!r} written {contents.count(expected)}x"

        tool_ids = [m["tool_call_id"] for m in session.messages if m["role"] == "tool"]
        assert len(tool_ids) == len(set(tool_ids)), "duplicate tool results persisted"

    @pytest.mark.asyncio
    async def test_cancel_after_checkpoint_does_not_duplicate(self, tmp_path):
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        _wire(loop, session)

        state = {"n": 0}
        tool_started = asyncio.Event()

        async def chat_with_retry(**kwargs):
            state["n"] += 1
            return LLMResponse(
                content=f"step {state['n']}",
                tool_calls=[ToolCallRequest(id=f"c{state['n']}", name="exec", arguments={})],
                usage={},
            )

        async def tool(name, args):
            if state["n"] >= 2:
                tool_started.set()
                await asyncio.sleep(60)
            return "out"

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(side_effect=tool)

        task = asyncio.create_task(loop._process_message(_msg("go")))
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        contents = [str(m.get("content")) for m in session.messages]
        assert contents.count("step 1") == 1
        assert contents.count("step 2") == 1

        declared = {
            tc["id"]
            for m in session.messages if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
        }
        answered = [m["tool_call_id"] for m in session.messages if m["role"] == "tool"]
        assert declared == set(answered)
        assert len(answered) == len(set(answered))


class TestRestartFlush:
    @pytest.mark.asyncio
    async def test_restart_flushes_cached_sessions_before_execv(self, tmp_path):
        """execv runs no finally handler, so /restart must persist first."""
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        s1, s2 = Session(key="test:a"), Session(key="test:b")
        loop.sessions._cache = {"test:a": s1, "test:b": s2}
        saved: list[str] = []
        loop.sessions.save = MagicMock(side_effect=lambda s: saved.append(s.key))

        order: list[str] = []
        msg = _msg("/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("nanobot.command.builtin.os.execv",
                   side_effect=lambda *a: order.append("execv")):
            out = await cmd_restart(ctx)
            await asyncio.sleep(1.2)

        assert "Restarting" in out.content
        assert sorted(saved) == ["test:a", "test:b"], "sessions not flushed before restart"
        assert order == ["execv"]

    @pytest.mark.asyncio
    async def test_broken_session_does_not_block_restart(self, tmp_path):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        loop.sessions._cache = {"test:a": Session(key="test:a")}
        loop.sessions.save = MagicMock(side_effect=OSError("disk full"))

        executed: list[str] = []
        msg = _msg("/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("nanobot.command.builtin.os.execv",
                   side_effect=lambda *a: executed.append("execv")):
            await cmd_restart(ctx)
            await asyncio.sleep(1.2)

        assert executed == ["execv"], "a failing flush blocked the restart"
