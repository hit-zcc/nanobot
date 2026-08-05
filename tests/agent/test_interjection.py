"""Mid-run interjections: a message sent while a turn runs must reach that turn.

Without this, anything typed during a long tool run waited for the whole turn to
finish, so corrections ("stop, wrong file") only landed after the wrong work was
already done.
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


def _msg(content="hi", **kw):
    return InboundMessage(channel="test", sender_id="u1", chat_id="c1", content=content, **kw)


class TestRouting:
    def test_message_during_active_run_is_queued_not_dispatched(self, tmp_path):
        loop = _make_loop(tmp_path)
        running = MagicMock()
        running.done.return_value = False
        loop._active_tasks["test:c1"] = [running]

        m = _msg("actually use the other file")
        assert loop._has_active_run(m) is True
        loop._queue_injection(m)

        assert [x.content for x in loop._pending_injections["test:c1"]] == [
            "actually use the other file"
        ]

    def test_idle_session_is_not_treated_as_active(self, tmp_path):
        loop = _make_loop(tmp_path)
        finished = MagicMock()
        finished.done.return_value = True
        loop._active_tasks["test:c1"] = [finished]

        assert loop._has_active_run(_msg()) is False

    def test_take_is_destructive(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._queue_injection(_msg("one"))
        assert len(loop._take_injections("test:c1")) == 1
        assert loop._take_injections("test:c1") == []


class TestBuildInjectionMessages:
    def test_multiple_messages_merge_with_steering_prefix(self, tmp_path):
        loop = _make_loop(tmp_path)
        built = loop._build_injection_messages([_msg("wait"), _msg("use utf-8")])

        assert len(built) == 1
        assert built[0]["role"] == "user"
        content = built[0]["content"]
        assert "wait\nuse utf-8" in content
        assert "不要重新开始已完成的工作" in content

    def test_empty_messages_produce_nothing(self, tmp_path):
        loop = _make_loop(tmp_path)
        assert loop._build_injection_messages([]) == []
        assert loop._build_injection_messages([_msg("   ")]) == []


class TestSplicedIntoRun:
    @pytest.mark.asyncio
    async def test_interjection_reaches_model_on_next_iteration(self, tmp_path):
        """End-to-end: queue while the tool runs, assert the next call sees it."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)

        session = Session(key="test:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
        loop.commands.dispatch = AsyncMock(return_value=None)
        loop.subagents.format_session_context = MagicMock(return_value=None)

        calls: list[list[dict]] = []
        tool_started = asyncio.Event()
        release_tool = asyncio.Event()

        async def chat_with_retry(*, messages, **kwargs):
            calls.append([dict(m) for m in messages])
            if len(calls) == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                    usage={},
                )
            return LLMResponse(content="done", tool_calls=[], usage={})

        async def slow_tool(name, args):
            tool_started.set()
            await release_tool.wait()
            return "tool output"

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(side_effect=slow_tool)

        task = asyncio.create_task(loop._process_message(_msg("do the task")))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        # User types while the tool is still running.
        loop._queue_injection(_msg("wait — use the other file"))
        release_tool.set()
        await asyncio.wait_for(task, timeout=5)

        assert len(calls) == 2, "model was not called again after the tool"
        second = calls[1]
        assert second[-1]["role"] == "user"
        assert "use the other file" in second[-1]["content"]
        # It lands after the tool result, keeping the exchange legal.
        assert second[-2]["role"] == "tool"
        # And the queue is drained, so it is not replayed as a second turn.
        assert loop._pending_injections.get("test:c1") in (None, [])

    @pytest.mark.asyncio
    async def test_interjection_is_persisted_in_history(self, tmp_path):
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
        loop.commands.dispatch = AsyncMock(return_value=None)
        loop.subagents.format_session_context = MagicMock(return_value=None)

        state = {"n": 0}

        async def chat_with_retry(**kwargs):
            state["n"] += 1
            if state["n"] == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                    usage={},
                )
            return LLMResponse(content="done", tool_calls=[], usage={})

        async def tool(name, args):
            loop._queue_injection(_msg("one more thing"))
            return "out"

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(side_effect=tool)

        await asyncio.wait_for(loop._process_message(_msg("go")), timeout=5)

        users = [m for m in session.messages if m["role"] == "user"]
        assert any("one more thing" in str(m["content"]) for m in users)


class TestLifecycle:
    def test_unconsumed_injection_is_replayed_as_new_turn(self, tmp_path):
        loop = _make_loop(tmp_path)
        spawned: list[InboundMessage] = []
        loop._spawn_dispatch = lambda m: spawned.append(m)

        loop._queue_injection(_msg("a"))
        loop._queue_injection(_msg("b"))
        loop._replay_unconsumed_injections("test:c1")

        assert len(spawned) == 1
        assert spawned[0].content == "a\nb"
        assert loop._pending_injections.get("test:c1") in (None, [])

    def test_cancelled_task_discards_injections_instead_of_replaying(self, tmp_path):
        """/stop must actually stop; queued text must not silently restart work."""
        loop = _make_loop(tmp_path)
        spawned: list[InboundMessage] = []
        loop._spawn_dispatch = lambda m: spawned.append(m)
        loop._queue_injection(_msg("queued during the run"))

        task = MagicMock()
        task.cancelled.return_value = True
        loop._active_tasks["test:c1"] = [task]
        loop._on_dispatch_done("test:c1", task)

        assert spawned == []
        assert loop._pending_injections.get("test:c1") in (None, [])

    def test_finished_task_replays_last_moment_injection(self, tmp_path):
        """Covers the race where a message lands after the final boundary check."""
        loop = _make_loop(tmp_path)
        spawned: list[InboundMessage] = []
        loop._spawn_dispatch = lambda m: spawned.append(m)
        loop._queue_injection(_msg("just missed the boundary"))

        task = MagicMock()
        task.cancelled.return_value = False
        loop._active_tasks["test:c1"] = [task]
        loop._on_dispatch_done("test:c1", task)

        assert [m.content for m in spawned] == ["just missed the boundary"]

    def test_other_still_running_task_keeps_injection_queued(self, tmp_path):
        loop = _make_loop(tmp_path)
        spawned: list[InboundMessage] = []
        loop._spawn_dispatch = lambda m: spawned.append(m)
        loop._queue_injection(_msg("for the other run"))

        done, still_running = MagicMock(), MagicMock()
        done.cancelled.return_value = False
        still_running.done.return_value = False
        loop._active_tasks["test:c1"] = [done, still_running]
        loop._on_dispatch_done("test:c1", done)

        assert spawned == []
        assert len(loop._pending_injections["test:c1"]) == 1

    @pytest.mark.asyncio
    async def test_cancelled_run_persists_orphaned_injection(self, tmp_path):
        """Text typed during a run that then got stopped must not vanish."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="test:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
        loop.commands.dispatch = AsyncMock(return_value=None)
        loop.subagents.format_session_context = MagicMock(return_value=None)

        tool_started = asyncio.Event()

        async def chat_with_retry(**kwargs):
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={})],
                usage={},
            )

        async def hanging_tool(name, args):
            tool_started.set()
            await asyncio.sleep(60)

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None
        loop.tools.execute = AsyncMock(side_effect=hanging_tool)

        task = asyncio.create_task(loop._process_message(_msg("go")))
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        loop._queue_injection(_msg("never mind, drop it"))

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert any(
            "never mind, drop it" in str(m.get("content"))
            for m in session.messages if m["role"] == "user"
        ), "interjection lost on cancellation"
        assert loop._pending_injections.get("test:c1") in (None, [])
