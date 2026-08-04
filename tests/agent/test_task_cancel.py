"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop(*, exec_config=None):
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace, exec_config=exec_config)
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    def test_exec_tool_not_registered_when_disabled(self):
        from nanobot.config.schema import ExecToolConfig

        loop, _bus = _make_loop(exec_config=ExecToolConfig(enable=False))

        assert loop.tools.get("exec") is None

    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_short_progress_query_returns_background_status_without_llm(self):
        from nanobot.bus.events import InboundMessage

        loop, _bus = _make_loop()
        loop.subagents.format_session_status.return_value = (
            "当前后台任务：\n\n⏳ 后台任务「重构」 `sub-1`\n"
            "状态：运行中 · 已用时 2分 · 第 3 轮"
        )
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        msg = InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="嗯 咋样了？",
        )
        result = await loop._process_message(msg)

        assert "后台任务「重构」" in result.content
        loop.subagents.format_session_status.assert_called_once_with("test:c1")
        loop.memory_consolidator.maybe_consolidate_by_tokens.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]

    @pytest.mark.asyncio
    async def test_system_completion_serializes_with_origin_session(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, _bus = _make_loop()
        order = []
        user_started = asyncio.Event()
        release_user = asyncio.Event()

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.sender_id}")
            if m.sender_id == "u1":
                user_started.set()
                await release_user.wait()
            order.append(f"end-{m.sender_id}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        user = InboundMessage(
            channel="test", sender_id="u1", chat_id="c1", content="好了吗",
        )
        completion = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="test:c1",
            content="done",
            session_key_override="test:c1",
        )

        user_task = asyncio.create_task(loop._dispatch(user))
        await user_started.wait()
        completion_task = asyncio.create_task(loop._dispatch(completion))
        await asyncio.sleep(0)

        assert order == ["start-u1"]

        release_user.set()
        await asyncio.gather(user_task, completion_task)
        assert order == [
            "start-u1",
            "end-u1",
            "start-subagent",
            "end-subagent",
        ]

    def test_legacy_system_message_resolves_origin_session_key(self):
        from nanobot.bus.events import InboundMessage

        loop, _bus = _make_loop()
        message = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="feishu:chat-1",
            content="done",
        )

        assert loop._dispatch_session_key(message) == "feishu:chat-1"


class TestSubagentCancellation:
    @pytest.mark.asyncio
    async def test_completion_announcement_keeps_origin_session_key(self, tmp_path):
        from nanobot.agent.subagent import SubagentManager, SubagentTaskState
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
        mgr._task_states["sub-1"] = SubagentTaskState(
            task_id="sub-1",
            label="后台任务",
            task="do work",
            origin_channel="feishu.jarvis",
            origin_chat_id="chat-1",
            session_key="feishu.jarvis:thread-1",
        )

        await mgr._announce_result(
            "sub-1",
            "后台任务",
            "do work",
            "done",
            {"channel": "feishu.jarvis", "chat_id": "chat-1"},
            "ok",
        )
        announcement = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)

        assert announcement.channel == "system"
        assert announcement.chat_id == "feishu.jarvis:chat-1"
        assert announcement.session_key == "feishu.jarvis:thread-1"

    @pytest.mark.asyncio
    async def test_spawn_exposes_structured_session_status(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
        started = asyncio.Event()
        release = asyncio.Event()

        async def hold(*args, **kwargs):
            started.set()
            await release.wait()

        monkeypatch.setattr(mgr, "_run_subagent", hold)

        response = await mgr.spawn(
            "refactor pools",
            label="共享线程池重构",
            origin_channel="feishu.jarvis",
            origin_chat_id="u1",
            session_key="feishu.jarvis:u1",
        )
        await started.wait()

        status = mgr.format_session_status("feishu.jarvis:u1")
        context = mgr.format_session_context("feishu.jarvis:u1")

        assert "共享线程池重构" in response
        assert "状态：运行中" in status
        assert "completed_tool_steps=0" in context

        release.set()
        await asyncio.gather(*mgr._running_tasks.values())

    @pytest.mark.asyncio
    async def test_subagent_publishes_updatable_progress_heartbeat(
        self, monkeypatch, tmp_path,
    ):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(
                content="compile",
                tool_calls=[ToolCallRequest(
                    id="call_1",
                    name="exec",
                    arguments={"command": "mvn -pl get-aisearch test"},
                )],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ])
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
        monkeypatch.setattr(mgr, "_PROGRESS_FIRST_SECONDS", 0.01)
        monkeypatch.setattr(mgr, "_PROGRESS_INTERVAL_SECONDS", 0.01)

        async def slow_execute(self, name, arguments):
            await asyncio.sleep(0.035)
            return "BUILD SUCCESS"

        monkeypatch.setattr(
            "nanobot.agent.tools.registry.ToolRegistry.execute",
            slow_execute,
        )

        await mgr._run_subagent(
            "sub-1",
            "refactor pools",
            "共享线程池重构",
            {"channel": "feishu.jarvis", "chat_id": "u1"},
            session_key="feishu.jarvis:u1",
        )

        progress = []
        while not bus.outbound.empty():
            progress.append(bus.outbound.get_nowait())

        assert any(
            "运行 Maven 编译/测试" in item.content
            and item.metadata["_tool_progress_done"] is False
            for item in progress
        )
        assert progress[-1].metadata["_tool_progress_done"] is True
        assert progress[-1].metadata["_tool_progress_id"] == "subagent-progress:sub-1"
        assert mgr._task_states["sub-1"].status == "completed"

    @pytest.mark.asyncio
    async def test_cancel_by_session(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        count = await mgr.cancel_by_session("test:c1")
        assert count == 1
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_by_session_no_tasks(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)
        assert await mgr.cancel_by_session("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_subagent_preserves_reasoning_fields_in_tool_turn(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        captured_second_call: list[dict] = []

        call_count = {"n": 0}

        async def scripted_chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="thinking",
                    tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
                    reasoning_content="hidden reasoning",
                    thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                )
            captured_second_call[:] = messages
            return LLMResponse(content="done", tool_calls=[])
        provider.chat_with_retry = scripted_chat_with_retry
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

        async def fake_execute(self, name, arguments):
            return "tool result"

        monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        assistant_messages = [
            msg for msg in captured_second_call
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
        assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]

    @pytest.mark.asyncio
    async def test_subagent_announces_error_when_tool_execution_fails(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="thinking",
            tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
        ))
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
        mgr._announce_result = AsyncMock()

        calls = {"n": 0}

        async def fake_execute(self, name, arguments):
            calls["n"] += 1
            if calls["n"] == 1:
                return "first result"
            raise RuntimeError("boom")

        monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        mgr._announce_result.assert_awaited_once()
        args = mgr._announce_result.await_args.args
        assert "Completed steps:" in args[3]
        assert "- list_dir: first result" in args[3]
        assert "Failure:" in args[3]
        assert "- list_dir: boom" in args[3]
        assert args[5] == "error"

    @pytest.mark.asyncio
    async def test_cancel_by_session_cancels_running_subagent_tool(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="thinking",
            tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
        ))
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
        mgr._announce_result = AsyncMock()

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_execute(self, name, arguments):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

        task = asyncio.create_task(
            mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})
        )
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        await started.wait()

        count = await mgr.cancel_by_session("test:c1")

        assert count == 1
        assert cancelled.is_set()
        assert task.cancelled()
        mgr._announce_result.assert_not_awaited()
