"""The _streamed flag must reflect whether content was actually streamed.

Regression test: when streaming is enabled but the model's turn produces zero
visible deltas (e.g. it only made tool calls and its final message was empty),
the final outbound message used to be marked _streamed and silently dropped by
the channel manager — the user got no reply at all.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


class TestStreamedFlag:
    @pytest.mark.asyncio
    async def test_no_streamed_flag_when_nothing_streamed(self, tmp_path: Path) -> None:
        """Tool-only turn with an empty final message: fallback must be sendable."""
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="write_file",
            arguments={"path": str(tmp_path / "out.txt"), "content": "data"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="", tool_calls=[]),
        ])

        async def fake_stream(*args, on_content_delta=None, **kwargs):
            return next(calls)  # never emits a delta

        loop.provider.chat_stream_with_retry = AsyncMock(side_effect=fake_stream)

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="hi")
        result = await loop._process_message(msg, on_stream=AsyncMock(), on_stream_end=AsyncMock())

        assert result is not None
        assert result.content  # placeholder text, not empty
        assert not result.metadata.get("_streamed"), (
            "zero-delta turn must not be marked _streamed, or the channel drops the reply"
        )

    @pytest.mark.asyncio
    async def test_streamed_flag_set_when_content_streamed(self, tmp_path: Path) -> None:
        """Normal streamed reply: final message stays marked _streamed (no duplicate send)."""
        loop = _make_loop(tmp_path)

        async def fake_stream(*args, on_content_delta=None, **kwargs):
            if on_content_delta:
                await on_content_delta("Hello there")
            return LLMResponse(content="Hello there", tool_calls=[])

        loop.provider.chat_stream_with_retry = AsyncMock(side_effect=fake_stream)

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="hi")
        result = await loop._process_message(msg, on_stream=AsyncMock(), on_stream_end=AsyncMock())

        assert result is not None
        assert result.metadata.get("_streamed") is True

    @pytest.mark.asyncio
    async def test_max_iteration_message_not_hidden_by_earlier_streaming(
        self, tmp_path: Path,
    ) -> None:
        """Tool commentary from earlier segments must not hide the hard-limit message."""
        loop = _make_loop(tmp_path)
        loop.max_iterations = 2
        tool_call = ToolCallRequest(
            id="call1", name="list_dir", arguments={"path": "."},
        )

        async def fake_stream(*args, on_content_delta=None, **kwargs):
            if on_content_delta:
                await on_content_delta("还在处理中")
            return LLMResponse(content="还在处理中", tool_calls=[tool_call])

        loop.provider.chat_stream_with_retry = AsyncMock(side_effect=fake_stream)
        loop.tools.execute = AsyncMock(return_value="ok")

        msg = InboundMessage(
            channel="feishu", sender_id="user1", chat_id="chat123", content="hi",
        )
        result = await loop._process_message(
            msg, on_stream=AsyncMock(), on_stream_end=AsyncMock(),
        )

        assert result is not None
        assert "工具调用硬限制（2轮）" in result.content
        assert not result.metadata.get("_streamed")


class TestToolProgressRouting:
    @pytest.mark.asyncio
    async def test_feishu_instance_uses_in_place_progress_metadata(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from nanobot.agent.runner import AgentRunner

        monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_FIRST", 0.01)
        monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_MAX", 0.01)
        monkeypatch.setattr(AgentRunner, "_TOOL_HEARTBEAT_GROWTH", 1.0)

        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(id="call1", name="exec", arguments={"command": "slow"})
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="done", tool_calls=[]),
        ])

        async def fake_chat(**kwargs):
            return next(calls)

        async def slow_tool(*args, **kwargs):
            import asyncio

            await asyncio.sleep(0.05)
            return "ok"

        loop.provider.chat_with_retry = AsyncMock(side_effect=fake_chat)
        loop.tools.execute = slow_tool
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        msg = InboundMessage(
            channel="feishu.jarvis",
            sender_id="user1",
            chat_id="chat123",
            content="hi",
        )
        result = await loop._process_message(msg)

        assert result is not None
        progress = []
        while loop.bus.outbound_size:
            outbound = loop.bus.outbound.get_nowait()
            if outbound.metadata.get("_tool_progress_id"):
                progress.append(outbound)

        assert len(progress) >= 2
        assert {item.metadata["_tool_progress_id"] for item in progress} == {
            progress[0].metadata["_tool_progress_id"]
        }
        assert progress[-1].metadata["_tool_progress_done"] is True


class TestIterationLimitNotices:
    @pytest.mark.asyncio
    async def test_main_agent_warning_is_sent_as_important_notice(
        self, tmp_path: Path,
    ) -> None:
        loop = _make_loop(tmp_path)
        loop.max_iterations = 11
        tool_call = ToolCallRequest(
            id="call1", name="list_dir", arguments={"path": "."},
        )
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="working", tool_calls=[tool_call]),
            LLMResponse(content="done", tool_calls=[]),
        ])
        loop.tools.execute = AsyncMock(return_value="ok")
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        msg = InboundMessage(
            channel="feishu", sender_id="user1", chat_id="chat123", content="hi",
        )
        result = await loop._process_message(msg)

        assert result is not None
        warning = next(
            item
            for item in (
                loop.bus.outbound.get_nowait()
                for _ in range(loop.bus.outbound_size)
            )
            if item.metadata.get("_important_notice")
        )
        assert "距离硬限制只剩 10 轮" in warning.content
        assert warning.metadata["_important_notice"] is True
