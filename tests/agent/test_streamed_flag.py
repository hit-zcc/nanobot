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
