"""Tests for buffering un-addressed Feishu group messages as context.

People routinely post the question first and @ the bot in a follow-up message.
The follow-up alone says nothing, so the messages in between are kept and handed
over on the next mention.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig


def _channel(**overrides) -> FeishuChannel:
    config = FeishuConfig(
        enabled=True,
        app_id="cli_test",
        app_secret="secret",
        allow_from=["*"],
        group_policy="mention",
        **overrides,
    )
    channel = FeishuChannel(config, MessageBus())
    channel._client = MagicMock()
    channel._handle_message = AsyncMock()
    return channel


_BOT_MENTION = SimpleNamespace(
    key="@_user_1",
    id=SimpleNamespace(open_id="ou_bot", user_id=None),
    name="wuw",
)


def _group_event(
    *,
    text: str,
    message_id: str = "om_001",
    sender_open_id: str = "ou_alice",
    mentions: list | None = None,
    create_time: str | None = None,
):
    # Fresh, otherwise the staleness guard drops the event before buffering.
    create_time = create_time or str(int(time.time() * 1000))
    message = SimpleNamespace(
        message_id=message_id,
        chat_id="oc_group",
        chat_type="group",
        message_type="text",
        content=json.dumps({"text": text}),
        create_time=create_time,
        parent_id=None,
        root_id=None,
        thread_id=None,
        mentions=mentions or [],
    )
    sender = SimpleNamespace(
        sender_type="user",
        sender_id=SimpleNamespace(open_id=sender_open_id),
    )
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


async def _deliver(channel: FeishuChannel, event) -> None:
    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(event)


@pytest.mark.asyncio
async def test_bare_mention_carries_the_preceding_message() -> None:
    channel = _channel()

    await _deliver(channel, _group_event(text="这个用户是什么版本的app"))
    # Un-addressed messages are never answered on their own.
    channel._handle_message.assert_not_awaited()

    await _deliver(channel, _group_event(
        text="@wuw", message_id="om_002", mentions=[_BOT_MENTION],
    ))

    channel._handle_message.assert_awaited_once()
    content = channel._handle_message.await_args.kwargs["content"]
    assert "这个用户是什么版本的app" in content
    assert content.endswith("@wuw")
    # The clicker posted both, so the buffered line is marked as theirs.
    assert "(same sender)" in content


@pytest.mark.asyncio
async def test_buffer_is_drained_after_use() -> None:
    channel = _channel()

    await _deliver(channel, _group_event(text="第一个问题"))
    await _deliver(channel, _group_event(
        text="@wuw", message_id="om_002", mentions=[_BOT_MENTION],
    ))
    await _deliver(channel, _group_event(
        text="@wuw 再说一次", message_id="om_003", mentions=[_BOT_MENTION],
    ))

    second = channel._handle_message.await_args_list[1].kwargs["content"]
    assert "第一个问题" not in second
    assert channel._group_context.get("oc_group") in (None, )


@pytest.mark.asyncio
async def test_only_the_most_recent_messages_are_kept() -> None:
    channel = _channel(group_context_messages=2)

    for i in range(4):
        await _deliver(channel, _group_event(text=f"消息{i}", message_id=f"om_{i}"))
    await _deliver(channel, _group_event(
        text="@wuw", message_id="om_last", mentions=[_BOT_MENTION],
    ))

    content = channel._handle_message.await_args.kwargs["content"]
    assert "消息0" not in content
    assert "消息1" not in content
    assert "消息2" in content
    assert "消息3" in content


@pytest.mark.asyncio
async def test_other_senders_are_labelled_by_open_id() -> None:
    channel = _channel()

    await _deliver(channel, _group_event(text="别人说的话", sender_open_id="ou_bob"))
    await _deliver(channel, _group_event(
        text="@wuw", message_id="om_002", mentions=[_BOT_MENTION],
    ))

    content = channel._handle_message.await_args.kwargs["content"]
    assert "(ou_bob)" in content


@pytest.mark.asyncio
async def test_buffer_can_be_disabled() -> None:
    channel = _channel(group_context_messages=0)

    await _deliver(channel, _group_event(text="这条不该被记住"))
    await _deliver(channel, _group_event(
        text="@wuw", message_id="om_002", mentions=[_BOT_MENTION],
    ))

    content = channel._handle_message.await_args.kwargs["content"]
    assert "这条不该被记住" not in content
    assert not channel._group_context


@pytest.mark.asyncio
async def test_p2p_messages_are_unaffected() -> None:
    channel = _channel()
    event = _group_event(text="hello")
    event.event.message.chat_type = "p2p"

    await _deliver(channel, event)

    # Answered directly, and nothing is buffered for a private chat.
    channel._handle_message.assert_awaited_once()
    assert channel._handle_message.await_args.kwargs["content"] == "hello"
    assert not channel._group_context
