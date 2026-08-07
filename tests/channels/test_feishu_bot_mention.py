"""Group mentions must address *this* bot, not merely be an @mention.

Feishu omits ``user_id`` on human mentions unless the app holds contact
scopes, so "no user_id" cannot stand in for "this is the bot".  Matching on
that alone made the bot answer every ``@colleague`` message in a group.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig

BOT_OPEN_ID = "ou_bot_self"


def _channel(*, bot_open_id: str | None = BOT_OPEN_ID, **overrides) -> FeishuChannel:
    config = FeishuConfig(
        **{
            "enabled": True,
            "app_id": "cli_test",
            "app_secret": "secret",
            "allow_from": ["*"],
            "group_policy": "mention",
            **overrides,
        }
    )
    channel = FeishuChannel(config, MessageBus())
    channel._client = MagicMock()
    channel._handle_message = AsyncMock()
    channel._bot_open_id = bot_open_id
    return channel


def _mention(open_id: str, user_id: str | None = None, name: str = "someone"):
    return SimpleNamespace(
        key="@_user_1",
        id=SimpleNamespace(open_id=open_id, user_id=user_id),
        name=name,
    )


def _message(text: str, mentions: list | None = None):
    return SimpleNamespace(
        message_id="om_1",
        chat_id="oc_group",
        chat_type="group",
        message_type="text",
        content=json.dumps({"text": text}),
        create_time=str(int(time.time() * 1000)),
        mentions=mentions or [],
    )


def test_mention_of_another_person_is_not_for_the_bot():
    """The regression: @colleague has no user_id either, yet is not for us."""
    channel = _channel()
    msg = _message("@杨溥 可以做", [_mention("ou_colleague", user_id=None, name="杨溥")])

    assert channel._is_bot_mentioned(msg) is False
    assert channel._is_group_message_for_bot(msg) is False


def test_mention_of_the_bot_is_recognised():
    channel = _channel()
    msg = _message("@wuw 看一下", [_mention(BOT_OPEN_ID, user_id=None, name="wuw")])

    assert channel._is_bot_mentioned(msg) is True
    assert channel._is_group_message_for_bot(msg) is True


def test_bot_found_among_several_mentions():
    channel = _channel()
    msg = _message(
        "@张聪聪 @wuw 看下这个",
        [_mention("ou_zcc", name="张聪聪"), _mention(BOT_OPEN_ID, name="wuw")],
    )

    assert channel._is_bot_mentioned(msg) is True


def test_mention_with_user_id_still_matches_by_open_id():
    """Contact scopes may populate user_id; identity must still be open_id."""
    channel = _channel()
    msg = _message("@wuw hi", [_mention(BOT_OPEN_ID, user_id="u-123", name="wuw")])

    assert channel._is_bot_mentioned(msg) is True


def test_no_mentions_at_all():
    channel = _channel()
    assert channel._is_bot_mentioned(_message("随便聊两句")) is False


def test_at_all_still_reaches_the_bot():
    channel = _channel()
    msg = SimpleNamespace(
        message_id="om_2",
        chat_id="oc_group",
        chat_type="group",
        message_type="text",
        content=json.dumps({"text": "@_all 通知"}),
        create_time=str(int(time.time() * 1000)),
        mentions=[],
    )
    assert channel._is_bot_mentioned(msg) is True


def test_degraded_path_when_identity_unknown():
    """If the identity lookup failed, stay reachable rather than going silent."""
    channel = _channel(bot_open_id=None)
    msg = _message("@谁都行", [_mention("ou_whoever", user_id=None)])

    assert channel._is_bot_mentioned(msg) is True


def test_open_policy_ignores_mentions_entirely():
    channel = _channel(group_policy="open")
    msg = _message("没有艾特任何人")

    assert channel._is_group_message_for_bot(msg) is True


@pytest.mark.asyncio
async def test_group_message_for_someone_else_is_buffered_not_answered():
    """End-to-end: the un-addressed message becomes context, not a reply."""
    channel = _channel()
    sender = SimpleNamespace(
        sender_id=SimpleNamespace(open_id="ou_alice"), sender_type="user"
    )
    message = _message(
        "@杨溥 这条不该触发回复", [_mention("ou_colleague", name="杨溥")]
    )
    data = SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))

    await channel._on_message(data)

    channel._handle_message.assert_not_called()
    assert channel._group_context.get("oc_group")
