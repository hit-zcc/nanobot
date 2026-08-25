"""Tests for Feishu interactive-card callbacks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig


def _channel(*, allow_from: list[str] | None = None) -> FeishuChannel:
    return FeishuChannel(
        FeishuConfig(
            enabled=True,
            app_id="cli_test",
            app_secret="secret",
            allow_from=allow_from or ["*"],
        ),
        MessageBus(),
    )


def _callback(
    *,
    token: str = "token-1",
    sender_id: str = "ou_alice",
    chat_id: str = "oc_group",
    message_id: str = "om_card",
    value: dict | None = None,
):
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            operator=SimpleNamespace(open_id=sender_id),
            context=SimpleNamespace(
                open_chat_id=chat_id,
                open_message_id=message_id,
            ),
            action=SimpleNamespace(
                tag="button",
                name="approval",
                value=value or {"action": "merge_only", "repo": "nanobot"},
                form_value=None,
                option=None,
                input_value=None,
                options=None,
                checked=None,
            ),
        )
    )


@pytest.mark.asyncio
async def test_card_action_is_forwarded_to_agent_bus() -> None:
    channel = _channel()
    channel._handle_message = AsyncMock()
    data = _callback()

    await channel._on_card_action(data)

    channel._handle_message.assert_awaited_once()
    kwargs = channel._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "ou_alice"
    assert kwargs["chat_id"] == "oc_group"
    assert '"action": "merge_only"' in kwargs["content"]
    assert kwargs["metadata"]["message_id"] == "om_card"
    assert kwargs["metadata"]["msg_type"] == "card_action"
    assert kwargs["metadata"]["card_action"]["value"]["repo"] == "nanobot"


@pytest.mark.asyncio
async def test_card_action_routes_back_to_p2p_session() -> None:
    """A click in a private chat must reuse the ou_-keyed session, not oc_."""
    channel = _channel()
    channel._handle_message = AsyncMock()
    # The p2p chat the card was sent in, as learned from an inbound message.
    channel._chat_routes["oc_private"] = "ou_alice"

    await channel._on_card_action(_callback(chat_id="oc_private"))

    kwargs = channel._handle_message.await_args.kwargs
    assert kwargs["chat_id"] == "ou_alice"
    assert kwargs["metadata"]["chat_type"] == "p2p"


@pytest.mark.asyncio
async def test_card_action_routes_by_outbound_message_id() -> None:
    """The message_id of the card the bot sent wins over the chat id."""
    channel = _channel()
    channel._handle_message = AsyncMock()
    channel._outbound_routes["om_card"] = "ou_alice"
    channel._chat_routes["oc_private"] = "oc_stale"

    await channel._on_card_action(_callback(chat_id="oc_private", message_id="om_card"))

    assert channel._handle_message.await_args.kwargs["chat_id"] == "ou_alice"


@pytest.mark.asyncio
async def test_card_action_asks_feishu_for_unknown_chat() -> None:
    """With empty caches (e.g. after a restart), fall back to the chat mode API."""
    channel = _channel()
    channel._handle_message = AsyncMock()
    channel._client = MagicMock()
    channel._get_chat_mode_sync = MagicMock(return_value="p2p")

    await channel._on_card_action(_callback(chat_id="oc_private"))

    channel._get_chat_mode_sync.assert_called_once_with("oc_private")
    assert channel._handle_message.await_args.kwargs["chat_id"] == "ou_alice"
    # The looked-up route is cached so the next click skips the API call.
    assert channel._chat_routes["oc_private"] == "ou_alice"


@pytest.mark.asyncio
async def test_card_action_keeps_group_chat_id() -> None:
    channel = _channel()
    channel._handle_message = AsyncMock()
    channel._chat_routes["oc_group"] = "oc_group"

    await channel._on_card_action(_callback(chat_id="oc_group"))

    kwargs = channel._handle_message.await_args.kwargs
    assert kwargs["chat_id"] == "oc_group"
    assert kwargs["metadata"]["chat_type"] == "group"
    # Lets the group reply mention whoever clicked.
    assert kwargs["metadata"]["sender_open_id"] == "ou_alice"


def test_card_action_sync_acknowledges_and_schedules_processing() -> None:
    channel = _channel()
    channel._loop = MagicMock()
    channel._loop.is_running.return_value = True
    future = MagicMock()

    def _schedule(coro, loop):
        coro.close()
        return future

    with patch.object(asyncio, "run_coroutine_threadsafe", side_effect=_schedule) as schedule:
        response = channel._on_card_action_sync(_callback())

    schedule.assert_called_once()
    future.add_done_callback.assert_called_once()
    assert response.toast.type == "success"
    assert response.toast.content == "操作已收到，正在处理"


def test_card_action_sync_deduplicates_callback_retries() -> None:
    channel = _channel()
    channel._loop = MagicMock()
    channel._loop.is_running.return_value = True
    future = MagicMock()

    def _schedule(coro, loop):
        coro.close()
        return future

    data = _callback()
    with patch.object(asyncio, "run_coroutine_threadsafe", side_effect=_schedule) as schedule:
        first = channel._on_card_action_sync(data)
        second = channel._on_card_action_sync(data)

    assert schedule.call_count == 1
    assert first.toast.type == "success"
    assert second.toast.type == "warning"
    assert "重复" in second.toast.content


def test_card_action_sync_rejects_unauthorized_operator() -> None:
    channel = _channel(allow_from=["ou_allowed"])
    response = channel._on_card_action_sync(_callback(sender_id="ou_denied"))

    assert response.toast.type == "error"
    assert "没有权限" in response.toast.content
    assert not channel._processed_card_actions


def test_card_action_sync_reports_unavailable_loop() -> None:
    channel = _channel()
    channel._loop = None

    response = channel._on_card_action_sync(_callback())

    assert response.toast.type == "error"
    assert "暂时不可用" in response.toast.content
    assert not channel._processed_card_actions


# ---------------------------------------------------------------------------
# Route bookkeeping that makes the resolution above possible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p2p_message_records_open_id_route() -> None:
    channel = _channel()
    channel._client = MagicMock()
    channel._handle_message = AsyncMock()
    event = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_001",
                chat_id="oc_private",
                chat_type="p2p",
                message_type="text",
                content='{"text": "hi"}',
                parent_id=None,
                root_id=None,
                mentions=[],
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_alice"),
            ),
        )
    )

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(event)

    assert channel._chat_routes["oc_private"] == "ou_alice"


@pytest.mark.asyncio
async def test_send_records_outbound_message_route() -> None:
    channel = _channel()
    channel._client = MagicMock()
    response = MagicMock()
    response.success.return_value = True
    response.data.message_id = "om_sent"
    channel._client.im.v1.message.create.return_value = response

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="ou_alice",
        content="approve?",
        metadata={},
    ))

    assert channel._outbound_routes["om_sent"] == "ou_alice"


# ---------------------------------------------------------------------------
# Mention throttling
# ---------------------------------------------------------------------------

def test_channel_never_auto_mentions() -> None:
    """通道层一律不自动 @ —— @ 谁完全由模型在正文里决定。

    2026-08-20 聪聪：「不要自动艾特，都交给模型来决策」。
    此前这里会自动 @ 群里的提问者，导致模型定向发给别人时
    飞书上显示成 @ 了两个人。
    """
    channel = _channel()
    meta = {
        "chat_type": "group",
        "sender_open_id": "ou_alice",
        "message_id": "om_001",
    }

    assert channel._outbound_mention_target(meta) is None
    assert channel._outbound_mention_target({**meta, "message_id": "om_002"}) is None
    assert channel._outbound_mention_target({}) is None
