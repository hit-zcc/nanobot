"""Tests for Feishu interactive-card callbacks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
