"""Tests for Feishu message reply (quote) feature."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check optional Feishu dependencies before running tests
try:
    from nanobot.channels import feishu
    FEISHU_AVAILABLE = getattr(feishu, "FEISHU_AVAILABLE", False)
except ImportError:
    FEISHU_AVAILABLE = False

if not FEISHU_AVAILABLE:
    pytest.skip("Feishu dependencies not installed (lark-oapi)", allow_module_level=True)

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import (
    FeishuChannel,
    FeishuConfig,
    _resolve_text_mentions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feishu_channel(reply_to_message: bool = False) -> FeishuChannel:
    config = FeishuConfig(
        enabled=True,
        app_id="cli_test",
        app_secret="secret",
        allow_from=["*"],
        reply_to_message=reply_to_message,
    )
    channel = FeishuChannel(config, MessageBus())
    channel._client = MagicMock()
    # _loop is only used by the WebSocket thread bridge; not needed for unit tests
    channel._loop = None
    return channel


def _make_feishu_event(
    *,
    message_id: str = "om_001",
    chat_id: str = "oc_abc",
    chat_type: str = "p2p",
    msg_type: str = "text",
    content: str = '{"text": "hello"}',
    sender_open_id: str = "ou_alice",
    sender_type: str = "user",
    mentions: list | None = None,
    parent_id: str | None = None,
    root_id: str | None = None,
):
    message = SimpleNamespace(
        message_id=message_id,
        chat_id=chat_id,
        chat_type=chat_type,
        message_type=msg_type,
        content=content,
        parent_id=parent_id,
        root_id=root_id,
        mentions=mentions or [],
    )
    sender = SimpleNamespace(
        sender_type=sender_type,
        sender_id=SimpleNamespace(open_id=sender_open_id),
    )
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


def _make_get_message_response(
    text: str,
    msg_type: str = "text",
    success: bool = True,
    mentions: list | None = None,
):
    """Build a fake im.v1.message.get response object."""
    body = SimpleNamespace(content=json.dumps({"text": text}))
    item = SimpleNamespace(msg_type=msg_type, body=body, mentions=mentions or [])
    data = SimpleNamespace(items=[item])
    resp = MagicMock()
    resp.success.return_value = success
    resp.data = data
    resp.code = 0
    resp.msg = "ok"
    return resp


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_feishu_config_reply_to_message_defaults_false() -> None:
    assert FeishuConfig().reply_to_message is False


def test_feishu_config_reply_to_message_can_be_enabled() -> None:
    config = FeishuConfig(reply_to_message=True)
    assert config.reply_to_message is True


def test_resolve_text_mentions_uses_name_and_preserves_ids() -> None:
    mention = SimpleNamespace(
        key="@_user_1",
        name="小步",
        id=SimpleNamespace(
            open_id="ou_bot",
            user_id=None,
            union_id="on_bot",
        ),
    )

    text, resolved = _resolve_text_mentions("@_user_1 请检查", [mention])

    assert text == "@小步 请检查"
    assert resolved == [{
        "key": "@_user_1",
        "name": "小步",
        "open_id": "ou_bot",
        "union_id": "on_bot",
    }]


def test_resolve_text_mentions_replaces_longer_keys_first() -> None:
    mentions = [
        SimpleNamespace(
            key="@_user_1",
            name="一号",
            id=SimpleNamespace(open_id="ou_1", user_id=None, union_id=None),
        ),
        SimpleNamespace(
            key="@_user_10",
            name="十号",
            id=SimpleNamespace(open_id="ou_10", user_id=None, union_id=None),
        ),
    ]

    text, _ = _resolve_text_mentions("@_user_10 回复 @_user_1", mentions)

    assert text == "@十号 回复 @一号"


# ---------------------------------------------------------------------------
# _get_message_content_sync tests
# ---------------------------------------------------------------------------

def test_get_message_content_sync_returns_reply_prefix() -> None:
    channel = _make_feishu_channel()
    channel._client.im.v1.message.get.return_value = _make_get_message_response("what time is it?")

    result = channel._get_message_content_sync("om_parent")

    assert result == "[Reply to: what time is it?]"


def test_get_message_content_sync_resolves_parent_mentions() -> None:
    channel = _make_feishu_channel()
    mention = SimpleNamespace(
        key="@_user_1",
        name="小橘",
        id=SimpleNamespace(open_id="ou_bot", user_id=None, union_id=None),
    )
    channel._client.im.v1.message.get.return_value = _make_get_message_response(
        "@_user_1 请重新回答",
        mentions=[mention],
    )

    result = channel._get_message_content_sync("om_parent")

    assert result == "[Reply to: @小橘 请重新回答]"


def test_get_message_content_sync_truncates_long_text() -> None:
    channel = _make_feishu_channel()
    long_text = "x" * (FeishuChannel._REPLY_CONTEXT_MAX_LEN + 50)
    channel._client.im.v1.message.get.return_value = _make_get_message_response(long_text)

    result = channel._get_message_content_sync("om_parent")

    assert result is not None
    assert result.endswith("...]")
    inner = result[len("[Reply to: ") : -1]
    assert len(inner) == FeishuChannel._REPLY_CONTEXT_MAX_LEN + len("...")


def test_get_message_content_sync_returns_none_on_api_failure() -> None:
    channel = _make_feishu_channel()
    resp = MagicMock()
    resp.success.return_value = False
    resp.code = 230002
    resp.msg = "bot not in group"
    channel._client.im.v1.message.get.return_value = resp

    result = channel._get_message_content_sync("om_parent")

    assert result is None


def test_get_message_content_sync_returns_none_for_non_text_type() -> None:
    channel = _make_feishu_channel()
    body = SimpleNamespace(content=json.dumps({"image_key": "img_1"}))
    item = SimpleNamespace(msg_type="image", body=body)
    data = SimpleNamespace(items=[item])
    resp = MagicMock()
    resp.success.return_value = True
    resp.data = data
    channel._client.im.v1.message.get.return_value = resp

    result = channel._get_message_content_sync("om_parent")

    assert result is None


def test_get_message_content_sync_returns_none_when_empty_text() -> None:
    channel = _make_feishu_channel()
    channel._client.im.v1.message.get.return_value = _make_get_message_response("   ")

    result = channel._get_message_content_sync("om_parent")

    assert result is None


# ---------------------------------------------------------------------------
# _reply_message_sync tests
# ---------------------------------------------------------------------------

def test_reply_message_sync_returns_true_on_success() -> None:
    channel = _make_feishu_channel()
    resp = MagicMock()
    resp.success.return_value = True
    channel._client.im.v1.message.reply.return_value = resp

    ok = channel._reply_message_sync("om_parent", "text", '{"text":"hi"}')

    assert ok is True
    channel._client.im.v1.message.reply.assert_called_once()


def test_reply_message_sync_returns_false_on_api_error() -> None:
    channel = _make_feishu_channel()
    resp = MagicMock()
    resp.success.return_value = False
    resp.code = 400
    resp.msg = "bad request"
    resp.get_log_id.return_value = "log_x"
    channel._client.im.v1.message.reply.return_value = resp

    ok = channel._reply_message_sync("om_parent", "text", '{"text":"hi"}')

    assert ok is False


def test_reply_message_sync_returns_false_on_exception() -> None:
    channel = _make_feishu_channel()
    channel._client.im.v1.message.reply.side_effect = RuntimeError("network error")

    ok = channel._reply_message_sync("om_parent", "text", '{"text":"hi"}')

    assert ok is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "expected_msg_type"),
    [
        ("voice.opus", "audio"),
        ("clip.mp4", "video"),
        ("report.pdf", "file"),
    ],
)
async def test_send_uses_expected_feishu_msg_type_for_uploaded_files(
    tmp_path: Path, filename: str, expected_msg_type: str
) -> None:
    channel = _make_feishu_channel()
    file_path = tmp_path / filename
    file_path.write_bytes(b"demo")

    send_calls: list[tuple[str, str, str, str]] = []

    def _record_send(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str:
        # 返回 message_id：真实实现成功时就返回它，替身不返回会被判成投递失败。
        send_calls.append((receive_id_type, receive_id, msg_type, content))
        return "om_sent"

    with patch.object(channel, "_upload_file_sync", return_value="file-key"), patch.object(
        channel, "_send_message_sync", side_effect=_record_send
    ):
        await channel.send(
            OutboundMessage(
                channel="feishu",
                chat_id="oc_test",
                content="",
                media=[str(file_path)],
                metadata={},
            )
        )

    assert len(send_calls) == 1
    receive_id_type, receive_id, msg_type, content = send_calls[0]
    assert receive_id_type == "chat_id"
    assert receive_id == "oc_test"
    assert msg_type == expected_msg_type
    assert json.loads(content) == {"file_key": "file-key"}


# ---------------------------------------------------------------------------
# send() — reply routing tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_uses_reply_api_when_configured() -> None:
    channel = _make_feishu_channel(reply_to_message=True)

    reply_resp = MagicMock()
    reply_resp.success.return_value = True
    channel._client.im.v1.message.reply.return_value = reply_resp

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="oc_abc",
        content="hello",
        metadata={"message_id": "om_001"},
    ))

    channel._client.im.v1.message.reply.assert_called_once()
    channel._client.im.v1.message.create.assert_not_called()


@pytest.mark.asyncio
async def test_send_uses_create_api_when_reply_disabled() -> None:
    channel = _make_feishu_channel(reply_to_message=False)

    create_resp = MagicMock()
    create_resp.success.return_value = True
    channel._client.im.v1.message.create.return_value = create_resp

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="oc_abc",
        content="hello",
        metadata={"message_id": "om_001"},
    ))

    channel._client.im.v1.message.create.assert_called_once()
    channel._client.im.v1.message.reply.assert_not_called()


@pytest.mark.asyncio
async def test_send_sanitizes_unresolved_mention_placeholder() -> None:
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        # 真实的 _send_message_sync 成功时返回 message_id；替身也必须返回一个，
        # 否则会被判成投递失败（list.append 返回 None）。
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content="@_user_1 查清楚了",
            metadata={},
        ))

    assert json.loads(sent[0][3])["text"] == "用户 查清楚了"


@pytest.mark.asyncio
async def test_send_does_not_auto_mention_group_sender() -> None:
    """群回复不再自动 @ 提问者 —— @ 谁由模型自己决定（2026-08-20 聪聪要求）。"""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content="查清楚了",
            metadata={
                "chat_type": "group",
                "sender_type": "user",
                "sender_open_id": "ou_alice",
            },
        ))

    assert sent[0][2] == "text"
    body = json.loads(sent[0][3])["text"]
    assert body == "查清楚了"
    assert "<at" not in body


@pytest.mark.asyncio
async def test_send_keeps_model_written_text_mention() -> None:
    """模型自己写的 <at> 必须原样生效，且不会被追加第二个 @。"""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content='<at user_id="ou_bob">牛小数</at> 在吗',
            metadata={
                "chat_type": "group",
                "sender_type": "user",
                "sender_open_id": "ou_alice",
            },
        ))

    body = json.loads(sent[0][3])["text"]
    assert body == '<at user_id="ou_bob">牛小数</at> 在吗'
    assert body.count("<at") == 1


@pytest.mark.asyncio
async def test_send_upgrades_known_leading_display_mention_to_native_at() -> None:
    """``@name`` must carry the cached open_id instead of looking clickable only."""
    channel = _make_feishu_channel(reply_to_message=False)
    channel._cache_user_name("ou_target", "目标机器人")
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content="@目标机器人 介绍一下你自己",
            metadata={"chat_type": "group", "sender_type": "user"},
        ))

    assert json.loads(sent[0][3])["text"] == (
        '<at user_id="ou_target">目标机器人</at> 介绍一下你自己'
    )


def test_doubao_open_id_is_available_before_runtime_cache_warms() -> None:
    channel = _make_feishu_channel(reply_to_message=False)
    assert channel._open_id_for_display_name("豆包") == (
        "ou_66244dbc857794ee92956876ab0f022b"
    )


@pytest.mark.asyncio
async def test_send_drops_fake_at_when_display_name_has_no_open_id() -> None:
    """An unresolved display name must not be emitted as a fake plain-text @."""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content="@不知道是谁 请看一下",
            metadata={"chat_type": "group", "sender_type": "user"},
        ))

    body = json.loads(sent[0][3])["text"]
    assert body == "不知道是谁 请看一下"
    assert not body.startswith("@")


@pytest.mark.asyncio
async def test_text_mention_written_in_card_syntax_is_normalized() -> None:
    """模型写成卡片语法 ``id=``，发 text 时要改写成 ``user_id=`` 才真的 @ 得到。"""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content='<at id="ou_bob"></at> 在吗',
            metadata={"chat_type": "group", "sender_type": "user"},
        ))

    assert json.loads(sent[0][3])["text"] == '<at user_id="ou_bob">用户</at> 在吗'


@pytest.mark.asyncio
async def test_post_renders_model_written_mention_as_at_element() -> None:
    """post 不认字符串形式的 <at>，必须转成 at 元素，否则整段标签露给用户看。"""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content='<at user_id="ou_bob">牛小数</at> 请看 [文档](https://example.com)',
            metadata={
                "chat_type": "group",
                "sender_type": "user",
                "sender_open_id": "ou_alice",
            },
        ))

    assert sent[0][2] == "post"
    paragraph = json.loads(sent[0][3])["zh_cn"]["content"][0]
    assert paragraph[0] == {
        "tag": "at",
        "user_id": "ou_bob",
        "user_name": "牛小数",
    }
    # 同一行里的链接不能因为 at 的插入而丢失
    assert {"tag": "a", "text": "文档", "href": "https://example.com"} in paragraph
    # 没有任何一段文字残留着原始标签
    assert all("<at" not in e.get("text", "") for e in paragraph)


@pytest.mark.asyncio
async def test_card_normalizes_model_written_mention() -> None:
    """卡片只认 ``<at id=…></at>``；模型写的 text 语法要改写过去。"""
    channel = _make_feishu_channel(reply_to_message=False)
    sent: list[tuple[str, str, str, str]] = []

    with patch.object(
        channel,
        "_send_message_sync",
        side_effect=lambda *args: (sent.append(args), "om_sent")[1],
    ):
        await channel.send(OutboundMessage(
            channel="feishu",
            chat_id="oc_abc",
            content='<at user_id="ou_bob">牛小数</at> **严重问题**',
            metadata={
                "chat_type": "group",
                "sender_type": "user",
                "sender_open_id": "ou_alice",
            },
        ))

    assert sent[0][2] == "interactive"
    card = json.loads(sent[0][3])
    content = card["elements"][0]["content"]
    assert content.startswith('<at id="ou_bob"></at> **严重问题**')
    assert "ou_alice" not in content  # 没有自动补上提问者


@pytest.mark.asyncio
async def test_send_uses_create_api_when_no_message_id() -> None:
    channel = _make_feishu_channel(reply_to_message=True)

    create_resp = MagicMock()
    create_resp.success.return_value = True
    channel._client.im.v1.message.create.return_value = create_resp

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="oc_abc",
        content="hello",
        metadata={},
    ))

    channel._client.im.v1.message.create.assert_called_once()
    channel._client.im.v1.message.reply.assert_not_called()


@pytest.mark.asyncio
async def test_send_skips_reply_for_progress_messages() -> None:
    channel = _make_feishu_channel(reply_to_message=True)

    create_resp = MagicMock()
    create_resp.success.return_value = True
    channel._client.im.v1.message.create.return_value = create_resp

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="oc_abc",
        content="thinking...",
        metadata={"message_id": "om_001", "_progress": True},
    ))

    channel._client.im.v1.message.create.assert_called_once()
    channel._client.im.v1.message.reply.assert_not_called()


@pytest.mark.asyncio
async def test_send_fallback_to_create_when_reply_fails() -> None:
    channel = _make_feishu_channel(reply_to_message=True)

    reply_resp = MagicMock()
    reply_resp.success.return_value = False
    reply_resp.code = 400
    reply_resp.msg = "error"
    reply_resp.get_log_id.return_value = "log_x"
    channel._client.im.v1.message.reply.return_value = reply_resp

    create_resp = MagicMock()
    create_resp.success.return_value = True
    channel._client.im.v1.message.create.return_value = create_resp

    await channel.send(OutboundMessage(
        channel="feishu",
        chat_id="oc_abc",
        content="hello",
        metadata={"message_id": "om_001"},
    ))

    # reply attempted first, then falls back to create
    channel._client.im.v1.message.reply.assert_called_once()
    channel._client.im.v1.message.create.assert_called_once()


# ---------------------------------------------------------------------------
# _on_message — parent_id / root_id metadata tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_message_captures_parent_and_root_id_in_metadata() -> None:
    channel = _make_feishu_channel()
    channel._processed_message_ids.clear()
    channel._client.im.v1.message.react.return_value = MagicMock(success=lambda: True)

    captured = []

    async def _capture(**kwargs):
        captured.append(kwargs)

    channel._handle_message = _capture

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(
            _make_feishu_event(
                parent_id="om_parent",
                root_id="om_root",
            )
        )

    assert len(captured) == 1
    meta = captured[0]["metadata"]
    assert meta["parent_id"] == "om_parent"
    assert meta["root_id"] == "om_root"
    assert meta["message_id"] == "om_001"


@pytest.mark.asyncio
async def test_on_message_parent_and_root_id_none_when_absent() -> None:
    channel = _make_feishu_channel()
    channel._processed_message_ids.clear()

    captured = []

    async def _capture(**kwargs):
        captured.append(kwargs)

    channel._handle_message = _capture

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(_make_feishu_event())

    assert len(captured) == 1
    meta = captured[0]["metadata"]
    assert meta["parent_id"] is None
    assert meta["root_id"] is None


@pytest.mark.asyncio
async def test_on_message_prepends_reply_context_when_parent_id_present() -> None:
    channel = _make_feishu_channel()
    channel._processed_message_ids.clear()
    channel._client.im.v1.message.get.return_value = _make_get_message_response("original question")

    captured = []

    async def _capture(**kwargs):
        captured.append(kwargs)

    channel._handle_message = _capture

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(
            _make_feishu_event(
                content='{"text": "my answer"}',
                parent_id="om_parent",
            )
        )

    assert len(captured) == 1
    content = captured[0]["content"]
    assert content.startswith(
        "[Feishu private message — speaker: name unresolved (ou_alice); owner: false]"
    )
    assert "[Reply to: original question]" in content
    assert "my answer" in content


@pytest.mark.asyncio
async def test_on_message_no_extra_api_call_when_no_parent_id() -> None:
    channel = _make_feishu_channel()
    channel._processed_message_ids.clear()

    captured = []

    async def _capture(**kwargs):
        captured.append(kwargs)

    channel._handle_message = _capture

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(_make_feishu_event())

    channel._client.im.v1.message.get.assert_not_called()
    assert len(captured) == 1


def _bot_mention() -> SimpleNamespace:
    return SimpleNamespace(
        key="@_user_1",
        name="Jarvis",
        id=SimpleNamespace(user_id=None, open_id="ou_current_bot"),
    )


@pytest.mark.asyncio
async def test_on_message_accepts_group_bot_message_when_explicitly_mentioned() -> None:
    channel = _make_feishu_channel()
    channel.config.group_policy = "open"
    captured = []

    async def _capture(**kwargs):
        captured.append(kwargs)

    channel._handle_message = _capture
    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(
            _make_feishu_event(
                chat_type="group",
                sender_type="bot",
                sender_open_id="ou_other_bot",
                mentions=[_bot_mention()],
                content='{"text": "@_user_1 请检查任务"}',
            )
        )

    assert len(captured) == 1
    assert captured[0]["sender_id"] == "ou_other_bot"
    assert captured[0]["chat_id"] == "oc_abc"
    assert captured[0]["content"].startswith("[Message from another Feishu bot")
    assert "@Jarvis 请检查任务" in captured[0]["content"]
    assert "@_user_1" not in captured[0]["content"]
    assert captured[0]["metadata"]["sender_type"] == "bot"
    assert captured[0]["metadata"]["sender_open_id"] == "ou_other_bot"
    assert captured[0]["metadata"]["mentions"] == [{
        "key": "@_user_1",
        "name": "Jarvis",
        "open_id": "ou_current_bot",
    }]


@pytest.mark.asyncio
async def test_on_message_ignores_undirected_group_bot_message_even_when_open() -> None:
    channel = _make_feishu_channel()
    channel.config.group_policy = "open"
    channel._handle_message = AsyncMock()

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(
            _make_feishu_event(
                chat_type="group",
                sender_type="bot",
                sender_open_id="ou_other_bot",
            )
        )

    channel._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_ignores_bot_direct_message() -> None:
    channel = _make_feishu_channel()
    channel._handle_message = AsyncMock()

    with patch.object(channel, "_add_reaction", return_value=None):
        await channel._on_message(
            _make_feishu_event(
                chat_type="p2p",
                sender_type="bot",
                sender_open_id="ou_other_bot",
                mentions=[_bot_mention()],
            )
        )

    channel._handle_message.assert_not_awaited()
