"""Tests for Feishu streaming (send_delta) via CardKit streaming API."""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels import feishu
from nanobot.channels.feishu import FeishuChannel, FeishuConfig, _FeishuStreamBuf


def _make_channel(streaming: bool = True) -> FeishuChannel:
    config = FeishuConfig(
        enabled=True,
        app_id="cli_test",
        app_secret="secret",
        allow_from=["*"],
        streaming=streaming,
    )
    ch = FeishuChannel(config, MessageBus())
    ch._client = MagicMock()
    ch._loop = None
    return ch


def _mock_create_card_response(card_id: str = "card_stream_001"):
    resp = MagicMock()
    resp.success.return_value = True
    resp.data = SimpleNamespace(card_id=card_id)
    return resp


def _mock_send_response(message_id: str = "om_stream_001"):
    resp = MagicMock()
    resp.success.return_value = True
    resp.data = SimpleNamespace(message_id=message_id)
    return resp


def _mock_content_response(success: bool = True):
    resp = MagicMock()
    resp.success.return_value = success
    resp.code = 0 if success else 99999
    resp.msg = "ok" if success else "error"
    return resp


class TestFeishuStreamingConfig:
    def test_streaming_default_true(self):
        assert FeishuConfig().streaming is True

    def test_supports_streaming_when_enabled(self):
        ch = _make_channel(streaming=True)
        assert ch.supports_streaming is True

    def test_supports_streaming_disabled(self):
        ch = _make_channel(streaming=False)
        assert ch.supports_streaming is False


class TestCreateStreamingCard:
    def test_returns_card_id_on_success(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.create.return_value = _mock_create_card_response("card_123")
        ch._client.im.v1.message.create.return_value = _mock_send_response()
        result = ch._create_streaming_card_sync("chat_id", "oc_chat1")
        assert result == "card_123"
        ch._client.cardkit.v1.card.create.assert_called_once()
        ch._client.im.v1.message.create.assert_called_once()

    def test_returns_none_on_failure(self):
        ch = _make_channel()
        resp = MagicMock()
        resp.success.return_value = False
        resp.code = 99999
        resp.msg = "error"
        ch._client.cardkit.v1.card.create.return_value = resp
        assert ch._create_streaming_card_sync("chat_id", "oc_chat1") is None

    def test_returns_none_on_exception(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.create.side_effect = RuntimeError("network")
        assert ch._create_streaming_card_sync("chat_id", "oc_chat1") is None

    def test_returns_none_when_card_send_fails(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.create.return_value = _mock_create_card_response("card_123")
        resp = MagicMock()
        resp.success.return_value = False
        resp.code = 99999
        resp.msg = "error"
        resp.get_log_id.return_value = "log1"
        ch._client.im.v1.message.create.return_value = resp
        assert ch._create_streaming_card_sync("chat_id", "oc_chat1") is None


class TestCloseStreamingMode:
    def test_returns_true_on_success(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.settings.return_value = _mock_content_response(True)
        assert ch._close_streaming_mode_sync("card_1", 10) is True

    def test_returns_false_on_failure(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.settings.return_value = _mock_content_response(False)
        assert ch._close_streaming_mode_sync("card_1", 10) is False

    def test_returns_false_on_exception(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.settings.side_effect = RuntimeError("err")
        assert ch._close_streaming_mode_sync("card_1", 10) is False


class TestStreamUpdateText:
    def test_reports_ok_on_success(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response(True)
        assert ch._stream_update_text_sync("card_1", "hello", 1) == feishu._STREAM_OK

    def test_reports_failed_on_failure(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response(False)
        assert ch._stream_update_text_sync("card_1", "hello", 1) == feishu._STREAM_FAILED

    def test_reports_failed_on_exception(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card_element.content.side_effect = RuntimeError("err")
        assert ch._stream_update_text_sync("card_1", "hello", 1) == feishu._STREAM_FAILED

    @pytest.mark.parametrize("code", [200850, 300309])
    def test_reports_dead_once_feishu_closed_the_stream(self, code):
        """Retrying these on the same card is pointless — it will never take another
        update. The caller has to move the rest of the answer to a new card."""
        ch = _make_channel()
        resp = _mock_content_response(False)
        resp.code = code
        ch._client.cardkit.v1.card_element.content.return_value = resp
        assert ch._stream_update_text_sync("card_1", "hello", 1) == feishu._STREAM_DEAD


class TestSendDelta:
    @pytest.mark.asyncio
    async def test_first_delta_creates_card_and_sends(self):
        ch = _make_channel()
        ch._client.cardkit.v1.card.create.return_value = _mock_create_card_response("card_new")
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_new")
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()

        await ch.send_delta("oc_chat1", "Hello ")

        assert "oc_chat1" in ch._stream_bufs
        buf = ch._stream_bufs["oc_chat1"]
        assert buf.text == "Hello "
        assert buf.card_id == "card_new"
        assert buf.sequence == 1
        ch._client.cardkit.v1.card.create.assert_called_once()
        ch._client.im.v1.message.create.assert_called_once()
        ch._client.cardkit.v1.card_element.content.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_delta_does_not_auto_mention(self):
        """流式卡片同样不再自动 @ 提问者（2026-08-20 聪聪要求）。"""
        ch = _make_channel()
        ch._reply_targets["om_user"] = "ou_alice"
        ch._client.cardkit.v1.card.create.return_value = _mock_create_card_response("card_new")
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_new")
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()

        await ch.send_delta(
            "oc_chat1",
            "查清楚了",
            metadata={"message_id": "om_user"},
        )

        buf = ch._stream_bufs["oc_chat1"]
        assert buf.text == "查清楚了"
        assert buf.mention_target is None
        assert buf.mention_pending is False

    @pytest.mark.asyncio
    async def test_streaming_does_not_hold_back_on_leading_at(self):
        """开头是 ``@`` 时不再等"显示名收全"——没有要剥的东西了，直接开卡。"""
        ch = _make_channel()
        ch._reply_targets["om_user"] = "ou_alice"
        ch._client.cardkit.v1.card.create.return_value = _mock_create_card_response("card_new")
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_new")
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()

        await ch.send_delta(
            "oc_chat1",
            "@w",
            metadata={"message_id": "om_user"},
        )
        await ch.send_delta(
            "oc_chat1",
            "uw Prompt 侧单独改以下 4 项即可",
            metadata={"message_id": "om_user"},
        )

        assert ch._stream_bufs["oc_chat1"].text == "@wuw Prompt 侧单独改以下 4 项即可"
        ch._client.cardkit.v1.card.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_delta_within_interval_skips_update(self):
        ch = _make_channel()
        buf = _FeishuStreamBuf(text="Hello ", card_id="card_1", sequence=1, last_edit=time.monotonic())
        ch._stream_bufs["oc_chat1"] = buf

        await ch.send_delta("oc_chat1", "world")

        assert buf.text == "Hello world"
        ch._client.cardkit.v1.card_element.content.assert_not_called()

    @pytest.mark.asyncio
    async def test_delta_after_interval_updates_text(self):
        ch = _make_channel()
        buf = _FeishuStreamBuf(text="Hello ", card_id="card_1", sequence=1, last_edit=time.monotonic() - 1.0)
        ch._stream_bufs["oc_chat1"] = buf

        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()
        await ch.send_delta("oc_chat1", "world")

        assert buf.text == "Hello world"
        assert buf.sequence == 2
        ch._client.cardkit.v1.card_element.content.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_end_sends_final_update(self):
        ch = _make_channel()
        ch._stream_bufs["oc_chat1"] = _FeishuStreamBuf(
            text="Final content", card_id="card_1", sequence=3, last_edit=0.0,
        )
        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()
        ch._client.cardkit.v1.card.settings.return_value = _mock_content_response()

        await ch.send_delta("oc_chat1", "", metadata={"_stream_end": True})

        assert "oc_chat1" not in ch._stream_bufs
        ch._client.cardkit.v1.card_element.content.assert_called_once()
        ch._client.cardkit.v1.card.settings.assert_called_once()
        settings_call = ch._client.cardkit.v1.card.settings.call_args[0][0]
        assert settings_call.body.sequence == 5  # after final content seq 4

    @pytest.mark.asyncio
    async def test_stream_end_fallback_when_no_card_id(self):
        """If card creation failed, stream_end falls back to a plain card message."""
        ch = _make_channel()
        ch._stream_bufs["oc_chat1"] = _FeishuStreamBuf(
            text="Fallback content", card_id=None, sequence=0, last_edit=0.0,
        )
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_fb")

        await ch.send_delta("oc_chat1", "", metadata={"_stream_end": True})

        assert "oc_chat1" not in ch._stream_bufs
        ch._client.cardkit.v1.card_element.content.assert_not_called()
        ch._client.im.v1.message.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_end_without_buf_is_noop(self):
        ch = _make_channel()
        await ch.send_delta("oc_chat1", "", metadata={"_stream_end": True})
        ch._client.cardkit.v1.card_element.content.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_delta_skips_send(self):
        ch = _make_channel()
        await ch.send_delta("oc_chat1", "   ")

        assert "oc_chat1" in ch._stream_bufs
        ch._client.cardkit.v1.card.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_client_returns_early(self):
        ch = _make_channel()
        ch._client = None
        await ch.send_delta("oc_chat1", "text")
        assert "oc_chat1" not in ch._stream_bufs

    @pytest.mark.asyncio
    async def test_sequence_increments_correctly(self):
        ch = _make_channel()
        buf = _FeishuStreamBuf(text="a", card_id="card_1", sequence=5, last_edit=0.0)
        ch._stream_bufs["oc_chat1"] = buf

        ch._client.cardkit.v1.card_element.content.return_value = _mock_content_response()
        await ch.send_delta("oc_chat1", "b")
        assert buf.sequence == 6

        buf.last_edit = 0.0  # reset to bypass throttle
        await ch.send_delta("oc_chat1", "c")
        assert buf.sequence == 7


class TestSendMessageReturnsId:
    def test_returns_message_id_on_success(self):
        ch = _make_channel()
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_abc")
        result = ch._send_message_sync("chat_id", "oc_chat1", "text", '{"text":"hi"}')
        assert result == "om_abc"

    def test_returns_none_on_failure(self):
        ch = _make_channel()
        resp = MagicMock()
        resp.success.return_value = False
        resp.code = 99999
        resp.msg = "error"
        resp.get_log_id.return_value = "log1"
        ch._client.im.v1.message.create.return_value = resp
        result = ch._send_message_sync("chat_id", "oc_chat1", "text", '{"text":"hi"}')
        assert result is None


class TestToolProgressMessages:
    @staticmethod
    def _message(content: str, *, done: bool = False, **metadata) -> OutboundMessage:
        return OutboundMessage(
            channel="feishu",
            chat_id="oc_chat1",
            content=content,
            metadata={
                "_progress": True,
                "_tool_progress_id": "progress-1",
                "_tool_progress_done": done,
                **metadata,
            },
        )

    @pytest.mark.asyncio
    async def test_creates_once_then_updates_and_completes(self):
        ch = _make_channel()
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_progress")
        ch._client.im.v1.message.update.return_value = _mock_content_response()

        await ch.send(self._message("⏳ 正在执行命令 · 已用时 20 秒"))
        await ch.send(self._message("⏳ 正在执行命令 · 已用时 50 秒"))
        await ch.send(self._message("✅ 执行命令完成 · 共用时 1 分 2 秒", done=True))

        assert ch._client.im.v1.message.create.call_count == 1
        assert ch._client.im.v1.message.update.call_count == 2
        update_request = ch._client.im.v1.message.update.call_args_list[0].args[0]
        assert update_request.body.msg_type == "text"
        assert json.loads(update_request.body.content) == {
            "text": "⏳ 正在执行命令 · 已用时 50 秒"
        }
        ch._client.im.v1.message.patch.assert_not_called()
        assert "progress-1" not in ch._tool_progress_messages
        assert "progress-1" not in ch._tool_progress_last_notified

    @pytest.mark.asyncio
    async def test_long_task_creates_fresh_liveness_notice(self):
        ch = _make_channel()
        ch.config.progress_notify_interval_s = 120
        ch._client.im.v1.message.create.side_effect = [
            _mock_send_response("om_progress_1"),
            _mock_send_response("om_progress_2"),
        ]

        await ch.send(self._message("⏳ 正在执行命令 · 已用时 20 秒"))
        ch._tool_progress_last_notified["progress-1"] -= 120
        await ch.send(self._message("⏳ 正在执行命令 · 已用时 2 分"))

        assert ch._client.im.v1.message.create.call_count == 2
        ch._client.im.v1.message.update.assert_not_called()
        assert ch._tool_progress_messages["progress-1"] == "om_progress_2"
        fresh_request = ch._client.im.v1.message.create.call_args.args[0]
        assert json.loads(fresh_request.body.content) == {
            "text": "⏳ 正在执行命令 · 已用时 2 分"
        }

    @pytest.mark.asyncio
    async def test_completion_without_heartbeat_is_noop(self):
        ch = _make_channel()

        await ch.send(self._message("✅ 执行命令完成 · 共用时 1 秒", done=True))

        ch._client.im.v1.message.create.assert_not_called()
        ch._client.im.v1.message.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure_does_not_create_replacement(self):
        ch = _make_channel()
        ch._client.im.v1.message.create.return_value = _mock_send_response("om_progress")
        ch._client.im.v1.message.update.return_value = _mock_content_response(False)

        await ch.send(self._message("⏳ 正在执行命令 · 已用时 20 秒"))
        await ch.send(self._message("⏳ 正在执行命令 · 已用时 50 秒"))

        assert ch._client.im.v1.message.create.call_count == 1
        assert ch._client.im.v1.message.update.call_count == 1
        assert ch._tool_progress_messages["progress-1"] == "om_progress"

    @pytest.mark.asyncio
    async def test_thread_progress_replies_once_and_caches_reply_id(self):
        ch = _make_channel()
        ch._client.im.v1.message.reply.return_value = _mock_send_response("om_reply_progress")

        await ch.send(self._message(
            "⏳ 正在执行命令 · 已用时 20 秒",
            thread_id="omt_thread",
            root_id="om_root",
        ))

        ch._client.im.v1.message.reply.assert_called_once()
        ch._client.im.v1.message.create.assert_not_called()
        assert ch._tool_progress_messages["progress-1"] == "om_reply_progress"


class TestSanitizeCardMarkup:
    """Markup the model gets wrong that the card `markdown` element rejects (code 11311)."""

    def test_rewrites_font_tag_with_known_color(self):
        out = feishu._sanitize_card_markup("<font color='red'>警告</font>文本")
        assert out == "<text_tag color='red'>警告</text_tag>文本"

    def test_drops_font_tag_with_unknown_color(self):
        out = feishu._sanitize_card_markup("<font color='pink'>警告</font>文本")
        assert out == "警告文本"

    def test_drops_text_tag_with_unknown_color(self):
        out = feishu._sanitize_card_markup("<text_tag color='pink'>警告</text_tag>文本")
        assert out == "警告文本"

    def test_drops_number_tag_with_unknown_color(self):
        out = feishu._sanitize_card_markup("<number_tag color='magenta'>1</number_tag>步骤")
        assert out == "1步骤"

    def test_keeps_text_tag_with_known_color(self):
        text = "<text_tag color='blue'>信息</text_tag>"
        assert feishu._sanitize_card_markup(text) == text

    def test_leaves_plain_text_untouched(self):
        text = "普通文本，没有任何标签"
        assert feishu._sanitize_card_markup(text) == text
