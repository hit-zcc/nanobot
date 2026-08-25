"""飞书不渲染 `<u>`，出站必须剥掉。

这不是理论问题：`<u>要点</u>` 会被飞书当普通文字原样显示，用户看到的是
一串带尖括号的标签。而发送端 API 照常返回成功，模型对污染零感知 ——
人格档案里已经写了"别用 `<u>`"，2026-08-19 / 08-20 仍连着犯了两轮。
所以规则只能落在出站链路上，这些用例就是那道拦截的护栏。
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.channels import feishu


class TestUnderlineSanitising:
    def test_strips_the_tag_and_keeps_the_words(self):
        assert feishu._sanitize_underline_tags("<u>要点</u>") == "要点"

    def test_is_case_insensitive(self):
        assert feishu._sanitize_underline_tags("<U>要点</U>") == "要点"
        assert feishu._sanitize_underline_tags("<u>要点</U>") == "要点"

    def test_strips_tags_carrying_attributes(self):
        out = feishu._sanitize_underline_tags('<u class="x">要点</u>')
        assert out == "要点"

    def test_strips_the_self_closing_form(self):
        assert feishu._sanitize_underline_tags("前<u/>后") == "前后"
        assert feishu._sanitize_underline_tags("前<u />后") == "前后"

    def test_strips_every_occurrence(self):
        out = feishu._sanitize_underline_tags("<u>甲</u> 和 <u>乙</u>")
        assert out == "甲 和 乙"

    def test_leaves_tags_that_merely_start_with_u(self):
        """`<ul>` / `<underline>` 不是我们要剥的东西，不能误伤。"""
        assert feishu._sanitize_underline_tags("<ul><li>a</li></ul>") == "<ul><li>a</li></ul>"
        assert feishu._sanitize_underline_tags("<underline>x</underline>") == "<underline>x</underline>"

    def test_leaves_plain_text_alone(self):
        assert feishu._sanitize_underline_tags("普通文本") == "普通文本"


class TestUnderlineInCardMarkup:
    """接进 `_sanitize_card_markup` 的调用链后才算真的生效。"""

    def test_card_markup_strips_underline(self):
        assert feishu._sanitize_card_markup("<u>要点</u>") == "要点"

    def test_card_markup_strips_uppercase_and_attributes(self):
        assert feishu._sanitize_card_markup('<U class="x">要点</U>') == "要点"

    def test_does_not_hurt_neighbouring_card_tags(self):
        """text_tag 是卡片能正常渲染的东西，剥 `<u>` 不能顺手把它带走。"""
        out = feishu._sanitize_card_markup("<u>标题</u> <text_tag color='red'>警告</text_tag>")
        assert out == "标题 <text_tag color='red'>警告</text_tag>"

    def test_does_not_hurt_mention_normalisation(self):
        out = feishu._sanitize_card_markup('<u>提醒</u> <at user_id="ou_bob">牛小数</at>')
        assert out == '提醒 <at id="ou_bob"></at>'


def _buf(text):
    buf = feishu._FeishuStreamBuf()
    buf.text = text
    buf.card_id = "card-1"
    return buf


class TestUnderlineOnStreamingPath:
    """增量卡片走的是 `_stream_write`，也必须被拦下来。"""

    @pytest.mark.asyncio
    async def test_streaming_frame_carries_no_underline_tags(self):
        ch = feishu.FeishuChannel.__new__(feishu.FeishuChannel)
        sent = []

        def fake_write(card_id, content, sequence, element_id=feishu._STREAM_ELEMENT_ID):
            sent.append(content)
            return feishu._STREAM_OK

        ch._stream_update_text_sync = fake_write
        buf = _buf('结论：<u class="k">要点</u>先看这里')

        await ch._stream_write(buf, buf.text, asyncio.get_running_loop())

        assert sent == ["结论：要点先看这里"]
        assert "<u" not in sent[0]


class TestUnderlineOnPlainTextPath:
    """text / post 消息不经过 `_sanitize_card_markup`，得在 send() 里拦。"""

    def _channel(self):
        ch = feishu.FeishuChannel.__new__(feishu.FeishuChannel)
        ch._client = MagicMock()
        ch.config = MagicMock(reply_to_message=False)
        ch._inbound_chats = {}
        ch._outbound_routes = {}
        sent = []

        def fake_send(receive_id_type, receive_id, msg_type, content):
            sent.append((msg_type, content))
            return "om_1"

        ch._send_message_sync = fake_send
        ch._remember = lambda *a, **kw: None
        ch._outbound_mention_target = lambda metadata: None
        return ch, sent

    @pytest.mark.asyncio
    async def test_short_text_message_is_stripped(self):
        ch, sent = self._channel()

        await ch.send(OutboundMessage(channel="feishu", chat_id="oc_1", content="重点是<u>这句</u>，别忘了"))

        assert len(sent) == 1
        msg_type, body = sent[0]
        assert msg_type == "text"
        assert json.loads(body)["text"] == "重点是这句，别忘了"

    @pytest.mark.asyncio
    async def test_post_message_is_stripped(self):
        ch, sent = self._channel()
        content = "见 [文档](https://example.com) 里<u>加粗那段</u>"

        await ch.send(OutboundMessage(channel="feishu", chat_id="oc_1", content=content))

        msg_type, body = sent[0]
        assert msg_type == "post"
        assert "<u>" not in body and "</u>" not in body
        assert "加粗那段" in body
