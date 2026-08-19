"""Surviving markup the card refuses to parse.

The failure these cover looked like the bot hanging: one `<number_tag>` in a
long answer, and the streaming card stopped updating for the rest of the turn.
Nothing retried, because retrying the same cumulative buffer reproduces the
same rejection every time.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from nanobot.channels import feishu


class TestNumberTagSanitising:
    """`number_tag` is an unread badge (1-99 + background_color), not a colour block."""

    def test_drops_bare_number_tag(self):
        # The shape that actually froze cards on 2026-08-13.
        assert feishu._sanitize_card_markup("共 <number_tag>101</number_tag> 次") == "共 101 次"

    def test_drops_number_tag_with_color_attribute(self):
        """`color=` is text_tag's attribute; number_tag rejects it."""
        out = feishu._sanitize_card_markup("<number_tag color='blue'>101</number_tag>")
        assert out == "101"

    def test_drops_number_tag_over_99(self):
        out = feishu._sanitize_card_markup("<number_tag background_color='blue'>101</number_tag>")
        assert out == "101"

    def test_drops_number_tag_with_non_digit_content(self):
        out = feishu._sanitize_card_markup("<number_tag background_color='blue'>abc</number_tag>")
        assert out == "abc"

    def test_keeps_the_one_legal_shape(self):
        text = "<number_tag background_color='blue'>51</number_tag>"
        assert feishu._sanitize_card_markup(text) == text

    def test_text_tag_still_survives(self):
        """The fix must not take the working colour block down with it."""
        text = "路径 <text_tag color='orange'>`/tmp` 13G</text_tag> 大"
        assert feishu._sanitize_card_markup(text) == text


class TestStripCardMarkup:
    def test_removes_tags_keeps_words(self):
        out = feishu._strip_card_markup("a <text_tag color='red'>b</text_tag> c")
        assert out == "a b c"

    def test_removes_known_tags_with_unknown_attributes(self):
        """Attribute combinations are what the model keeps inventing."""
        out = feishu._strip_card_markup("<number_tag weird='1'>9</number_tag>x")
        assert out == "9x"

    def test_leaves_tags_the_card_has_no_opinion_about(self):
        """Probed 2026-08-13: the card takes an unknown tag as literal text."""
        assert feishu._strip_card_markup("<weird_tag>x</weird_tag>") == "<weird_tag>x</weird_tag>"

    def test_leaves_plain_text_alone(self):
        assert feishu._strip_card_markup("普通文本") == "普通文本"


def _channel_with_writes(results):
    """A channel whose card writes return *results* in order, recording content."""
    ch = feishu.FeishuChannel.__new__(feishu.FeishuChannel)
    sent = []

    def fake_write(card_id, content, sequence, element_id=feishu._STREAM_ELEMENT_ID):
        sent.append(content)
        return results[len(sent) - 1]

    ch._stream_update_text_sync = fake_write
    return ch, sent


def _buf(text):
    buf = feishu._FeishuStreamBuf()
    buf.text = text
    buf.card_id = "card-1"
    return buf


class TestBadMarkupRecovery:
    """A rejected frame must not take the rest of the turn with it."""

    @pytest.mark.asyncio
    async def test_retries_the_frame_as_plain_text(self):
        ch, sent = _channel_with_writes([feishu._STREAM_BAD_MARKUP, feishu._STREAM_OK])
        buf = _buf("共 <number_tag weird>101</number_tag> 次")

        result = await ch._stream_write(buf, buf.text, asyncio.get_running_loop())

        assert result == feishu._STREAM_OK
        assert sent[1] == "共 101 次", "the retry must carry the words without the tags"

    @pytest.mark.asyncio
    async def test_disarms_the_buffer_so_later_frames_are_clean(self):
        """The actual bug: the poison lived in the cumulative buffer."""
        ch, sent = _channel_with_writes(
            [feishu._STREAM_BAD_MARKUP, feishu._STREAM_OK, feishu._STREAM_OK],
        )
        buf = _buf("共 <number_tag weird>101</number_tag> 次")
        loop = asyncio.get_running_loop()

        await ch._stream_write(buf, buf.text, loop)
        buf.text += "，后续内容"
        await ch._stream_write(buf, buf.text, loop)

        assert "number_tag" not in sent[2]
        assert sent[2] == "共 101 次，后续内容"

    @pytest.mark.asyncio
    async def test_sequence_keeps_increasing_across_the_retry(self):
        """Feishu drops frames whose sequence is not strictly increasing."""
        ch, _ = _channel_with_writes([feishu._STREAM_BAD_MARKUP, feishu._STREAM_OK])
        buf = _buf("<number_tag weird>101</number_tag>")

        await ch._stream_write(buf, buf.text, asyncio.get_running_loop())

        assert buf.sequence == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_when_there_is_no_markup_to_drop(self):
        """A rejection with nothing tag-shaped in it is not ours to fix."""
        ch, sent = _channel_with_writes([feishu._STREAM_BAD_MARKUP])
        buf = _buf("没有任何标签的纯文本")

        result = await ch._stream_write(buf, buf.text, asyncio.get_running_loop())

        assert result == feishu._STREAM_FAILED
        assert len(sent) == 1, "no second call should be spent guessing"

    @pytest.mark.asyncio
    async def test_a_dead_card_is_not_treated_as_bad_markup(self):
        """200850 means the stream is closed; stripping tags would not help."""
        ch, sent = _channel_with_writes([feishu._STREAM_DEAD])
        buf = _buf("<text_tag color='red'>x</text_tag>")

        result = await ch._stream_write(buf, buf.text, asyncio.get_running_loop())

        assert result == feishu._STREAM_DEAD
        assert len(sent) == 1
        assert buf.card_id is None, "the dead card should still be retired"
