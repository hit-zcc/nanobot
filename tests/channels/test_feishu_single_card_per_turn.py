"""One turn should arrive as one card, even when it pauses to run tools.

A streaming card is sealed by turning ``streaming_mode`` off, after which any
further text has to open a new card. The agent loop ends a stream every time
the model stops to call a tool, so sealing on every stream end turned a single
answer into one notification per tool call. The loop already distinguishes the
two cases via ``_resuming``; these tests pin down that the Feishu side honours it.
"""

try:
    from nanobot.channels import feishu
    FEISHU_AVAILABLE = getattr(feishu, "FEISHU_AVAILABLE", False)
except ImportError:
    FEISHU_AVAILABLE = False

if not FEISHU_AVAILABLE:
    import pytest
    pytest.skip("Feishu dependencies not installed (lark-oapi)", allow_module_level=True)

import pytest

from nanobot.channels.feishu import (
    _STREAM_ELEMENT_ID,
    FeishuChannel,
    _FeishuStreamBuf,
)


def _channel(*, single_card: bool = True) -> FeishuChannel:
    channel = FeishuChannel.__new__(FeishuChannel)
    channel._client = object()
    channel._stream_bufs = {}
    channel._reply_targets = {}
    channel.config = type("Cfg", (), {
        "reply_to_message": False,
        "single_card_per_turn": single_card,
    })()
    channel.closed = []
    channel.updates = []

    def close(card_id, sequence):
        channel.closed.append(card_id)
        return True

    def update(card_id, text, sequence, element_id=_STREAM_ELEMENT_ID):
        channel.updates.append((card_id, text))
        return True

    channel._close_streaming_mode_sync = close
    channel._stream_update_text_sync = update
    return channel


def _seed(channel: FeishuChannel, chat_id: str, text: str) -> _FeishuStreamBuf:
    buf = _FeishuStreamBuf()
    buf.text = text
    buf.card_id = "card-1"
    channel._stream_bufs[chat_id] = buf
    return buf


@pytest.mark.asyncio
async def test_pause_for_tools_keeps_the_card_open() -> None:
    channel = _channel()
    _seed(channel, "oc_1", "第一段")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})

    assert channel.closed == [], "card was sealed mid-turn, so the next segment starts a new one"
    assert channel.updates == [("card-1", "第一段")], "text so far should still be flushed"
    assert "oc_1" in channel._stream_bufs, "buffer must survive so the card keeps growing"


@pytest.mark.asyncio
async def test_second_segment_grows_the_same_card() -> None:
    channel = _channel()
    _seed(channel, "oc_1", "第一段")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})
    channel._stream_bufs["oc_1"].text += "第二段"
    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": False})

    assert [card for card, _ in channel.updates] == ["card-1", "card-1"]
    assert channel.updates[-1][1] == "第一段第二段"


@pytest.mark.asyncio
async def test_real_end_seals_the_card() -> None:
    """Without this the chat list preview stays stuck on the typing placeholder."""
    channel = _channel()
    _seed(channel, "oc_1", "全部说完")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": False})

    assert channel.closed == ["card-1"]
    assert "oc_1" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_switch_off_restores_a_card_per_segment() -> None:
    channel = _channel(single_card=False)
    _seed(channel, "oc_1", "第一段")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})

    assert channel.closed == ["card-1"]
    assert "oc_1" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_pause_before_any_card_exists_keeps_the_text() -> None:
    """Nothing has rendered yet, so there is nothing to seal — but the words
    already spoken must not be dropped on the floor."""
    channel = _channel()
    buf = _FeishuStreamBuf()
    buf.text = "还没建卡"
    channel._stream_bufs["oc_1"] = buf

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})

    assert channel.updates == []
    assert channel.closed == []
    assert channel._stream_bufs["oc_1"].text == "还没建卡"


@pytest.mark.asyncio
async def test_interjection_seals_the_old_card() -> None:
    """An interjection changes the question, so the answer to it belongs in its
    own card — one that quotes the new message rather than the original one."""
    channel = _channel()
    _seed(channel, "oc_1", "回答旧问题")

    await channel.send_delta(
        "oc_1", "回答新问题",
        {"_stream_delta": True, "_card_break": True, "message_id": "om_new"},
    )

    assert channel.closed == ["card-1"], "the card answering the old question must be sealed"
    assert channel.updates[0] == ("card-1", "回答旧问题"), "its text must be flushed first"
    assert channel._stream_bufs["oc_1"].card_id is None, "next delta opens a fresh card"
    assert "回答新问题" in channel._stream_bufs["oc_1"].text


@pytest.mark.asyncio
async def test_segments_are_separated_by_a_rule() -> None:
    """One card per turn hides the seams where work happened, so the segments
    run together as a single block. A rule marks where the pause was."""
    channel = _channel()
    _seed(channel, "oc_1", "第一段")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})
    await channel.send_delta("oc_1", "第二段", {"_stream_delta": True})

    assert channel._stream_bufs["oc_1"].text == "第一段\n\n---\n\n第二段"


@pytest.mark.asyncio
async def test_no_trailing_rule_when_nothing_follows() -> None:
    """A pause that is never followed by more text must not leave the card
    ending on a horizontal rule."""
    channel = _channel()
    _seed(channel, "oc_1", "只有一段")

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": True})
    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": False})

    assert channel.updates[-1][1] == "只有一段"
