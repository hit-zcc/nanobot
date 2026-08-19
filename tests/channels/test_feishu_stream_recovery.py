"""A streaming card Feishu closed under us must not swallow the rest of the turn.

Feishu ends a card's streaming session on its own — 200850 once the card has sat
too long without an update, then 300309 for every write after that. The channel
used to log those and carry on writing to the same card, so an answer that paused
for a long tool call finished into a card nobody could see updating; worse, the
buffer holding that dead card outlived the turn and every later answer in the
chat went the same way. These tests pin down the three defences: keep the card
warm while tools run, move to a new card when one dies anyway, and never inherit
a card from a previous turn.
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
    _STREAM_DEAD,
    _STREAM_ELEMENT_ID,
    _STREAM_OK,
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
    channel.sent = []
    channel.created = []
    # Result of the next _stream_update_text_sync call, per card id.
    channel.update_results = {}
    channel.next_card_id = iter(["card-2", "card-3", "card-4"])

    def close(card_id, sequence):
        channel.closed.append(card_id)
        return True

    def update(card_id, text, sequence, element_id=_STREAM_ELEMENT_ID):
        channel.updates.append((card_id, text))
        return channel.update_results.get(card_id, _STREAM_OK)

    def create(rid_type, chat_id, reply_message_id=None):
        card_id = next(channel.next_card_id)
        channel.created.append(card_id)
        return card_id

    def send(rid_type, chat_id, msg_type, content):
        channel.sent.append(content)
        return "om_1"

    channel._close_streaming_mode_sync = close
    channel._stream_update_text_sync = update
    channel._create_streaming_card_sync = create
    channel._send_message_sync = send
    return channel


def _seed(channel: FeishuChannel, chat_id: str, text: str, *, turn: str = "turn-a") -> _FeishuStreamBuf:
    """An open card mid-turn, with everything so far already rendered."""
    buf = _FeishuStreamBuf()
    buf.text = text
    buf.card_id = "card-1"
    buf.rendered = len(text)
    buf.turn_id = turn
    channel._stream_bufs[chat_id] = buf
    return buf


def _delta_meta(turn: str = "turn-a", segment: int = 0) -> dict:
    return {"_stream_delta": True, "_stream_id": f"{turn}:{segment}"}


@pytest.mark.asyncio
async def test_dead_card_is_replaced_and_the_tail_continues() -> None:
    channel = _channel()
    _seed(channel, "oc_1", "已经说完的部分")
    channel.update_results["card-1"] = _STREAM_DEAD

    # This delta hits the dead card...
    await channel.send_delta("oc_1", "后面的部分", _delta_meta())
    # ...and the next one must land somewhere the user can see.
    channel._stream_bufs["oc_1"].last_edit = 0.0
    await channel.send_delta("oc_1", "还有更多", _delta_meta())

    assert channel.created == ["card-2"], "a dead card must be replaced, not written to forever"
    assert channel._stream_bufs["oc_1"].card_id == "card-2"
    assert channel.updates[-1] == ("card-2", "后面的部分还有更多")


@pytest.mark.asyncio
async def test_the_new_card_does_not_repeat_what_the_dead_one_shows() -> None:
    """The dead card keeps displaying what it rendered; only the tail moves on."""
    channel = _channel()
    _seed(channel, "oc_1", "已渲染")
    channel.update_results["card-1"] = _STREAM_DEAD

    await channel.send_delta("oc_1", "未渲染", _delta_meta())

    assert channel.created == ["card-2"]
    assert channel.updates[-1] == ("card-2", "未渲染")


@pytest.mark.asyncio
async def test_a_card_that_dies_at_the_end_still_delivers_the_tail() -> None:
    """No later delta is coming, so the leftover text has to be sent outright."""
    channel = _channel()
    _seed(channel, "oc_1", "已渲染")
    channel._stream_bufs["oc_1"].text += "未渲染"
    channel.update_results["card-1"] = _STREAM_DEAD

    await channel.send_delta("oc_1", "", {"_stream_end": True, "_resuming": False})

    assert channel.closed == [], "a card Feishu already closed must not be sealed again"
    assert len(channel.sent) == 1, "the tail must reach the chat as an ordinary message"
    assert "未渲染" in channel.sent[0]
    assert "已渲染" not in channel.sent[0], "the dead card already shows this"
    assert "oc_1" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_tool_heartbeat_keeps_the_card_warm() -> None:
    """The silence during a long tool call is what makes Feishu close the stream."""
    channel = _channel()
    buf = _seed(channel, "oc_1", "工具执行前说的话")
    buf.last_edit = 0.0  # long past the keepalive interval

    await channel._keepalive_stream_card("oc_1", _loop())

    assert channel.updates == [("card-1", "工具执行前说的话")]
    assert buf.last_edit > 0.0


@pytest.mark.asyncio
async def test_keepalive_does_not_fire_on_a_card_just_touched() -> None:
    import time

    channel = _channel()
    buf = _seed(channel, "oc_1", "刚说过话")
    buf.last_edit = time.monotonic()

    await channel._keepalive_stream_card("oc_1", _loop())

    assert channel.updates == []


@pytest.mark.asyncio
async def test_keepalive_replaces_a_card_that_already_died() -> None:
    channel = _channel()
    buf = _seed(channel, "oc_1", "说到一半")
    buf.last_edit = 0.0
    channel.update_results["card-1"] = _STREAM_DEAD

    await channel._keepalive_stream_card("oc_1", _loop())

    assert buf.card_id is None, "the next delta opens a fresh card"


@pytest.mark.asyncio
async def test_a_new_turn_never_inherits_the_previous_turn_s_card() -> None:
    """A cancelled turn cannot always close its own card. If one is left open,
    the next answer must not grow inside it — that card may already be dead, in
    which case the new answer would vanish."""
    channel = _channel()
    _seed(channel, "oc_1", "上一轮的回答", turn="turn-a")

    await channel.send_delta("oc_1", "这一轮的回答", _delta_meta(turn="turn-b"))

    assert channel.closed == ["card-1"], "the leftover card is sealed"
    assert channel.created == ["card-2"], "the new turn opens its own card"
    assert channel._stream_bufs["oc_1"].text == "这一轮的回答"
    assert channel._stream_bufs["oc_1"].turn_id == "turn-b"


@pytest.mark.asyncio
async def test_the_same_turn_keeps_growing_one_card() -> None:
    """The guard must not fire between segments of a single turn."""
    channel = _channel()
    buf = _seed(channel, "oc_1", "第一段")
    buf.last_edit = 0.0

    await channel.send_delta("oc_1", "第二段", _delta_meta(turn="turn-a", segment=1))

    assert channel.created == []
    assert channel.updates[-1] == ("card-1", "第一段第二段")


def _loop():
    import asyncio
    return asyncio.get_running_loop()
