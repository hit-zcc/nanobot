"""An interjection redirects the current turn's replies — and only that turn.

When the user interrupts a running turn, the answer from that point on is
addressed to the new message, so replies should quote it. The subtlety is where
that retargeting is allowed to happen: the interjection queue is also drained
when a run is cancelled and when leftovers are replayed as a fresh turn. Doing
the retarget inside the drain helper meant those paths re-armed a stale quote
*after* the turn had already cleaned up, and the next turn answered a new
question while pointing at an old one.
"""

from types import SimpleNamespace

import pytest

from nanobot.agent.loop import AgentLoop


class _Loop:
    """Just the reply-target bookkeeping, without booting a whole agent."""

    _REPLY_KEYS = AgentLoop._REPLY_KEYS
    _take_injections = AgentLoop._take_injections
    _retarget_replies_to = AgentLoop._retarget_replies_to

    def __init__(self) -> None:
        self._pending_injections: dict[str, list] = {}
        self._injection_reply_meta: dict[str, dict[str, str]] = {}
        self._card_break: set[str] = set()


def _msg(message_id: str):
    return SimpleNamespace(metadata={"message_id": message_id})


def test_retarget_points_replies_at_the_interjection() -> None:
    loop = _Loop()

    loop._retarget_replies_to("s1", _msg("om_new"))

    assert loop._injection_reply_meta["s1"] == {"message_id": "om_new"}
    assert "s1" in loop._card_break, "the card answering the old question must be sealed"


def test_draining_the_queue_does_not_retarget() -> None:
    """A cancelled run and a replayed leftover both drain the queue. Neither is
    an answer in flight, so neither may leave a quote armed behind them."""
    loop = _Loop()
    loop._pending_injections["s1"] = [_msg("om_new")]

    drained = loop._take_injections("s1")

    assert len(drained) == 1
    assert loop._injection_reply_meta == {}
    assert loop._card_break == set()


def test_retarget_does_not_survive_the_turn() -> None:
    """The bug this pins down: the next turn answered its own question while
    still quoting the previous turn's interjection."""
    loop = _Loop()
    loop._retarget_replies_to("s1", _msg("om_new"))

    # End-of-turn cleanup, as performed by _on_dispatch_done.
    loop._injection_reply_meta.pop("s1", None)
    loop._card_break.discard("s1")
    # A cancelled run then drains whatever was queued.
    loop._pending_injections["s1"] = [_msg("om_stale")]
    loop._take_injections("s1")

    assert loop._injection_reply_meta == {}, "stale quote leaked into the next turn"


def test_latest_interjection_wins() -> None:
    loop = _Loop()

    loop._retarget_replies_to("s1", _msg("om_first"))
    loop._retarget_replies_to("s1", _msg("om_second"))

    assert loop._injection_reply_meta["s1"] == {"message_id": "om_second"}


def test_interjection_without_routing_keys_is_ignored() -> None:
    """A message with nothing to quote must not blank out the existing target."""
    loop = _Loop()
    loop._retarget_replies_to("s1", _msg("om_new"))

    loop._retarget_replies_to("s1", SimpleNamespace(metadata={}))

    assert loop._injection_reply_meta["s1"] == {"message_id": "om_new"}


def test_sessions_do_not_bleed_into_each_other() -> None:
    loop = _Loop()

    loop._retarget_replies_to("s1", _msg("om_a"))
    loop._retarget_replies_to("s2", _msg("om_b"))

    assert loop._injection_reply_meta["s1"] == {"message_id": "om_a"}
    assert loop._injection_reply_meta["s2"] == {"message_id": "om_b"}
