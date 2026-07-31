from nanobot.cli.commands import _pick_delivery_target


class FakeSessions:
    def list_sessions(self):
        return [
            {"key": "feishu.jarvis:work-user", "updated_at": "2026-07-30T22:45:00"},
            {"key": "feishu:life-user", "updated_at": "2026-07-30T02:08:00"},
        ]


def test_explicit_heartbeat_target_never_follows_recent_session() -> None:
    target = _pick_delivery_target(
        FakeSessions(),
        {"feishu", "feishu.jarvis"},
        preferred_channel="feishu",
        preferred_chat_id="life-user",
    )

    assert target == ("feishu", "life-user")


def test_preferred_channel_filters_recent_session() -> None:
    target = _pick_delivery_target(
        FakeSessions(),
        {"feishu", "feishu.jarvis"},
        preferred_channel="feishu",
    )

    assert target == ("feishu", "life-user")
