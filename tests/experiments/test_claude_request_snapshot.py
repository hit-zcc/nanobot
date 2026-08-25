"""Tests for the local-only Claude request snapshot experiment."""
import json
from uuid import UUID

import httpx
import pytest

from nanobot.experiments import claude_request_snapshot as snapshot
from nanobot.providers.claude_oauth_provider import ClaudeOAuthProvider


def test_three_identity_mappings_are_exact():
    assert snapshot.IDENTITIES == {
        "interactive": "You are Claude Code, Anthropic's official CLI for Claude.",
        "noninteractive_append": (
            "You are Claude Code, Anthropic's official CLI for Claude, running within "
            "the Claude Agent SDK."
        ),
        "noninteractive_plain": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
    }


def test_current_nanobot_prefix_is_captured():
    result = snapshot.build_snapshot(["interactive"])
    assert result["nanobot_current"]["dimensions"]["system_first_block"] == (
        ClaudeOAuthProvider._CLAUDE_CODE_SYSTEM_PREFIX
    )


def test_never_loads_credentials_or_refreshes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("credential/login/refresh function was called")

    for name in ("_load_token", "_load_claude_code_token", "_read_claude_code_keychain",
                 "_refresh_access_token", "login_claude_oauth"):
        monkeypatch.setattr(snapshot.oauth_module, name, forbidden)
    result = snapshot.build_snapshot()
    rendered = json.dumps(result)
    assert snapshot.MOCK_TOKEN in rendered
    assert result["safety"]["credential_files_read"] == []
    assert "access_token" not in rendered and "refresh_token" not in rendered


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com/v1/messages", "https://claude.com/",
    "https://claude.ai/oauth/authorize", "https://platform.claude.com/v1/oauth/token",
    "http://example.com/", "https://127.0.0.1/v1/messages",
])
def test_network_guard_rejects_every_nonlocal_or_https_target(url):
    with pytest.raises(ValueError, match="network guard rejected"):
        snapshot.guard_local_url(url)


def test_reverse_validation_real_anthropic_host_turns_red():
    with pytest.raises(ValueError, match="api.anthropic.com"):
        snapshot.capture_with_mock_transport(
            {"test": True}, {"Authorization": f"Bearer {snapshot.MOCK_TOKEN}"},
            "https://api.anthropic.com/v1/messages",
        )


def test_mock_transport_capture_is_local_and_socketless(monkeypatch):
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("socket construction is forbidden")

    monkeypatch.setattr("socket.socket", forbidden_socket)
    capture = snapshot.capture_with_mock_transport(
        {"fixture": True}, {"Authorization": f"Bearer {snapshot.MOCK_TOKEN}"},
    )
    assert capture["target_host"] == "127.0.0.1"
    assert capture["transport"] == "httpx.MockTransport (no socket)"


def test_session_ids_are_shape_only_or_fixed_test_uuid():
    result = snapshot.build_snapshot()
    assert UUID(result["fixture"]["test_session_uuid"])
    assert result["fixture"]["contains_real_session_id"] is False
    assert result["nanobot_current"]["dimensions"]["session_id_shape"] == (
        "32 lowercase hex characters"
    )
    for reference in result["claude_code_static_references"].values():
        assert reference["headers"]["X-Claude-Code-Session-Id"] == snapshot.TEST_SESSION_UUID


def test_snapshot_is_stable_and_json_round_trips():
    first, second = snapshot.build_snapshot(), snapshot.build_snapshot()
    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first


def test_body_captures_tools_thinking_and_cache_control():
    d = snapshot.build_snapshot(["interactive"])["nanobot_current"]["dimensions"]
    assert d["model"] == "claude-sonnet-4-20250514"
    assert d["messages"] and d["tools"][0]["name"] == "snapshot_test_tool"
    assert d["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert d["cache_control_locations"] == ["system", "messages", "tools"]


def test_report_statistics_are_derived_from_checks():
    result = snapshot.build_snapshot()
    report = snapshot.render_markdown(result)
    for comparison in result["comparisons"].values():
        checks, stats = comparison["checks"], comparison["stats"]
        assert stats["matches"] == sum(checks.values())
        assert stats["differences"] == len(checks) - sum(checks.values())
        assert f"{stats['matches']}/{stats['compared']}" in report


def test_only_mock_transport_is_constructed(monkeypatch):
    real_client = httpx.Client

    def guarded_client(*args, **kwargs):
        assert isinstance(kwargs.get("transport"), httpx.MockTransport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(snapshot.httpx, "Client", guarded_client)
    snapshot.build_snapshot(["interactive"])
