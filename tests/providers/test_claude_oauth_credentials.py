"""Tests for Claude OAuth credential resolution."""

import json
import time

import pytest

from nanobot.providers import claude_oauth_provider as mod

LIVE = time.time() + 3600
DEAD = time.time() - 3600


def _write_nanobot_token(path, expires_at):
    path.write_text(json.dumps({
        "access_token": "nanobot-token",
        "refresh_token": "nanobot-refresh",
        "expires_at": expires_at,
        "token_type": "Bearer",
    }))


@pytest.fixture
def token_file(tmp_path, monkeypatch):
    path = tmp_path / "claude-oauth.json"
    monkeypatch.setattr(mod, "_token_path", lambda: path)
    monkeypatch.setattr(mod, "_claude_code_credentials_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: None)
    return path


def _keychain_blob(expires_at, **extra):
    """Claude Code's blob shape, including fields nanobot doesn't model."""
    return json.dumps({"claudeAiOauth": {
        "accessToken": "cc-token",
        "refreshToken": "cc-refresh",
        "expiresAt": expires_at * 1000,  # Claude Code stores milliseconds
        "subscriptionType": "max",
        "scopes": ["user:inference", "user:profile"],
        **extra,
    }})


def test_prefers_live_nanobot_token(token_file, monkeypatch):
    _write_nanobot_token(token_file, LIVE)
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: _keychain_blob(LIVE))

    assert mod._load_token()["access_token"] == "nanobot-token"


def test_falls_back_to_keychain_when_local_token_expired(token_file, monkeypatch):
    """A revoked local token must not shadow a live Claude Code login."""
    _write_nanobot_token(token_file, DEAD)
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: _keychain_blob(LIVE))

    token = mod._load_token()
    assert token["access_token"] == "cc-token"
    assert token["source"] == mod.SOURCE_KEYCHAIN


def test_credentials_file_wins_over_keychain(token_file, tmp_path, monkeypatch):
    creds = tmp_path / "creds.json"
    creds.write_text(_keychain_blob(LIVE).replace("cc-token", "file-token"))
    monkeypatch.setattr(mod, "_claude_code_credentials_path", lambda: creds)
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: _keychain_blob(LIVE))

    assert mod._load_token()["access_token"] == "file-token"


def test_expired_local_token_returned_when_nothing_is_live(token_file):
    """Still hand back a refreshable token so _ensure_token can renew it."""
    _write_nanobot_token(token_file, DEAD)

    token = mod._load_token()
    assert token["access_token"] == "nanobot-token"
    assert "source" not in token


def test_no_credentials_anywhere(token_file):
    assert mod._load_token() is None


def test_source_marks_which_store_a_token_came_from(token_file, monkeypatch):
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: _keychain_blob(LIVE))
    assert mod._load_token()["source"] == mod.SOURCE_KEYCHAIN


def test_merge_preserves_claude_code_only_fields():
    """Write-back must not drop state the `claude` CLI depends on."""
    merged = mod._merge_claude_code_credentials(
        _keychain_blob(DEAD, rateLimitTier="default"),
        {"access_token": "new-access", "refresh_token": "new-refresh", "expires_at": LIVE},
    )

    oauth = json.loads(merged)["claudeAiOauth"]
    assert oauth["accessToken"] == "new-access"
    assert oauth["refreshToken"] == "new-refresh"
    assert oauth["expiresAt"] == int(LIVE * 1000)  # milliseconds, integer
    # Untouched Claude Code fields survive.
    assert oauth["subscriptionType"] == "max"
    assert oauth["scopes"] == ["user:inference", "user:profile"]
    assert oauth["rateLimitTier"] == "default"


def test_merge_keeps_existing_refresh_token_when_not_rotated():
    merged = mod._merge_claude_code_credentials(
        _keychain_blob(DEAD),
        {"access_token": "new-access", "refresh_token": "", "expires_at": LIVE},
    )
    assert json.loads(merged)["claudeAiOauth"]["refreshToken"] == "cc-refresh"


def test_merge_rejects_a_malformed_blob():
    """A parse failure must abort the write-back, never clobber the store."""
    assert mod._merge_claude_code_credentials("not json", {}) is None
    assert mod._merge_claude_code_credentials('{"other": 1}', {}) is None


def test_refresh_writes_rotated_token_back_to_keychain(token_file, monkeypatch):
    """Rotation must land in Claude Code's store, not only in nanobot's file."""
    written = {}
    monkeypatch.setattr(mod, "_read_claude_code_keychain", lambda: _keychain_blob(DEAD))
    monkeypatch.setattr(
        mod, "_write_claude_code_keychain",
        lambda raw: written.setdefault("raw", raw) is not None,
    )

    class _Resp:
        status_code = 200
        @staticmethod
        def json():
            return {"access_token": "rotated-access", "refresh_token": "rotated-refresh",
                    "expires_in": 28800}

    class _Client:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): return _Resp()

    monkeypatch.setattr(mod.httpx, "Client", lambda **kw: _Client())

    result = mod._refresh_access_token("cc-refresh", mod.SOURCE_KEYCHAIN)

    assert result["access_token"] == "rotated-access"
    oauth = json.loads(written["raw"])["claudeAiOauth"]
    assert oauth["accessToken"] == "rotated-access"
    assert oauth["refreshToken"] == "rotated-refresh"
    # A borrowed token stays in Claude Code's store; no stale nanobot copy.
    assert not token_file.exists()
