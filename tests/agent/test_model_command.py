"""`/model` switches the running agent, not just the label it reports.

Everything that cached the provider at construction — the runner, subagents,
the memory consolidator — has to move with it, or the next subagent call goes
to the old backend carrying the new model's name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nanobot.command.builtin import cmd_model
from nanobot.config.loader import load_config, set_config_path
from nanobot.providers.base import GenerationSettings


CONFIG = {
    "agents": {
        "defaults": {
            "model": "openai-codex/gpt-5.6-sol",
            "provider": "openai_codex",
            "maxTokens": 16384,
            "contextWindowTokens": 245760,
        },
        "presets": {
            "claude": {
                "model": "claude-oauth/claude-opus-5",
                "provider": "claude_oauth",
                "contextWindowTokens": 200000,
            },
        },
    },
}


class _Provider:
    def __init__(self, model: str) -> None:
        self.default_model = model
        self.generation = GenerationSettings(max_tokens=16384)


def _loop() -> SimpleNamespace:
    old = _Provider("openai-codex/gpt-5.6-sol")
    loop = SimpleNamespace(
        model="openai-codex/gpt-5.6-sol",
        provider=old,
        context_window_tokens=245760,
        runner=SimpleNamespace(provider=old),
        subagents=SimpleNamespace(provider=old, model="openai-codex/gpt-5.6-sol",
                                  runner=SimpleNamespace(provider=old)),
        memory_consolidator=SimpleNamespace(retune=lambda **kw: retuned.update(kw)),
    )
    from nanobot.agent.loop import AgentLoop
    loop.switch_model = lambda *a, **kw: AgentLoop.switch_model(loop, *a, **kw)
    return loop


retuned: dict = {}


def _ctx(loop, args: str) -> SimpleNamespace:
    msg = SimpleNamespace(channel="feishu", chat_id="a")
    return SimpleNamespace(loop=loop, session=None, key="feishu:a", raw=f"/model {args}",
                           args=args, msg=msg)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    set_config_path(path)
    monkeypatch.setattr(
        "nanobot.providers.factory.build_provider",
        lambda config: _Provider(config.agents.defaults.model),
    )
    retuned.clear()
    yield path
    set_config_path(None)


@pytest.mark.asyncio
async def test_preset_switch_rebinds_every_provider_holder(config_path):
    loop = _loop()

    result = await cmd_model(_ctx(loop, "claude"))

    assert "claude-oauth/claude-opus-5" in result.content
    assert loop.model == "claude-oauth/claude-opus-5"
    assert loop.context_window_tokens == 200000
    for holder in (loop.runner, loop.subagents, loop.subagents.runner):
        assert holder.provider is loop.provider
    assert loop.subagents.model == "claude-oauth/claude-opus-5"
    assert retuned["model"] == "claude-oauth/claude-opus-5"
    assert retuned["context_window_tokens"] == 200000


@pytest.mark.asyncio
async def test_preset_switch_is_persisted_without_touching_other_keys(config_path):
    await cmd_model(_ctx(_loop(), "claude"))

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = saved["agents"]["defaults"]
    assert defaults["model"] == "claude-oauth/claude-opus-5"
    assert defaults["provider"] == "claude_oauth"
    assert defaults["contextWindowTokens"] == 200000
    assert defaults["maxTokens"] == 16384  # untouched by the preset
    assert saved["agents"]["presets"]["claude"]["model"] == "claude-oauth/claude-opus-5"


@pytest.mark.asyncio
async def test_full_model_id_clears_the_forced_provider(config_path):
    loop = _loop()

    await cmd_model(_ctx(loop, "anthropic/claude-sonnet-4-5"))

    assert loop.model == "anthropic/claude-sonnet-4-5"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    # Left as "openai_codex", the forced provider would outrank the prefix.
    assert saved["agents"]["defaults"]["provider"] == "auto"


@pytest.mark.asyncio
async def test_unknown_alias_lists_presets_and_changes_nothing(config_path):
    loop = _loop()

    result = await cmd_model(_ctx(loop, "gemini"))

    assert "Unknown model preset" in result.content
    assert "claude" in result.content
    assert loop.model == "openai-codex/gpt-5.6-sol"
    assert load_config().agents.defaults.model == "openai-codex/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_bare_command_reports_current_model_and_presets(config_path):
    result = await cmd_model(_ctx(_loop(), ""))

    assert "openai-codex/gpt-5.6-sol" in result.content
    assert "claude → claude-oauth/claude-opus-5" in result.content
