"""The heartbeat must not outlive a `/model` switch on its own agent.

`HeartbeatService` used to snapshot ``provider`` and ``model`` at gateway
startup. ``AgentLoop.switch_model`` rebinds the runner, the subagent manager
and the memory consolidator in place, but it never knew about the heartbeat —
so after a switch the heartbeat kept waking up on the *old* backend for the
rest of the process's life. Observed in production: main chat and subagents on
claude-agent-sdk/claude-opus-5, heartbeat still calling openai-codex/gpt-5.6-sol
43 minutes later.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest

from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class _RecordingProvider(LLMProvider):
    """Records the model each call was made with, and always says 'skip'."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.models: list[str | None] = []
        self.default_model = name
        self.generation = GenerationSettings(max_tokens=4096)

    async def chat(self, *args, model=None, **kwargs) -> LLMResponse:
        self.models.append(model)
        return LLMResponse(
            content="",
            tool_calls=[ToolCallRequest(id="hb", name="heartbeat", arguments={"action": "skip"})],
        )

    def get_default_model(self) -> str:
        return self.name


@pytest.mark.asyncio
async def test_decide_follows_the_agent_after_a_model_switch(tmp_path) -> None:
    old = _RecordingProvider("openai-codex/gpt-5.6-sol")
    new = _RecordingProvider("claude-agent-sdk/claude-opus-5")
    agent = SimpleNamespace(provider=old, model="openai-codex/gpt-5.6-sol")

    service = HeartbeatService(
        workspace=tmp_path,
        provider=old,
        model="openai-codex/gpt-5.6-sol",
        agent=agent,
    )

    await service._decide("- [ ] something")
    assert old.models == ["openai-codex/gpt-5.6-sol"]

    # What AgentLoop.switch_model does to the loop, in place.
    agent.provider = new
    agent.model = "claude-agent-sdk/claude-opus-5"

    await service._decide("- [ ] something")

    assert service.provider is new
    assert service.model == "claude-agent-sdk/claude-opus-5"
    assert new.models == ["claude-agent-sdk/claude-opus-5"]
    assert old.models == ["openai-codex/gpt-5.6-sol"]  # no second call on the old backend


@pytest.mark.asyncio
async def test_tick_evaluator_also_uses_the_switched_backend(tmp_path, monkeypatch) -> None:
    """Phase 3 (notify-or-not) is a second provider call and must switch too."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check things", encoding="utf-8")

    old = _RecordingProvider("openai-codex/gpt-5.6-sol")
    new = _RecordingProvider("claude-agent-sdk/claude-opus-5")
    agent = SimpleNamespace(provider=new, model="claude-agent-sdk/claude-opus-5")

    async def _on_execute(tasks: str) -> str:
        return "result"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=old,
        model="openai-codex/gpt-5.6-sol",
        on_execute=_on_execute,
        on_notify=None,
        agent=agent,
    )
    # Phase 1 must say "run" so phase 3 is reached.
    async def _run_decision(*args, **kwargs):
        return "run", "check things"

    monkeypatch.setattr(service, "_decide", _run_decision)

    seen: dict = {}

    async def _evaluate(response, task_context, provider, model):
        seen["provider"], seen["model"] = provider, model
        return False

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _evaluate)

    await service._tick()

    assert seen["provider"] is new
    assert seen["model"] == "claude-agent-sdk/claude-opus-5"


def test_explicit_provider_still_wins_when_there_is_no_agent(tmp_path) -> None:
    """CLI/tests construct the service without an agent; nothing changes for them."""
    provider = _RecordingProvider("openai/gpt-4o-mini")
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="openai/gpt-4o-mini")

    assert service.provider is provider
    assert service.model == "openai/gpt-4o-mini"

    other = _RecordingProvider("other")
    service.provider = other
    service.model = "other"
    assert service.provider is other
    assert service.model == "other"


def test_gateway_wires_the_heartbeat_to_its_agent() -> None:
    """The fix is worthless if the gateway forgets to pass ``agent=``."""
    from nanobot.cli import commands

    tree = ast.parse(textwrap.dedent(inspect.getsource(commands.gateway)))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HeartbeatService"
    ]
    assert constructions, "gateway no longer constructs a HeartbeatService"
    for call in constructions:
        kwargs = {kw.arg for kw in call.keywords}
        assert "agent" in kwargs, (
            "HeartbeatService must be given agent=<the heartbeat agent> so it "
            "follows /model instead of freezing the startup provider"
        )


def test_switch_model_reaches_a_heartbeat_bound_to_the_loop(tmp_path, monkeypatch) -> None:
    """End-to-end through the real `/model` command, not a hand-rolled switch."""
    import asyncio

    from nanobot.agent.loop import AgentLoop
    from nanobot.command.builtin import cmd_model
    from nanobot.config.loader import set_config_path

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "model": "openai-codex/gpt-5.6-sol",
                        "provider": "openai_codex",
                    },
                    "presets": {
                        "claude-sdk": {
                            "model": "claude-agent-sdk/claude-opus-5",
                            "provider": "claude_agent_sdk",
                            "contextWindowTokens": 200000,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    set_config_path(config_path)
    monkeypatch.setattr(
        "nanobot.providers.factory.build_provider",
        lambda config: _RecordingProvider(config.agents.defaults.model),
    )

    try:
        old = _RecordingProvider("openai-codex/gpt-5.6-sol")
        loop = SimpleNamespace(
            model="openai-codex/gpt-5.6-sol",
            provider=old,
            context_window_tokens=245760,
            runner=SimpleNamespace(provider=old),
            subagents=SimpleNamespace(
                provider=old, model="openai-codex/gpt-5.6-sol",
                runner=SimpleNamespace(provider=old),
            ),
            memory_consolidator=SimpleNamespace(retune=lambda **kw: None),
        )
        loop.switch_model = lambda *a, **kw: AgentLoop.switch_model(loop, *a, **kw)

        heartbeat = HeartbeatService(
            workspace=tmp_path,
            provider=old,
            model="openai-codex/gpt-5.6-sol",
            agent=loop,
        )

        ctx = SimpleNamespace(
            loop=loop, session=None, key="feishu:a", raw="/model claude-sdk",
            args="claude-sdk", msg=SimpleNamespace(channel="feishu", chat_id="a"),
        )
        asyncio.run(cmd_model(ctx))

        assert loop.model == "claude-agent-sdk/claude-opus-5"
        assert heartbeat.model == "claude-agent-sdk/claude-opus-5"
        assert heartbeat.provider is loop.provider
        assert heartbeat.provider is not old
    finally:
        set_config_path(None)
