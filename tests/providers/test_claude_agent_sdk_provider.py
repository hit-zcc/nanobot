from __future__ import annotations

from pathlib import Path

from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider


def test_sdk_options_are_isolated_from_claude_code_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
    provider = ClaudeAgentSDKProvider("claude-agent-sdk/claude-opus-5", tmp_path)

    options = provider.sdk_options(
        tools=[],
        mcp_servers={"nanobot": object()},
        allowed_tools=["mcp__nanobot__read_file"],
    )

    assert options.model == "claude-opus-5"
    assert options.cwd == Path(tmp_path)
    assert options.cli_path == "/usr/local/bin/claude"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.skills == []
    assert options.strict_mcp_config is True
    assert options.allowed_tools == ["mcp__nanobot__read_file"]


def test_split_prompt_keeps_system_separate_and_flattens_history():
    system, prompt = ClaudeAgentSDKProvider.split_prompt([
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest"},
    ])

    assert system == "system rules"
    assert "<USER>\nfirst\n</USER>" in prompt
    assert "<ASSISTANT>\nreply\n</ASSISTANT>" in prompt
    assert prompt.endswith("<USER>\nlatest\n</USER>")


def test_runner_factory_does_not_change_codex_path():
    from nanobot.agent.runner import AgentRunner
    from nanobot.agent.runner_factory import build_agent_runner

    ordinary_provider = object()
    assert type(build_agent_runner(ordinary_provider)) is AgentRunner

    sdk_provider = ClaudeAgentSDKProvider("claude-agent-sdk/claude-opus-5")
    assert type(build_agent_runner(sdk_provider)).__name__ == "ClaudeAgentSDKRunner"
