"""Select an agent runner without changing existing provider behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanobot.agent.runner import AgentRunner

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


def build_agent_runner(provider: LLMProvider):
    """Return the SDK runner only for its explicit marker provider."""
    if getattr(provider, "uses_agent_sdk", False) is True:
        from nanobot.agent.claude_sdk_runner import ClaudeAgentSDKRunner

        return ClaudeAgentSDKRunner(provider)
    return AgentRunner(provider)
