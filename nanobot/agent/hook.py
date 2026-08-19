"""Shared lifecycle hook primitives for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nanobot.providers.base import LLMResponse, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to runner hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    # Recipes looked up before the tools ran, keyed by tool_call id. Filled by
    # a hook in before_execute_tools and consumed when the results are turned
    # into messages, so the advice and the output it concerns arrive together.
    preflight_notes: dict[str, str] = field(default_factory=dict)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """Minimal lifecycle surface for shared runner customization."""

    def wants_streaming(self) -> bool:
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def on_notice(self, context: AgentHookContext, message: str) -> None:
        """Surface an out-of-band status note to the user (e.g. a model downgrade)."""
        pass

    async def on_tool_heartbeat(
        self, context: AgentHookContext, *, elapsed: float, pending: list[str],
    ) -> None:
        """Report that *pending* tools are still running, *elapsed* seconds in.

        Only fires while tool execution outlives ``_TOOL_HEARTBEAT_FIRST``, so
        fast tools stay silent. Intervals back off (see ``AgentRunner``) to keep
        a multi-minute job from flooding the channel.
        """
        pass

    async def take_injections(self, context: AgentHookContext) -> list[dict[str, Any]]:
        """Return user messages to splice into the run at an iteration boundary.

        Lets a user interrupt a long run: anything they send while tools are
        executing is handed to the model on the next iteration instead of
        waiting for the whole turn to finish. Returning an empty list (the
        default) keeps the run untouched.
        """
        return []

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content
