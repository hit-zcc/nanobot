"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ITERATION_WARNING_MESSAGE = (
    "⚠️ This task has used {used_iterations}/{max_iterations} model/tool iterations "
    "and has {remaining_iterations} remaining before the hard limit."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_DEFAULT_REFUSAL_MESSAGE = (
    "The model declined to continue on this request (safety refusal). "
    "Try rephrasing, breaking it into smaller steps, or using a different model."
)


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    refusal_message: str | None = _DEFAULT_REFUSAL_MESSAGE
    refusal_fallback_model: str | None = None
    max_iterations_message: str | None = None
    iteration_warning_remaining: int | None = None
    iteration_warning_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    # Tools report nothing while they run, so a single long exec used to leave
    # the channel silent for its whole timeout (up to 10 minutes). Emit liveness
    # notices instead, backing off so a long job doesn't flood the chat: first
    # at 20s, then 30s, 45s, … capped at 2 minutes.
    _TOOL_HEARTBEAT_FIRST = 20.0
    _TOOL_HEARTBEAT_MAX = 120.0
    _TOOL_HEARTBEAT_GROWTH = 1.5

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        current_model = spec.model
        downgraded = False

        for iteration in range(spec.max_iterations):
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            remaining_iterations = spec.max_iterations - iteration
            if (
                spec.iteration_warning_remaining is not None
                and spec.max_iterations > spec.iteration_warning_remaining
                and remaining_iterations == spec.iteration_warning_remaining
            ):
                template = (
                    spec.iteration_warning_message
                    or _DEFAULT_ITERATION_WARNING_MESSAGE
                )
                await hook.on_notice(
                    context,
                    template.format(
                        used_iterations=iteration,
                        max_iterations=spec.max_iterations,
                        remaining_iterations=remaining_iterations,
                    ),
                )
            kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": current_model,
            }
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(context, delta)

                response = await self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            else:
                response = await self.provider.chat_with_retry(**kwargs)

            raw_usage = response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            # A safety refusal truncates the turn (often mid-tool-call), so it
            # must be handled BEFORE tool execution — otherwise a partial
            # tool_use runs and the refusal signal is lost. First refusal:
            # transparently retry on the stronger fallback model. If that model
            # also refuses (or no fallback is set), surface the refusal message.
            if response.finish_reason == "refusal":
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                fallback = spec.refusal_fallback_model
                if fallback and not downgraded and fallback != current_model:
                    logger.warning(
                        "Model {} refused (safety stop); retrying on {}",
                        current_model, fallback,
                    )
                    downgraded = True
                    current_model = fallback
                    await hook.on_notice(
                        context,
                        f"⚠️ 这条被模型安全策略拒答了，正在用更强的模型（{fallback}）重试…",
                    )
                    await hook.after_iteration(context)
                    continue
                clean = hook.finalize_content(context, response.content)
                final_content = clean or spec.refusal_message or _DEFAULT_REFUSAL_MESSAGE
                stop_reason = "refusal"
                context.final_content = final_content
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            # Output hit the max_tokens ceiling in the middle of a tool call:
            # streamed tool-call JSON is truncated, and partial-JSON parsing
            # silently drops incomplete trailing fields (e.g. write_file losing
            # its `content` argument). Executing such a call would corrupt data,
            # so skip execution and feed the truncation back for a chunked retry.
            if response.finish_reason == "length" and response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)
                logger.warning(
                    "Output truncated by max_tokens mid tool call ({}); skipping execution",
                    ", ".join(tc.name for tc in response.tool_calls),
                )
                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                for tc in response.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": (
                            "Error: the model output hit the max_tokens limit in the "
                            "middle of this tool call, so its arguments were truncated "
                            "and it was NOT executed. Retry with a smaller payload — "
                            "e.g. write the file in several chunks (a first write_file "
                            "with the opening part, then append the rest via smaller "
                            "edits/appends), or split the command into smaller steps."
                        ),
                    })
                await hook.after_iteration(context)
                continue

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                tools_used.extend(tc.name for tc in response.tool_calls)

                await hook.before_execute_tools(context)

                results, new_events, fatal_error = await self._execute_tools(
                    spec, response.tool_calls, hook, context,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
                await hook.after_iteration(context)
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=False)

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            messages.append(build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
        )

    async def _tool_heartbeat(
        self,
        hook: AgentHook,
        context: AgentHookContext,
        pending: dict[int, str],
    ) -> None:
        """Notify *hook* at backing-off intervals while *pending* tools run."""
        started = time.monotonic()
        delay = self._TOOL_HEARTBEAT_FIRST
        while True:
            await asyncio.sleep(delay)
            if not pending:
                return
            try:
                await hook.on_tool_heartbeat(
                    context,
                    elapsed=time.monotonic() - started,
                    pending=sorted(set(pending.values())),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # A broken progress channel must never stall tool execution.
                logger.exception("Tool heartbeat hook failed; stopping notices")
                return
            delay = min(delay * self._TOOL_HEARTBEAT_GROWTH, self._TOOL_HEARTBEAT_MAX)

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        # Drop each entry as its tool settles so the heartbeat reports only what
        # is genuinely still running.
        pending = {i: tc.name for i, tc in enumerate(tool_calls)}

        async def _run(index: int, tool_call: ToolCallRequest):
            try:
                return await self._run_tool(spec, tool_call)
            finally:
                pending.pop(index, None)

        beat: asyncio.Task | None = None
        if hook is not None and context is not None:
            beat = asyncio.create_task(self._tool_heartbeat(hook, context, pending))
        try:
            if spec.concurrent_tools:
                tool_results = await asyncio.gather(*(
                    _run(i, tool_call)
                    for i, tool_call in enumerate(tool_calls)
                ))
            else:
                tool_results = [
                    await _run(i, tool_call)
                    for i, tool_call in enumerate(tool_calls)
                ]
        finally:
            if beat is not None:
                beat.cancel()
                with suppress(asyncio.CancelledError):
                    await beat

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {
            "name": tool_call.name,
            "status": "error" if isinstance(result, str) and result.startswith("Error") else "ok",
            "detail": detail,
        }, None
