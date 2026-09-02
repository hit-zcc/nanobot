"""Agent runner backed by the official Claude Agent SDK.

The SDK owns Claude's inner model/tool loop.  Nanobot's tools are exposed as a
single isolated in-process MCP server, so built-in Claude Code tools cannot
bypass nanobot's validation, workspace restrictions, hooks, or audit trail.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.providers.base import ToolCallRequest
from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider
from nanobot.utils.helpers import build_assistant_message


class ClaudeAgentSDKRunner:
    """Run one nanobot turn through an isolated official SDK client."""

    _TOOL_HEARTBEAT_FIRST = 20.0
    _TOOL_HEARTBEAT_MAX = 120.0
    _TOOL_HEARTBEAT_GROWTH = 1.5

    def __init__(self, provider: ClaudeAgentSDKProvider):
        self.provider = provider

    @staticmethod
    def _mcp_content(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, str):
            return [{"type": "text", "text": result}]
        if not isinstance(result, list):
            return [{"type": "text", "text": str(result)}]
        content: list[dict[str, Any]] = []
        for block in result:
            if not isinstance(block, dict):
                content.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") == "text":
                content.append({"type": "text", "text": str(block.get("text") or "")})
                continue
            if block.get("type") == "image_url":
                url = ((block.get("image_url") or {}).get("url") or "")
                if url.startswith("data:") and ";base64," in url:
                    header, data = url.split(",", 1)
                    content.append({
                        "type": "image",
                        "data": data,
                        "mimeType": header[5:].split(";", 1)[0],
                    })
                    continue
            content.append({"type": "text", "text": str(block)})
        return content

    async def _execute_with_heartbeat(
        self,
        spec: AgentRunSpec,
        hook,
        context: AgentHookContext,
        call: ToolCallRequest,
    ) -> Any:
        task = asyncio.create_task(spec.tools.execute(call.name, call.arguments))
        started = time.monotonic()
        delay = self._TOOL_HEARTBEAT_FIRST
        try:
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=delay)
                except asyncio.TimeoutError:
                    await hook.on_tool_heartbeat(
                        context,
                        elapsed=time.monotonic() - started,
                        pending=[call.name],
                    )
                    delay = min(delay * self._TOOL_HEARTBEAT_GROWTH, self._TOOL_HEARTBEAT_MAX)
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeSDKClient,
                ResultMessage,
                SdkMcpTool,
                StreamEvent,
                TextBlock,
                create_sdk_mcp_server,
            )
        except ImportError as exc:
            message = (
                "Claude Agent SDK is not installed. Install nanobot with the "
                "'claude-agent-sdk' extra."
            )
            return AgentRunResult(
                final_content=message,
                messages=list(spec.initial_messages),
                stop_reason="error",
                error=str(exc),
            )

        hook = spec.hook
        if hook is None:
            from nanobot.agent.hook import AgentHook
            hook = AgentHook()
        messages = spec.message_sink if spec.message_sink is not None else []
        messages.clear()
        messages.extend(spec.initial_messages)
        system, prompt = self.provider.split_prompt(spec.initial_messages)
        tools_used: list[str] = []
        tool_events: list[dict[str, str]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        final_content: str | None = None
        result_error: str | None = None
        result_stop = "completed"
        iteration = 0

        sdk_tools = []
        allowed_tools = []
        for definition in spec.tools.get_definitions():
            function = definition.get("function") or definition
            name = function.get("name")
            if not name:
                continue
            description = function.get("description") or "Nanobot tool"
            schema = function.get("parameters") or {"type": "object", "properties": {}}

            async def handler(args: dict[str, Any], *, _name=name) -> dict[str, Any]:
                nonlocal iteration
                iteration += 1
                call = ToolCallRequest(
                    id=f"sdk-{time.time_ns()}", name=_name, arguments=dict(args or {})
                )
                context = AgentHookContext(
                    iteration=iteration - 1,
                    messages=messages,
                    tool_calls=[call],
                )
                await hook.before_iteration(context)
                await hook.before_execute_tools(context)
                tools_used.append(_name)
                messages.append(build_assistant_message(
                    None, tool_calls=[call.to_openai_tool_call()]
                ))
                result = await self._execute_with_heartbeat(spec, hook, context, call)
                note = context.preflight_notes.get(call.id)
                if note:
                    if isinstance(result, str):
                        result += note
                    elif isinstance(result, list):
                        result = [*result, {"type": "text", "text": note}]
                context.tool_results = [result]
                context.tool_events = [{"tool": _name, "status": "completed"}]
                tool_events.extend(context.tool_events)
                injections = await hook.take_injections(context)
                if injections:
                    injected = "\n\n".join(
                        self.provider._content_text(item.get("content")) for item in injections
                    )
                    suffix = f"\n\n[New user message while this tool ran]\n{injected}"
                    if isinstance(result, str):
                        result += suffix
                    else:
                        result = [*result, {"type": "text", "text": suffix}]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": _name,
                    "content": result,
                })
                await hook.after_iteration(context)
                is_error = isinstance(result, str) and result.startswith("Error")
                return {"content": self._mcp_content(result), "is_error": is_error}

            sdk_tools.append(SdkMcpTool(
                name=name,
                description=description,
                input_schema=schema,
                handler=handler,
            ))
            allowed_tools.append(f"mcp__nanobot__{name}")

        server = create_sdk_mcp_server(name="nanobot", version="1.0.0", tools=sdk_tools)
        options = self.provider.sdk_options(
            model=self.provider.strip_model_prefix(spec.model),
            system_prompt=system,
            tools=[],
            mcp_servers={"nanobot": server},
            allowed_tools=allowed_tools,
            permission_mode="dontAsk",
            max_turns=spec.max_iterations,
            include_partial_messages=hook.wants_streaming(),
        )
        stream_open = False
        last_assistant_text = ""

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, StreamEvent):
                        event = message.event or {}
                        delta = event.get("delta") or {}
                        if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                            text = str(delta.get("text") or "")
                            if text:
                                stream_open = True
                                await hook.on_stream(
                                    AgentHookContext(iteration=iteration, messages=messages), text
                                )
                    elif isinstance(message, AssistantMessage):
                        text = "".join(
                            block.text for block in message.content if isinstance(block, TextBlock)
                        )
                        if text:
                            last_assistant_text = text
                    elif isinstance(message, ResultMessage):
                        raw = message.usage or {}
                        usage = {
                            "prompt_tokens": int(raw.get("input_tokens", 0) or 0),
                            "completion_tokens": int(raw.get("output_tokens", 0) or 0),
                        }
                        final_content = message.result or last_assistant_text or None
                        if message.is_error:
                            result_error = "; ".join(message.errors or []) or message.subtype
                            result_stop = (
                                "max_iterations"
                                if "max_turn" in message.subtype
                                else "error"
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Claude Agent SDK run failed")
            result_error = f"Error calling Claude Agent SDK: {exc}"
            result_stop = "error"

        context = AgentHookContext(
            iteration=iteration,
            messages=messages,
            usage=usage,
            final_content=final_content,
            stop_reason=result_stop,
            error=result_error,
        )
        if stream_open:
            await hook.on_stream_end(context, resuming=False)
        if result_error and not final_content:
            final_content = spec.error_message or result_error
        final_content = hook.finalize_content(context, final_content)
        if final_content:
            messages.append(build_assistant_message(final_content))
        await hook.after_iteration(context)
        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=result_stop,
            error=result_error,
            tool_events=tool_events,
        )
