"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator

import httpx

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.codex_credentials import CodexCredentialManager

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
_CONTINUATION_CACHE_LIMIT = 32
_CONTINUATION_CACHE_TTL_SECONDS = 300.0

ContinuationKey = tuple[str, tuple[str, ...]]


class _CodexHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.6-sol",
        credential_manager: CodexCredentialManager | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.credentials = credential_manager or CodexCredentialManager()
        self._continuations: OrderedDict[
            ContinuationKey, tuple[float, list[dict[str, Any]]]
        ] = OrderedDict()

    def _prune_continuations(self) -> None:
        cutoff = time.monotonic() - _CONTINUATION_CACHE_TTL_SECONDS
        expired = [key for key, (created_at, _) in self._continuations.items() if created_at < cutoff]
        for key in expired:
            self._continuations.pop(key, None)
        while len(self._continuations) > _CONTINUATION_CACHE_LIMIT:
            self._continuations.popitem(last=False)

    def _find_continuation(
        self, messages: list[dict[str, Any]]
    ) -> tuple[ContinuationKey | None, int | None, list[dict[str, Any]] | None]:
        self._prune_continuations()
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            call_ids = tuple(
                call.get("id")
                for call in message["tool_calls"]
                if isinstance(call, dict) and isinstance(call.get("id"), str)
            )
            if len(call_ids) != len(message["tool_calls"]):
                return None, None, None
            key = (_prompt_cache_key(messages[:index]), call_ids)
            cached = self._continuations.get(key)
            if cached is not None:
                self._continuations.move_to_end(key)
                return key, index, [dict(item) for item in cached[1]]
            return None, None, None
        return None, None, None

    def _cache_continuation(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
        output_items: list[dict[str, Any]],
    ) -> None:
        if not tool_calls or not any(item.get("type") == "reasoning" for item in output_items):
            return
        key = (_prompt_cache_key(messages), tuple(call.id for call in tool_calls))
        self._continuations[key] = (time.monotonic(), [dict(item) for item in output_items])
        self._continuations.move_to_end(key)
        self._prune_continuations()

    async def _request_with_auth(
        self,
        body: dict[str, Any],
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[ToolCallRequest], str, list[dict[str, Any]]]:
        credentials = await self.credentials.get_credentials()
        for attempt in range(2):
            try:
                return await _request_codex(
                    DEFAULT_CODEX_URL,
                    _build_headers(credentials.account_id, credentials.access_token),
                    body,
                    on_content_delta=on_content_delta,
                )
            except _CodexHTTPError as exc:
                if exc.status_code != 401 or attempt == 1:
                    raise
                credentials = await self.credentials.get_credentials(
                    force_refresh=True,
                    rejected_access_token=credentials.access_token,
                )
        raise AssertionError("unreachable")

    async def _call_codex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Shared request logic for both chat() and chat_stream()."""
        model = model or self.default_model
        continuation_key, assistant_index, continuation_items = self._find_continuation(messages)
        system_prompt, input_items = _convert_messages(
            messages,
            continuation_assistant_index=assistant_index,
            continuation_items=continuation_items,
        )
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
        ):
            tool_choice = {"type": "function", "name": tool_choice["function"].get("name")}

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": True,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = _convert_tools(tools)

        try:
            content, tool_calls, finish_reason, output_items = await self._request_with_auth(
                body, on_content_delta=on_content_delta
            )
            if continuation_key is not None:
                self._continuations.pop(continuation_key, None)
            self._cache_continuation(messages, tool_calls, output_items)
            return LLMResponse(content=content, tool_calls=tool_calls, finish_reason=finish_reason)
        except Exception as e:
            return LLMResponse(content=f"Error calling Codex: {e}", finish_reason="error")

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(messages, tools, model, reasoning_effort, tool_choice)

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(messages, tools, model, reasoning_effort, tool_choice, on_content_delta)

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, list[dict[str, Any]]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise _CodexHTTPError(
                    response.status_code,
                    _friendly_error(response.status_code, text.decode("utf-8", "ignore")),
                )
            return await _consume_sse(response, on_content_delta)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _convert_messages(
    messages: list[dict[str, Any]],
    *,
    continuation_assistant_index: int | None = None,
    continuation_items: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                    "status": "completed", "id": f"msg_{idx}",
                })
            if idx == continuation_assistant_index and continuation_items is not None:
                input_items.extend(dict(item) for item in continuation_items)
                continue
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                input_items.append({
                    "type": "function_call",
                    "id": item_id or f"fc_{idx}",
                    "call_id": call_id or f"call_{idx}",
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append({"type": "function_call_output", "call_id": call_id, "output": output_text})

    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [entry[5:].strip() for entry in buffer if entry.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(
    response: httpx.Response,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, list[dict[str, Any]]]:
    content = ""
    completed_output: list[tuple[int, int, dict[str, Any]]] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"
    completion_order = 0

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            delta_text = event.get("delta") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") in {"reasoning", "function_call"}:
                output_index = event.get("output_index")
                order = output_index if isinstance(output_index, int) else completion_order
                completed_output.append((order, completion_order, dict(item)))
                completion_order += 1
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    completed_output.sort(key=lambda entry: (entry[0], entry[1]))
    output_items: list[dict[str, Any]] = []
    tool_calls: list[ToolCallRequest] = []
    for _, _, item in completed_output:
        item_type = item.get("type")
        if item_type == "reasoning":
            if item.get("encrypted_content"):
                output_items.append(item)
            continue
        call_id = item.get("call_id")
        if not call_id:
            continue
        buf = tool_call_buffers.get(call_id) or {}
        args_raw = buf.get("arguments") or item.get("arguments") or "{}"
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {"raw": args_raw}
        tool_calls.append(
            ToolCallRequest(
                id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                name=buf.get("name") or item.get("name"),
                arguments=args,
            )
        )
        output_items.append(item)

    return content, tool_calls, finish_reason, output_items


_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _friendly_error(status_code: int, _raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    if status_code == 401:
        return "Codex authentication failed. Run: codex login"
    return f"HTTP {status_code}: Codex request failed"
