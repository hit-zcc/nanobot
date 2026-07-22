"""LLM API call logger — writes structured JSONL for monitoring."""

import json
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Any


_LOG_DIR = Path.home() / ".nanobot" / "workspace" / "llm_logs"
_LOG_FILE = _LOG_DIR / "calls.jsonl"
_MAX_SIZE = 50 * 1024 * 1024  # 50 MB
_MSG_PREVIEW_LEN = 200
_SYSTEM_FULL_LIMIT = 2000

_lock = threading.Lock()


def _ensure_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed() -> None:
    """Rotate log file if it exceeds _MAX_SIZE."""
    try:
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > _MAX_SIZE:
            ts = time.strftime("%Y%m%d_%H%M%S")
            _LOG_FILE.rename(_LOG_DIR / f"calls_{ts}.jsonl")
    except OSError:
        pass


def _extract_content_text(content: Any, *, full: bool = False) -> str:
    """Convert a message content field to a string.

    When *full* is True all text is kept; binary blobs are replaced with
    placeholders.  When *full* is False only the first _MSG_PREVIEW_LEN chars
    are returned.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "?")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype in ("image_url", "image"):
                    parts.append(f"[{btype}]")
                elif btype == "tool_use":
                    input_repr = json.dumps(block.get("input", {}), ensure_ascii=False)
                    parts.append(f"[tool_use: {block.get('name', '?')}] {input_repr}")
                elif btype == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = "\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in tc
                        )
                    parts.append(f"[tool_result] {tc}")
                else:
                    parts.append(f"[{btype}]")
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    elif content is None:
        text = ""
    else:
        text = str(content)

    if not full:
        return text[:_MSG_PREVIEW_LEN]
    return text


def _summarize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Extract role + first N chars of each message (for list view)."""
    if not messages:
        return []
    return [
        {"role": msg.get("role", "?"), "preview": _extract_content_text(msg.get("content", ""), full=False)}
        for msg in messages
    ]


def _sanitize_content_block(block: Any) -> Any:
    """Sanitize a single content block, replacing binary data with placeholders."""
    if not isinstance(block, dict):
        return block
    btype = block.get("type", "")
    if btype in ("image_url", "image"):
        return {"type": btype, "source": "[binary image data]"}
    return block


def _full_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Extract full message content, preserving original structure.

    Keeps tool_use blocks with name/id/input, tool_result blocks, etc.
    Binary blobs (images) are replaced with placeholders.
    System messages are truncated at _SYSTEM_FULL_LIMIT characters.
    """
    if not messages:
        return []
    result = []
    for msg in messages:
        entry: dict[str, Any] = {}
        # Copy all fields except content (handle separately)
        for k, v in msg.items():
            if k == "content":
                continue
            entry[k] = v

        content = msg.get("content", "")
        if isinstance(content, list):
            # Preserve structured content blocks (tool_use, tool_result, text, etc.)
            entry["content"] = [_sanitize_content_block(b) for b in content]
        elif content is None:
            entry["content"] = ""
        else:
            text = str(content)
            role = msg.get("role", "")
            if role == "system" and len(text) > _SYSTEM_FULL_LIMIT:
                omitted = len(text) - _SYSTEM_FULL_LIMIT
                text = text[:_SYSTEM_FULL_LIMIT] + f"\n... [truncated {omitted} chars]"
            entry["content"] = text

        result.append(entry)
    return result


def _summarize_response(response: Any) -> str:
    """Get a short preview of the response content."""
    content = getattr(response, "content", None) or ""
    return str(content)[:_MSG_PREVIEW_LEN]


def _full_response(response: Any) -> str:
    """Get the full response content."""
    content = getattr(response, "content", None)
    if content is None:
        return ""
    return str(content)


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    """Extract tool/function names from tool definitions.

    Supports both OpenAI format ``{"type": "function", "function": {"name": ...}}``
    and flat format ``{"name": ...}``.
    """
    if not tools:
        return []
    names = []
    for t in tools:
        if not isinstance(t, dict):
            names.append("?")
            continue
        func = t.get("function")
        if isinstance(func, dict):
            name = func.get("name")
        else:
            name = None
        if not name:
            name = t.get("name") or "?"
        names.append(name)
    return names


def log_call(
    *,
    model: str | None,
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
    temperature: Any,
    max_tokens: Any,
    reasoning_effort: Any,
    tool_choice: Any,
    response: Any,
    duration_ms: float,
    stream: bool = False,
) -> None:
    """Write one JSONL record. Runs in a daemon thread to avoid blocking."""
    try:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": round(time.time(), 3),
            "model": model,
            "response_model": getattr(response, "response_model", None),
            "stream": stream,
            "messages_summary": _summarize_messages(messages),
            "messages_full": _full_messages(messages),
            "message_count": len(messages) if messages else 0,
            "tools": _tool_names(tools),
            "tool_choice": str(tool_choice) if tool_choice else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "response_preview": _summarize_response(response),
            "response_full": _full_response(response),
            "finish_reason": getattr(response, "finish_reason", None),
            "tool_calls": [getattr(tc, "name", str(tc)) for tc in (getattr(response, "tool_calls", None) or [])],
            "usage": getattr(response, "usage", {}),
            "duration_ms": round(duration_ms, 1),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception:
        print(f"[llm_logger] serialize error: {traceback.format_exc()}", file=sys.stderr)
        return

    def _write() -> None:
        try:
            with _lock:
                _ensure_dir()
                _rotate_if_needed()
                with open(_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            print(f"[llm_logger] write error: {traceback.format_exc()}", file=sys.stderr)

    t = threading.Thread(target=_write, daemon=True)
    t.start()
