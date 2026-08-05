"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    estimate_text_tokens,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage. "
            "Update long-term memory INCREMENTALLY: only report newly learned facts and "
            "facts that became obsolete. Do NOT echo the whole memory back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "new_facts": {
                        "type": "string",
                        "description": "Durable new facts to ADD to long-term memory, as markdown. "
                        "Include only facts NOT already present in the current memory shown to you. "
                        "Use an empty string when nothing new is worth remembering.",
                    },
                    "obsolete_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Snippets of the CURRENT memory that are now outdated or wrong "
                        "and should be removed. Each item must be an exact substring of the current "
                        "memory. Use an empty list when nothing should be removed.",
                    },
                },
                "required": ["history_entry"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def _apply_memory_update(current: str, args: dict[str, Any]) -> str:
    """Compute the new long-term memory from an incremental save_memory payload.

    Supports two shapes:
    - Legacy full rewrite: ``memory_update`` carries the entire memory.
    - Incremental (default): ``new_facts`` is appended and ``obsolete_facts``
      snippets are removed. This keeps tool-call output small so it never gets
      truncated by max_tokens, regardless of how large the memory grows.
    """
    if args.get("memory_update") is not None:
        return _ensure_text(args["memory_update"])

    obsolete = args.get("obsolete_facts") or []
    new_facts = _ensure_text(args.get("new_facts") or "").strip()
    if not obsolete and not new_facts:
        return current  # nothing to change

    updated = current
    if isinstance(obsolete, list):
        for snippet in obsolete:
            text = _ensure_text(snippet)
            if text and text in updated:
                updated = updated.replace(text, "")

    if new_facts:
        updated = f"{updated.rstrip()}\n\n{new_facts}\n" if updated.strip() else f"{new_facts}\n"

    # Collapse blank-line runs left behind by removals / appends.
    return re.sub(r"\n{3,}", "\n\n", updated).lstrip("\n")

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # --- long-term memory compaction -------------------------------------------------
    #
    # MEMORY.md is re-read into the system prompt on *every* turn while save_memory
    # only ever appends, so without a ceiling it grows until it owns the whole context
    # window (it reached 64k tokens in production before this existed).  Compaction
    # spills the fattest entries into a dated archive and leaves their headline behind
    # as an index -- nothing is deleted, the agent can grep the archive when a headline
    # turns out to matter.

    _COMPACT_MIN_ENTRY_TOKENS = 120  # smaller entries cost less than the index line saves
    _COMPACT_TARGET_RATIO = 0.8  # compact below the cap so it does not retrigger at once
    _ARCHIVE_POINTER = (
        "> 部分条目只剩标题，正文已归档到 `memory/MEMORY_archive_*.md`（按标题 grep 取回全文），\n"
        "> 更细的过程记录在 `memory/HISTORY.md`。需要细节时去读，别凭标题猜。"
    )

    @staticmethod
    def _split_entries(text: str) -> list[str]:
        """Split memory into top-level bullet entries; non-bullet lines stay as-is."""
        blocks: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            starts_entry = line.startswith("- ") or re.match(r"^#{1,6} ", line)
            if starts_entry and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    @staticmethod
    def _entry_headline(entry: str) -> str:
        """Shrink one bullet entry to an index line that still identifies it."""
        first = entry.split("\n", 1)[0]
        if bold := re.match(r"(- \*\*.*?\*\*)", first):
            return bold.group(1)
        # No bold title: cut at the first clause separator so the line stays readable.
        for sep in ("：", ": ", "。"):
            head, found, _ = first.partition(sep)
            if found and len(head) > 4:
                return head + found.rstrip(" ")
        return first[:200]

    def compact_long_term(self, max_tokens: int) -> bool:
        """Spill the fattest MEMORY.md entries into a dated archive. Returns True if changed."""
        if max_tokens <= 0:
            return False
        text = self.read_long_term()
        total = estimate_text_tokens(text)
        if total <= max_tokens:
            return False

        target = int(max_tokens * self._COMPACT_TARGET_RATIO)
        entries = self._split_entries(text)
        sizes = [estimate_text_tokens(entry) for entry in entries]

        # Largest-first: fewest entries lose their body for a given saving.
        order = sorted(
            (i for i, size in enumerate(sizes) if size >= self._COMPACT_MIN_ENTRY_TOKENS
             and entries[i].startswith("- ")),
            key=lambda i: sizes[i],
            reverse=True,
        )
        spilled: dict[int, str] = {}
        running = total
        for idx in order:
            if running <= target:
                break
            headline = self._entry_headline(entries[idx])
            saving = sizes[idx] - estimate_text_tokens(headline)
            if saving <= 0:
                continue
            spilled[idx] = headline
            running -= saving

        if not spilled:
            logger.warning(
                "Long-term memory is {} tokens (cap {}) but has no entry big enough to "
                "compact -- it needs manual cleanup", total, max_tokens,
            )
            return False

        archive = self.memory_dir / f"MEMORY_archive_{datetime.now().strftime('%Y-%m-%d')}.md"
        with open(archive, "a", encoding="utf-8") as fh:
            if archive.stat().st_size == 0:
                fh.write("# MEMORY 归档\n\nMEMORY.md 中被压成标题的条目全文。\n\n")
            for idx in sorted(spilled):
                fh.write(entries[idx].rstrip() + "\n\n")

        compacted = [spilled.get(i, entry) for i, entry in enumerate(entries)]
        updated = "\n".join(compacted)
        if self._ARCHIVE_POINTER not in updated:
            updated = f"{self._ARCHIVE_POINTER}\n\n{updated.lstrip()}"
        self.write_long_term(re.sub(r"\n{3,}", "\n\n", updated))

        logger.info(
            "Long-term memory compacted: {} -> {} tokens (cap {}), {} entries archived to {}",
            total, estimate_text_tokens(updated), max_tokens, len(spilled), archive.name,
        )
        return True

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        max_tokens: int | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

Update long-term memory INCREMENTALLY: put genuinely new durable facts in `new_facts`,
list any now-outdated snippets of the current memory in `obsolete_facts`, and leave both
empty when nothing changed. Never paste the whole memory back.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with an incremental update (new_facts / obsolete_facts) of the conversation."},
            {"role": "user", "content": prompt},
        ]

        extra = {"max_tokens": max_tokens} if max_tokens is not None else {}

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
                **extra,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                    **extra,
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or args["history_entry"] is None:
                logger.warning("Memory consolidation: save_memory payload missing history_entry")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(args["history_entry"]).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            self.append_history(entry)
            update = _apply_memory_update(current_memory, args)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _CONSOLIDATION_TRIGGER_RATIO = 0.75

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    # Floor for the consolidation tool-call output. Incremental updates keep the
    # payload small, but the history_entry plus new_facts on a busy chunk can still
    # be a few thousand tokens, so guarantee comfortable headroom regardless of the
    # provider's (often 4096) default completion budget.
    _MIN_CONSOLIDATION_MAX_TOKENS = 8192

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        memory_max_tokens: int = 24_000,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.memory_max_tokens = memory_max_tokens
        try:
            _base_max_tokens = int(max_completion_tokens)
        except (TypeError, ValueError):
            _base_max_tokens = 4096
        self.consolidation_max_tokens = max(
            _base_max_tokens, self._MIN_CONSOLIDATION_MAX_TOKENS
        )
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        ok = await self.store.consolidate(
            messages, self.provider, self.model, max_tokens=self.consolidation_max_tokens
        )
        # Consolidation is the only thing that grows MEMORY.md, so this is the one
        # place the cap can be enforced without checking on every turn.
        try:
            self.store.compact_long_term(self.memory_max_tokens)
        except Exception:
            logger.exception("Long-term memory compaction failed")
        return ok

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            trigger = min(
                budget,
                int(self.context_window_tokens * self._CONSOLIDATION_TRIGGER_RATIO),
            )
            if trigger <= 0:
                return
            target = max(1, budget // 2)
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < trigger:
                logger.debug(
                    "Token consolidation idle {}: {}/{} (trigger={}) via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    trigger,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
