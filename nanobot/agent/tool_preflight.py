"""Playbook lookups that fire *before* a tool runs.

Why this exists (and why it is not just tool_triggers backwards)
---------------------------------------------------------------
``tool_triggers`` fires after a tool settles. That timing is right for one
job — reading a *result* and catching a stale assumption about it — and it
does that job well: "your query came back empty" is a fact, not a guess.

But it is structurally incapable of a second job. A malformed command has
already failed by the time the result exists. During one release check the
after-the-fact rules caught nothing on six separate mistakes, every one of
which was documented beforehand: wrong subcommand, a flag that does not
exist, a field with no SQL index, pod name read off the wrong dict. The
knowledge was on disk. It arrived one step too late to matter.

So this module runs at the other end: the model has asked for a tool, the
arguments are known, nothing has happened yet. If the workspace holds a
playbook for this kind of call, it is spliced into the tool result the model
is about to read.

Two layers, and the split is the whole point
--------------------------------------------
Layer 1 — deterministic match. "Is this an SLS query?" is answered by looking
for ``query_sls`` in the command. Paying a model 1.8s to confirm something a
substring already settled is waste, and on a hot path that waste is the
difference between a feature that stays on and one that gets switched off.
Fifteen SLS calls in a single sweep would have added ~27s of pure latency.

Layer 2 — semantic fallback (optional, off unless a resolver is supplied).
Reserved for what substrings genuinely cannot do: reading an unfamiliar
command and deciding which of many notes actually bear on it.

The layers are separate because they fail differently. Layer 1 is exact and
blind; layer 2 is perceptive and occasionally wrong. Collapsing them would
mean paying layer 2's cost on layer 1's traffic — the reason the earlier
regex-only design was cheap but silent, and the reason an all-model design
would be smart but too slow to keep.

What gets injected
------------------
A *playbook*, not a warning: the commands with the traps already designed
out, so the right call is the one that is easiest to copy. Prose explaining
a trap still requires translating advice into a command, and that translation
step is where tired evenings lose.

Playbooks live in ``knowledge/**/*.md`` between markers, so the note and the
thing that arms it stay in one file::

    <!-- pf: exec /query_sls|dcp sls/ -->
    ```bash
    ...copy-ready commands...
    ```
    <!-- /pf -->

Same authoring principle as the inline ``<!-- t: -->`` triggers: a rule kept
in a separate file is a second thing to maintain, and it is exactly the thing
that gets skipped.
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

# A playbook is meant to be copied, so it may run longer than a one-line
# nudge — but not so long that it displaces the actual work in context.
_MAX_PLAYBOOK_CHARS = 1800
# Only one playbook per tool call. Two competing checklists is no checklist.
_MAX_PER_CALL = 1
# Same reasoning as tool_triggers: repeating a playbook the model has already
# been shown trains it to skip the section.
_ONCE_PER_SESSION = True
# "Session" means session, not process: the engine is built once per process,
# so the shown-set is kept per session key. Bounded and LRU-evicted — the
# worst case of evicting a live session is one playbook shown twice.
_MAX_SESSION_BUCKETS = 64
# Callers with no session key share one bucket (the old process-wide
# behaviour) rather than losing de-duplication altogether.
_DEFAULT_SESSION = "__default__"

# <!-- pf: <tool> /<arg regex>/ [title] -->  body  <!-- /pf -->
_PLAYBOOK_BLOCK = re.compile(
    r"<!--\s*pf:\s*(?P<tool>[a-z_]+)?\s*/(?P<pattern>(?:[^/\\]|\\.)*)/"
    r"\s*(?P<title>[^\n>]*?)\s*-->\n(?P<body>.*?)<!--\s*/pf\s*-->",
    re.I | re.S,
)


def _unescape_delims(src: str) -> str:
    return src.replace(r"\/", "/")


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by ``` fences, so examples can be told from rules.

    Fences inside a real playbook's body are balanced and nested within it,
    so tracking state line by line leaves the block's own marker outside any
    span — which is the distinction being drawn.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    open_at: int | None = None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if open_at is None:
                open_at = pos
            else:
                spans.append((open_at, pos + len(line)))
                open_at = None
        pos += len(line)
    if open_at is not None:  # unterminated fence: treat the tail as fenced
        spans.append((open_at, len(text)))
    return spans


@dataclass(slots=True)
class Playbook:
    """A copy-ready recipe, keyed to the kind of call it belongs to."""

    playbook_id: str
    body: str
    title: str = ""
    tool: str | None = None
    arg_matches: re.Pattern[str] | None = None

    def matches(self, tool_name: str, args_blob: str) -> bool:
        if self.tool and self.tool != tool_name:
            return False
        if self.arg_matches and not self.arg_matches.search(args_blob):
            return False
        return bool(self.tool or self.arg_matches)


@dataclass(slots=True)
class PreflightEngine:
    """Finds the playbook for a call that has not happened yet.

    ``resolver`` is the optional layer 2: given the tool name, the argument
    blob and the playbooks that layer 1 did *not* match, it returns the id of
    one worth showing, or None. Left unset, the engine is purely
    deterministic — and deliberately so, since layer 2 costs latency on a
    blocking path.
    """

    playbooks: list[Playbook] = field(default_factory=list)
    log_path: Path | None = None
    resolver: Callable[[str, str, list[Playbook]], str | None] | None = None
    # session key -> playbook ids already shown in that session.
    _shown_by_session: OrderedDict[str, set[str]] = field(default_factory=OrderedDict)

    # ---- loading -----------------------------------------------------

    @classmethod
    def load(
        cls,
        knowledge_dir: Path,
        log_path: Path | None = None,
        resolver: Callable[[str, str, list[Playbook]], str | None] | None = None,
    ) -> PreflightEngine:
        """Collect playbooks from the knowledge base. Never raises: a broken
        note must not take the agent down, it just means no playbook."""
        engine = cls(
            log_path=log_path if isinstance(log_path, Path) else None,
            resolver=resolver,
        )
        if not isinstance(knowledge_dir, Path) or not knowledge_dir.is_dir():
            return engine
        for md in sorted(knowledge_dir.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue
            if "<!-- pf:" not in text and "<!--pf:" not in text:
                continue
            rel = md.relative_to(knowledge_dir.parent) if knowledge_dir.parent in md.parents else Path(md.name)
            fenced = _fenced_spans(text)
            for match in _PLAYBOOK_BLOCK.finditer(text):
                # Documenting the syntax must not arm it. The note explaining
                # how to write a playbook contains, by construction, a
                # complete playbook — and on the first run it was loaded as a
                # real one, injecting "📖 标题 / 配方正文" into a live result.
                # Same shape as the heredoc trap in tool_triggers: writing
                # *about* a thing counted as doing it. Handled in the engine
                # rather than asked of the author, because a precaution you
                # have to remember is one you will eventually skip. A worked
                # example always sits in a code fence; a real playbook's
                # marker never does (its fences are inside its body).
                if any(lo <= match.start() < hi for lo, hi in fenced):
                    continue
                pb = cls._parse(match, str(rel))
                if pb is not None:
                    engine.playbooks.append(pb)
        if engine.playbooks:
            logger.info(
                "tool-preflight: loaded {} playbook(s) from {}",
                len(engine.playbooks), knowledge_dir,
            )
        return engine

    @staticmethod
    def _parse(match: re.Match[str], origin: str) -> Playbook | None:
        tool = (match.group("tool") or "").strip() or None
        pattern_src = _unescape_delims(match.group("pattern") or "")
        body = (match.group("body") or "").strip()
        title = (match.group("title") or "").strip()
        if not body:
            return None
        try:
            pattern = re.compile(pattern_src, re.I | re.S) if pattern_src else None
        except re.error as exc:
            logger.warning("tool-preflight: bad regex in {}: {}", origin, exc)
            return None
        if not (tool or pattern):
            return None
        if len(body) > _MAX_PLAYBOOK_CHARS:
            body = body[:_MAX_PLAYBOOK_CHARS] + "\n…(截断，完整见 " + origin + ")"
        return Playbook(
            playbook_id=origin,
            body=body,
            title=title,
            tool=tool,
            arg_matches=pattern,
        )

    # ---- matching ----------------------------------------------------

    def _bucket(self, session_key: str | None) -> set[str]:
        """Shown-state for one session, creating it (and evicting) as needed."""
        key = session_key or _DEFAULT_SESSION
        shown = self._shown_by_session.get(key)
        if shown is None:
            shown = set()
            self._shown_by_session[key] = shown
            while len(self._shown_by_session) > _MAX_SESSION_BUCKETS:
                self._shown_by_session.popitem(last=False)
        else:
            self._shown_by_session.move_to_end(key)
        return shown

    def check(
        self,
        tool_name: str,
        arguments: Any,
        session_key: str | None = None,
    ) -> list[Playbook]:
        """Return the playbook(s) to show before this call runs.

        *session_key* scopes the once-per-session rule; omitting it uses a
        shared default bucket.
        """
        if not self.playbooks:
            return []
        args_blob = self._blob(arguments)

        hits: list[Playbook] = []
        unmatched: list[Playbook] = []
        for pb in self.playbooks:
            if pb.matches(tool_name, args_blob):
                hits.append(pb)
            else:
                unmatched.append(pb)

        # `dcp ddsv deploy history -c prod.dedao-brain-dedao-bu` matches both
        # the generic dcp flag reference and the specific sweep procedure for
        # that service — and only one gets shown. The narrower pattern wins:
        # it was written for this exact call, while the general one would be
        # correct but beside the point. Pattern length stands in for
        # specificity, which is crude but observable, where "importance"
        # would be a judgement call made at authoring time and wrong later.
        if len(hits) > 1:
            hits.sort(
                key=lambda pb: len(pb.arg_matches.pattern) if pb.arg_matches else 0,
                reverse=True,
            )

        # Layer 2 only earns its latency where layer 1 came up empty.
        if not hits and self.resolver is not None and unmatched:
            try:
                picked = self.resolver(tool_name, args_blob, unmatched)
            except Exception as exc:
                logger.warning("tool-preflight: resolver failed: {}", exc)
                picked = None
            if picked:
                hits = [pb for pb in unmatched if pb.playbook_id == picked]

        shown = self._bucket(session_key)
        out: list[Playbook] = []
        for pb in hits:
            if _ONCE_PER_SESSION and pb.playbook_id in shown:
                continue
            out.append(pb)
            shown.add(pb.playbook_id)
            self._record(pb, tool_name)
            if len(out) >= _MAX_PER_CALL:
                break
        return out

    @staticmethod
    def _blob(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except Exception:
            return str(arguments)

    def _record(self, pb: Playbook, tool_name: str) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "playbook": pb.playbook_id,
                "tool": tool_name,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def reset_session(self, session_key: str | None = None) -> None:
        """Re-arm playbooks: one session with a key, all sessions without."""
        if session_key is None:
            self._shown_by_session.clear()
        else:
            self._shown_by_session.pop(session_key, None)


def build_playbook_note(playbooks: list[Playbook]) -> str | None:
    """Render playbooks as a block appended to the tool result.

    It rides along with the result rather than arriving as a separate message
    because a tool call the model is mid-way through is the one moment the
    advice is unambiguously about. A standalone message has to be connected
    back to the call it concerns; text attached to the result does not.
    """
    if not playbooks:
        return None
    parts: list[str] = []
    for pb in playbooks:
        head = f"📖 {pb.title}" if pb.title else "📖 这类调用有现成配方"
        parts.append(f"{head}\n{pb.body}")
    body = "\n\n".join(parts)
    return (
        "\n\n---\n[preflight · 来自你自己的档案，不是用户发言]\n"
        f"{body}\n"
        "若与现场证据冲突，以实测为准。"
    )
