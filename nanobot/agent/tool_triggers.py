"""Rule-driven reminders that fire *after* a tool runs.

Why this exists
---------------
An agent accumulates hard-won knowledge (pitfall notes, playbooks, past
findings) and then fails to consult it at the moment it matters — not because
the notes are missing, but because nothing prompts a lookup. Recall depends on
the agent first realising it needs to recall, and the cases that hurt most are
exactly the ones where it feels certain and never asks.

So instead of hoping the agent remembers to look, this module lets knowledge
*arrive on its own*: match on what a tool did **and what it returned**, then
splice a one-line reminder into the conversation.

Firing after execution (rather than before) is deliberate — the result is the
strongest signal available. "You ran an SLS query" is a guess about intent;
"your SLS query came back with zero rows" is a fact, and it is precisely when a
stale assumption is about to be made.

Design constraints, learned the hard way
----------------------------------------
A reminder system that fires too often becomes wallpaper — indistinguishable
from the 55 red markers that made an earlier memory file useless. Hence:

* ``once_per: session`` — a rule speaks once per conversation, then shuts up.
  Scoping is by session key (see ``check(session_key=…)``): a rule that has
  spoken in one chat is still armed in every other chat, cron run and
  heartbeat tick. It used to be per *process*, which meant one firing
  anywhere disarmed the rule everywhere until a restart.
* Reminders carry a pointer, not a copy. Two lines max: the conclusion and the
  file to read. Long text trains the reader to skip.
* Every fire is logged so the rules can be audited later. A rule that trips on
  30%+ of tool calls is noise and should be narrowed or dropped.

Rules live in ``workspace/rules/tool-triggers.yaml`` so they can be edited
without touching code or restarting.
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with the base deps
    yaml = None  # type: ignore[assignment]

# A reminder is a nudge, not a lecture. Anything longer gets skimmed.
_MAX_REMINDER_CHARS = 400
# Cap per iteration so a burst of parallel tools cannot bury the actual work.
_MAX_PER_ITERATION = 2
# Tool output can be huge; only the head is scanned for result patterns.
_RESULT_SCAN_CHARS = 4000
# ``once_per: session`` means *session*, not *process*. Firing state is kept
# per session key, which means the engine (built once in AgentLoop.__init__)
# would otherwise accumulate one bucket per conversation for the life of the
# process. Bounded, LRU-evicted: the worst case of evicting a live session is
# one reminder shown twice, which is far cheaper than unbounded growth.
_MAX_SESSION_BUCKETS = 64
# Callers that cannot supply a session key share this bucket, which restores
# the historical (process-wide) behaviour for them rather than disabling
# de-duplication entirely.
_DEFAULT_SESSION = "__default__"

# <!-- t: <tool> /<regex>/ <reminder> -->  — terse on purpose: a syntax that
# takes effort to recall is one that gets skipped at the moment of writing.
_INLINE_TRIGGER = re.compile(
    r"<!--\s*t:\s*(?P<tool>[a-z_]+)?\s*/(?P<pattern>(?:[^/\\]|\\.)*)/"
    r"(?:\s*=>\s*/(?P<result>(?:[^/\\]|\\.)*)/)?\s*(?P<say>.*?)\s*-->",
    re.I | re.S,
)

# Writing *about* a command must not count as running one. A heredoc that
# appends to a pitfall note contains, by construction, every literal the rule
# is looking for — so the note-taking itself trips the alarm. This was hit
# once in the standalone rules and again while measuring the inline ones, in
# both cases by someone who knew about it and forgot anyway. Hence the guard
# is applied by the engine rather than asked of the author: a precaution you
# have to remember is one you will eventually skip.
#
# The delimiter is *not* always ``EOF``. The first version of this guard
# matched ``<<\s*['\"]?\w*EOF`` and therefore let ``<<'PY'`` and ``<<'ZZ'``
# straight through — three false fires before anyone noticed that the guard
# had a vocabulary of one word. So: any heredoc operator (``<<`` or ``<<-``),
# optional space, then either a quoted delimiter or an unquoted one in the
# SHOUTING form that heredocs conventionally use.
#
# The unquoted branch is deliberately restricted to upper case: ``<<`` is also
# a shift operator, and ``cout << endl`` is not someone writing a note. A
# guard that swallowed those would silently disarm rules, which is the failure
# mode that is hardest to notice.
#
# Known and accepted limitation (do not try to fix it here): this separates
# "a command containing a heredoc" from "a plain command". It cannot separate
# "running a command" from "writing prose about that command" in general —
# that needs a different signal than the argument blob.
#
# ``\\?`` before each quote is not decoration: the blob being scanned is the
# JSON-serialised arguments, so a double-quoted delimiter arrives as
# ``<<\"ZZ\"``. Without it that exact case slips past the guard — caught by
# the parametrised test, not by reading the regex.
_HEREDOC_OP = r"<<-?\s*(?:\\?['\"][A-Za-z_]\w*\\?['\"]|[A-Z_][A-Z0-9_]*)"
_HEREDOC_GUARD = r"\A(?![\s\S]*(?:cat\s*>|tee\s|" + _HEREDOC_OP + r"))"


def _unescape_delims(src: str) -> str:
    """Turn the authored ``\\/`` back into a plain slash for the regex engine."""
    return src.replace(r"\/", "/")


def _guard(pattern_src: str) -> str:
    """Wrap an author's pattern so it cannot fire on text merely being written."""
    return _HEREDOC_GUARD + r"[\s\S]*(?:" + pattern_src + r")"


@dataclass(slots=True)
class TriggerRule:
    """One rule: when these conditions hold, surface this reminder."""

    rule_id: str
    say: str
    tool: str | None = None
    arg_matches: re.Pattern[str] | None = None
    result_matches: re.Pattern[str] | None = None
    once_per: str = "session"

    def matches(self, tool_name: str, args_blob: str, result_blob: str) -> bool:
        if self.tool and self.tool != tool_name:
            return False
        if self.arg_matches and not self.arg_matches.search(args_blob):
            return False
        if self.result_matches and not self.result_matches.search(result_blob):
            return False
        # A rule with no condition at all would fire on every single tool call.
        return bool(self.tool or self.arg_matches or self.result_matches)


@dataclass(slots=True)
class TriggerEngine:
    """Loads rules, matches them against settled tool calls, tracks firing."""

    rules: list[TriggerRule] = field(default_factory=list)
    log_path: Path | None = None
    # session key -> rule ids already fired in that session. The engine
    # outlives any single conversation (one instance per process), so keying
    # the state by session is what makes ``once_per: session`` mean what the
    # docstring and the YAML comments say it means.
    _fired_by_session: OrderedDict[str, set[str]] = field(default_factory=OrderedDict)

    # ---- loading -----------------------------------------------------

    @classmethod
    def load(cls, rules_path: Path, log_path: Path | None = None) -> TriggerEngine:
        """Build an engine from a YAML file. Never raises: a broken rules file
        must not take the agent down, it just means no reminders."""
        # Type-check the path rather than trusting it: this runs inside
        # AgentLoop.__init__, where a caller (or a test double) may hand over
        # something that merely looks path-like. Feeding such an object to a
        # YAML loader lets it be treated as a stream and read forever.
        engine = cls(log_path=log_path if isinstance(log_path, Path) else None)
        if yaml is None or not isinstance(rules_path, Path) or not rules_path.exists():
            return engine
        try:
            text = rules_path.read_text(encoding="utf-8")
            if not isinstance(text, str):
                return engine
            raw = yaml.safe_load(text) or []
        except Exception as exc:
            logger.warning("tool-triggers: cannot parse {}: {}", rules_path, exc)
            return engine
        if not isinstance(raw, list):
            logger.warning("tool-triggers: {} must hold a list of rules", rules_path)
            return engine

        for index, entry in enumerate(raw):
            rule = cls._parse_rule(entry, index)
            if rule is not None:
                engine.rules.append(rule)
        logger.info("tool-triggers: loaded {} rule(s) from {}", len(engine.rules), rules_path)
        return engine

    def absorb_knowledge_dir(self, knowledge_dir: Path) -> None:
        """Pick up inline triggers written next to the knowledge they guard.

        A rule kept in a separate file is a second thing to maintain, and the
        step that maintains it is the one that gets skipped: pitfalls get
        written up after an incident, the matching regex does not. Tonight's
        replay made the cost concrete — six fresh mistakes, zero caught, while
        the standalone rules still fired on 22% of calls. Busy, and pointed at
        the wrong things.

        So a trigger lives on the line below the note it belongs to::

            - **SQL 长字段没索引**: log_val_1 这类字段 regexp_extract 返回 null
              <!-- t: exec /SELECT.*log_val/ 长字段无 SQL 索引，null 是静默失败 -->

        Writing the note and arming the reminder become one action, which is
        the only version of this that survives contact with a tired evening.
        The source path is appended automatically: one less thing to type, and
        it cannot drift when a file is renamed.
        """
        if not isinstance(knowledge_dir, Path) or not knowledge_dir.is_dir():
            return
        found = 0
        for md in sorted(knowledge_dir.rglob("*.md")):
            try:
                lines = md.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            rel = md.relative_to(knowledge_dir.parent) if knowledge_dir.parent in md.parents else md.name
            for lineno, line in enumerate(lines, 1):
                match = _INLINE_TRIGGER.search(line)
                if not match:
                    continue
                rule = self._parse_inline(match, f"{rel}:{lineno}")
                if rule is not None:
                    self.rules.append(rule)
                    found += 1
        if found:
            logger.info("tool-triggers: +{} inline trigger(s) from {}", found, knowledge_dir)

    @staticmethod
    def _parse_inline(match: re.Match[str], origin: str) -> TriggerRule | None:
        tool = (match.group("tool") or "").strip() or None
        pattern_src = _unescape_delims(match.group("pattern") or "")
        say = (match.group("say") or "").strip()
        if not say:
            return None
        try:
            pattern = re.compile(_guard(pattern_src), re.I | re.S) if pattern_src else None
        except re.error as exc:
            logger.warning("tool-triggers: bad inline regex at {}: {}", origin, exc)
            return None
        if not (tool or pattern):
            return None
        # The pointer is the whole value of a reminder; never make the author
        # retype it, and never let it go stale.
        say = f"{say} → {origin}"
        if len(say) > _MAX_REMINDER_CHARS:
            say = say[:_MAX_REMINDER_CHARS] + " …"
        # An optional "=> /regex/" clause matches the tool's *output*. Firing on
        # "you searched" is a guess about intent and mostly noise; firing on
        # "your search came back empty" is the moment a wrong conclusion forms.
        result_src = _unescape_delims(match.groupdict().get("result") or "")
        try:
            result_pattern = re.compile(result_src, re.I | re.S) if result_src else None
        except re.error as exc:
            logger.warning("tool-triggers: bad inline result regex at {}: {}", origin, exc)
            result_pattern = None
        return TriggerRule(
            rule_id=f"kb:{origin}",
            say=say,
            tool=tool,
            arg_matches=pattern,
            result_matches=result_pattern,
            once_per="session",
        )

    @staticmethod
    def _parse_rule(entry: Any, index: int) -> TriggerRule | None:
        if not isinstance(entry, dict):
            return None
        when = entry.get("when") or {}
        say = str(entry.get("say") or "").strip()
        if not say or not isinstance(when, dict):
            return None
        if len(say) > _MAX_REMINDER_CHARS:
            say = say[:_MAX_REMINDER_CHARS] + " …"

        def compile_pattern(key: str, guard: bool = False) -> re.Pattern[str] | None:
            pattern = when.get(key)
            if not pattern:
                return None
            src = _guard(str(pattern)) if guard else str(pattern)
            try:
                return re.compile(src, re.I | re.S)
            except re.error as exc:
                logger.warning("tool-triggers: bad regex in rule #{} {}: {}", index, key, exc)
                return None

        rule = TriggerRule(
            rule_id=str(entry.get("id") or f"rule-{index}"),
            say=say,
            tool=str(when["tool"]) if when.get("tool") else None,
            # YAML rules used to be compiled verbatim, so the self-fire guard
            # was something each author had to remember to hand-write — and
            # 6 of the 8 rules on disk had not. Applying it in the engine
            # makes the two hand-written guards redundant but harmless: the
            # nested `\A(?!…)` still matches at position 0, verified in
            # tests/agent/test_tool_triggers.py.
            # Only arguments are guarded. A *result* may legitimately quote a
            # heredoc (it is output, not something being authored).
            arg_matches=compile_pattern("arg_matches", guard=True),
            result_matches=compile_pattern("result_matches"),
            once_per=str(entry.get("once_per") or "session"),
        )
        # Reject rules whose only condition failed to compile — they would
        # otherwise degrade into "fire on everything".
        if not (rule.tool or rule.arg_matches or rule.result_matches):
            logger.warning("tool-triggers: rule #{} has no usable condition, skipped", index)
            return None
        return rule

    # ---- matching ----------------------------------------------------

    def _bucket(self, session_key: str | None) -> set[str]:
        """Firing state for one session, creating it (and evicting) as needed."""
        key = session_key or _DEFAULT_SESSION
        fired = self._fired_by_session.get(key)
        if fired is None:
            fired = set()
            self._fired_by_session[key] = fired
            while len(self._fired_by_session) > _MAX_SESSION_BUCKETS:
                self._fired_by_session.popitem(last=False)
        else:
            self._fired_by_session.move_to_end(key)
        return fired

    def check(
        self,
        tool_name: str,
        arguments: Any,
        result: Any,
        session_key: str | None = None,
    ) -> list[str]:
        """Return reminders for one settled tool call (may be empty).

        *session_key* scopes ``once_per: session`` de-duplication. Omitting it
        puts the call in a shared default bucket.
        """
        if not self.rules:
            return []
        args_blob = self._blob(arguments)
        result_blob = self._blob(result)[:_RESULT_SCAN_CHARS]
        fired = self._bucket(session_key)

        out: list[str] = []
        for rule in self.rules:
            if rule.once_per == "session" and rule.rule_id in fired:
                continue
            try:
                hit = rule.matches(tool_name, args_blob, result_blob)
            except Exception as exc:  # a bad rule must not break the turn
                logger.warning("tool-triggers: rule {} raised {}", rule.rule_id, exc)
                continue
            if not hit:
                continue
            fired.add(rule.rule_id)
            out.append(rule.say)
            self._log(rule.rule_id, tool_name)
            if len(out) >= _MAX_PER_ITERATION:
                break
        return out

    @staticmethod
    def _blob(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _log(self, rule_id: str, tool_name: str) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rule": rule_id,
                "tool": tool_name,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # logging is best-effort, never fatal
            pass

    def reset(self, session_key: str | None = None) -> None:
        """Clear firing state so a conversation starts fresh.

        With a *session_key*, only that conversation is re-armed; without one,
        every session is (the behaviour ``/restart`` used to be needed for).
        """
        if session_key is None:
            self._fired_by_session.clear()
        else:
            self._fired_by_session.pop(session_key, None)


def build_reminder_message(reminders: list[str]) -> dict[str, Any] | None:
    """Wrap reminders as a user-role message the model will actually read.

    Marked as a system reminder rather than disguised as the user talking:
    mistaking automation for a real instruction from the user would be a far
    worse failure than the one this feature is trying to prevent.
    """
    if not reminders:
        return None
    body = "\n".join(f"• {line}" for line in reminders)
    return {
        "role": "user",
        "content": (
            "[自动提醒 · 非用户发言] 你刚才的操作命中了以前踩过的坑：\n"
            f"{body}\n"
            "如果与当前判断冲突，以实际查证为准。"
        ),
    }
