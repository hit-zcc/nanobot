"""A standing index of the workspace's own notes, so the agent can decide what to reread.

Why an index rather than a trigger
----------------------------------
The failure this addresses is not missing knowledge — the notes exist and are
detailed. It is that nothing prompts a lookup at the moment one is needed, and
the moments that hurt most are exactly those where the agent feels certain and
never thinks to ask.

An earlier attempt matched regexes against each command and pushed a reminder
when one hit. Replaying a real evening through it settled the question: of six
fresh mistakes it caught three, while interrupting 36% of all commands. Worse,
half the rules were themselves wrong — patterns assuming an ordering the real
command did not have, a slash inside a character class silently truncating the
expression. Every fix made the regex harder to read, and each one was still a
guess about intent made from surface text.

So this module does not guess. It states what exists and when each note
applies, and leaves the judgement where the context actually is: with the
model. The same bet ``SkillsLoader`` makes — a short description stays
resident, the body is read on demand — which keeps the standing cost
proportional to the number of notes rather than their length.

Being always present is the point. A trigger only fires on an action already
taken; an index is readable *before* acting, which is the only way to catch
the most expensive kind of mistake — a false premise written into a plan while
no tool has been called at all.

Opting in
---------
Any ``.md`` under the scanned roots joins the index by declaring one line::

    ---
    when: 要查任何 SLS 日志：假 0 与假命中、-n 是拉取上限不是计数
    ---

Files without that field are skipped and cost nothing, so the roots can be
broad without turning the index into a directory listing. The field says *when
to open the file*, not what it contains — a title answers "what is this", while
the decision being made needs "does this apply to what I am about to do".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# One line per note. Long entries would defeat the purpose: an index that costs
# as much as the notes may as well be the notes.
_MAX_WHEN_CHARS = 300
# Frontmatter sits at the top; reading the whole of a 1200-line map to find out
# it has none is wasted work at every startup.
_HEAD_CHARS = 4000
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_WHEN_FIELD = re.compile(r"^when:\s*(.+?)\s*$", re.M | re.S)

# Where notes may live. Broad on purpose — membership is decided by the file
# itself, so adding a directory here cannot flood the index.
DEFAULT_ROOTS = ("knowledge", "skills", "memory")


@dataclass
class IndexScan:
    """The index plus what it failed to read.

    Discovery that silently returns fewer results than it should is worse than
    discovery that fails loudly: the caller cannot tell "nothing matched" from
    "the lookup broke", and reads a partial answer as a complete one.
    """

    entries: list[dict[str, str]] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unreadable


class KnowledgeIndex:
    """Collects ``when:`` frontmatter across the workspace into a compact index."""

    def __init__(self, workspace: Path, roots: tuple[str, ...] = DEFAULT_ROOTS) -> None:
        self.workspace = workspace if isinstance(workspace, Path) else None
        self.roots = roots

    def scan(self) -> "IndexScan":
        """Collect the index, keeping the files that could not be read.

        A note that fails to load and one that opted out are indistinguishable
        in the result — both simply aren't there — which is the shape of every
        "it returned nothing, so nothing exists" mistake. Keeping the failures
        lets the caller say *index incomplete* instead of quietly shipping a
        short list.
        """
        if self.workspace is None:
            return IndexScan(entries=[], unreadable=[])
        seen: set[Path] = set()
        out: list[dict[str, str]] = []
        unreadable: list[tuple[str, str]] = []
        for root_name in self.roots:
            root = self.workspace / root_name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.md")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    name = str(path.relative_to(self.workspace))
                except ValueError:
                    name = path.name
                try:
                    when = self._read_when(path)
                except OSError as e:
                    # Unreadable, not opted out. Surfaced rather than dropped.
                    unreadable.append((name, str(e)))
                    logger.warning("knowledge-index: cannot read {}: {}", name, e)
                    continue
                if not when:
                    continue
                out.append({"name": name, "when": when, "location": str(path)})
        return IndexScan(entries=out, unreadable=unreadable)

    def entries(self) -> list[dict[str, str]]:
        """Return ``{name, when, location}`` for every note that declares one."""
        return self.scan().entries

    @staticmethod
    def _read_when(path: Path) -> str:
        """Extract the ``when:`` field. Raises ``OSError`` if the file can't be read."""
        # Decoding errors are replaced rather than raised: a stray byte in an
        # otherwise fine note should cost that note's frontmatter at worst, not
        # mark the file unreadable.
        head = path.read_text(encoding="utf-8", errors="replace")[:_HEAD_CHARS]
        block = _FRONTMATTER.match(head)
        if not block:
            return ""
        field = _WHEN_FIELD.search(block.group(1))
        if not field:
            return ""
        when = " ".join(field.group(1).split())
        if len(when) > _MAX_WHEN_CHARS:
            when = when[:_MAX_WHEN_CHARS] + "…"
        return when

    def build_summary(self) -> str:
        """Render the index for the system prompt, or "" when there is nothing."""
        scan = self.scan()
        if not scan.entries and scan.complete:
            return ""

        def esc(text: str) -> str:
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<notes>"]
        for entry in scan.entries:
            lines.append("  <note>")
            lines.append(f"    <when>{esc(entry['when'])}</when>")
            lines.append(f"    <location>{entry['location']}</location>")
            lines.append("  </note>")
        if scan.unreadable:
            # Say so in the prompt itself. The agent is the one about to decide
            # "I have no note about this", and it should know that conclusion
            # rests on an index that is missing entries.
            listed = ", ".join(esc(name) for name, _ in scan.unreadable[:5])
            more = f" (+{len(scan.unreadable) - 5} more)" if len(scan.unreadable) > 5 else ""
            lines.append("  <incomplete>")
            lines.append(
                f"    This index is INCOMPLETE: {len(scan.unreadable)} note(s) could not be "
                f"read, so a note may exist that is not listed here: {listed}{more}"
            )
            lines.append("  </incomplete>")
        lines.append("</notes>")
        logger.debug(
            "knowledge-index: {} entr(ies), {} unreadable",
            len(scan.entries), len(scan.unreadable),
        )
        return "\n".join(lines)
