"""Boundaries of the before-the-call playbook engine.

The interesting cases are all about telling a *rule* from *prose about a
rule*: the note that documents the syntax contains, by construction, a
complete playbook, and on the first run it was loaded as a real one.
"""

import re
from pathlib import Path

from nanobot.agent.tool_preflight import (
    _MAX_PER_CALL,
    _MAX_PLAYBOOK_CHARS,
    _MAX_SESSION_BUCKETS,
    Playbook,
    PreflightEngine,
    _fenced_spans,
    build_playbook_note,
)

PLAYBOOK = """# SLS

<!-- pf: exec /query_sls|dcp sls/ SLS 查询配方 -->
```bash
python3 query_sls.py --logstore prod
```
<!-- /pf -->
"""


def _kb(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "knowledge"
    root.mkdir(exist_ok=True)
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


# ---- loading: a worked example is not a rule ----------------------------


def test_loads_a_real_playbook(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))

    assert len(engine.playbooks) == 1
    pb = engine.playbooks[0]
    assert pb.title == "SLS 查询配方"
    assert "query_sls.py --logstore" in pb.body


def test_documented_syntax_inside_a_fence_is_not_armed(tmp_path: Path) -> None:
    """The regression that shipped: the note explaining how to write a
    playbook was loaded as one and injected into a live tool result."""
    doc = "# How to write one\n\n```markdown\n" + PLAYBOOK + "```\n"
    engine = PreflightEngine.load(_kb(tmp_path, {"howto.md": doc}))

    assert engine.playbooks == []


def test_a_real_playbook_next_to_a_documented_one_still_loads(tmp_path: Path) -> None:
    doc = "```markdown\n" + PLAYBOOK + "```\n\n" + PLAYBOOK
    engine = PreflightEngine.load(_kb(tmp_path, {"mixed.md": doc}))

    assert len(engine.playbooks) == 1


def test_fenced_spans_cover_only_closed_fences() -> None:
    text = "a\n```\nb\n```\nc\n"
    spans = _fenced_spans(text)

    assert len(spans) == 1
    lo, hi = spans[0]
    assert text[lo:hi] == "```\nb\n```\n"


def test_unterminated_fence_swallows_the_tail() -> None:
    """An open fence with no closer is treated as fenced to the end: better to
    lose a playbook than to arm an example."""
    text = "a\n```\nb\nc\n"
    (lo, hi), = _fenced_spans(text)

    assert text[lo:hi] == "```\nb\nc\n"
    assert hi == len(text)


def test_files_without_the_marker_are_skipped(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"plain.md": "# nothing to see\n"}))

    assert engine.playbooks == []


def test_missing_directory_is_not_fatal(tmp_path: Path) -> None:
    engine = PreflightEngine.load(tmp_path / "nope")

    assert engine.playbooks == []
    assert engine.check("exec", {"command": "ls"}) == []


def test_body_is_truncated_with_a_pointer_home(tmp_path: Path) -> None:
    big = ("<!-- pf: exec /huge/ -->\n" + "x" * (_MAX_PLAYBOOK_CHARS + 500)
           + "\n<!-- /pf -->\n")
    engine = PreflightEngine.load(_kb(tmp_path, {"big.md": big}))

    body = engine.playbooks[0].body
    assert len(body) < _MAX_PLAYBOOK_CHARS + 200
    assert "完整见" in body and "big.md" in body


def test_playbook_without_a_body_is_dropped(tmp_path: Path) -> None:
    engine = PreflightEngine.load(
        _kb(tmp_path, {"empty.md": "<!-- pf: exec /x/ t -->\n\n<!-- /pf -->\n"})
    )

    assert engine.playbooks == []


# ---- matching -----------------------------------------------------------


def test_matches_on_the_argument_blob(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))

    assert engine.check("exec", {"command": "python3 query_sls.py"})
    assert engine.check("exec", {"command": "ls -la"}, session_key="x") == []


def test_tool_name_must_match(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))

    assert engine.check("read_file", {"path": "query_sls.py"}) == []


def test_narrower_pattern_wins(tmp_path: Path) -> None:
    """`dcp ddsv deploy history -c prod.x` matches both the generic flag
    reference and the procedure written for that exact service; only one is
    shown, and it should be the specific one."""
    engine = PreflightEngine(playbooks=[
        Playbook("generic.md", body="general dcp flags", arg_matches=re.compile("dcp")),
        Playbook(
            "specific.md", body="the sweep for dedao-bu",
            arg_matches=re.compile(r"dcp ddsv deploy history -c prod\.dedao-bu"),
        ),
    ])

    out = engine.check("exec", {"command": "dcp ddsv deploy history -c prod.dedao-bu"})

    assert [pb.playbook_id for pb in out] == ["specific.md"]


def test_only_one_playbook_per_call(tmp_path: Path) -> None:
    engine = PreflightEngine(playbooks=[
        Playbook(f"p{n}.md", body=f"body {n}", arg_matches=re.compile("dcp"))
        for n in range(_MAX_PER_CALL + 3)
    ])

    assert len(engine.check("exec", {"command": "dcp x"})) == _MAX_PER_CALL


def test_resolver_only_runs_when_layer_one_is_empty(tmp_path: Path) -> None:
    calls: list[str] = []

    def resolver(tool, blob, unmatched):
        calls.append(tool)
        return unmatched[0].playbook_id

    engine = PreflightEngine.load(
        _kb(tmp_path, {"sls.md": PLAYBOOK}), resolver=resolver,
    )

    assert engine.check("exec", {"command": "query_sls"}, session_key="a")
    assert calls == []  # layer 1 hit: no model call

    out = engine.check("exec", {"command": "something unfamiliar"}, session_key="b")
    assert calls == ["exec"]
    assert [pb.playbook_id for pb in out] == ["knowledge/sls.md"]


def test_resolver_failure_is_swallowed(tmp_path: Path) -> None:
    def resolver(*_args):
        raise RuntimeError("model down")

    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}), resolver=resolver)

    assert engine.check("exec", {"command": "unrelated"}) == []


# ---- once per session, not once per process -----------------------------


def test_same_session_shows_a_playbook_once(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))
    args = {"command": "query_sls.py"}

    assert engine.check("exec", args, session_key="feishu:a")
    assert engine.check("exec", args, session_key="feishu:a") == []


def test_other_sessions_still_get_the_playbook(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))
    args = {"command": "query_sls.py"}

    assert engine.check("exec", args, session_key="feishu:a")
    assert engine.check("exec", args, session_key="cron:nightly")
    assert engine.check("exec", args, session_key="feishu:group")


def test_reset_session_rearms_one_session(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))
    args = {"command": "query_sls.py"}
    engine.check("exec", args, session_key="a")
    engine.check("exec", args, session_key="b")

    engine.reset_session("a")

    assert engine.check("exec", args, session_key="a")
    assert engine.check("exec", args, session_key="b") == []


def test_session_buckets_are_bounded(tmp_path: Path) -> None:
    engine = PreflightEngine.load(_kb(tmp_path, {"sls.md": PLAYBOOK}))

    for n in range(_MAX_SESSION_BUCKETS * 3):
        engine.check("exec", {"command": "query_sls.py"}, session_key=f"s{n}")

    assert len(engine._shown_by_session) <= _MAX_SESSION_BUCKETS


# ---- rendering ----------------------------------------------------------


def test_note_is_labelled_as_the_agents_own_file() -> None:
    note = build_playbook_note([Playbook("p.md", body="cmd", title="配方")])

    assert note is not None
    assert "不是用户发言" in note
    assert "📖 配方" in note
    assert "cmd" in note
    assert build_playbook_note([]) is None


def test_untitled_playbook_still_gets_a_heading() -> None:
    note = build_playbook_note([Playbook("p.md", body="cmd")])

    assert note is not None and "📖 这类调用有现成配方" in note
