"""Boundaries of the after-the-fact reminder engine.

Each test here corresponds to a way the engine has actually been wrong:
a rule that went quiet everywhere after firing once, a heredoc delimiter the
guard had never heard of, a YAML rule whose author forgot the guard entirely.
"""

from pathlib import Path

import pytest

from nanobot.agent.tool_triggers import (
    _MAX_PER_ITERATION,
    _MAX_SESSION_BUCKETS,
    TriggerEngine,
    TriggerRule,
    build_reminder_message,
)


def _rules_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tool-triggers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _engine(tmp_path: Path, body: str) -> TriggerEngine:
    return TriggerEngine.load(_rules_file(tmp_path, body))


ONE_RULE = """
- id: sls-zero
  when:
    tool: exec
    arg_matches: "query_sls\\\\.py"
  say: 0 条不等于不存在
"""


# ---- once_per: session is per *session*, not per process -----------------


def test_same_session_fires_once(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ONE_RULE)

    first = engine.check("exec", {"command": "python query_sls.py"}, "cnt: 0", session_key="feishu:a")
    second = engine.check("exec", {"command": "python query_sls.py"}, "cnt: 0", session_key="feishu:a")

    assert first == ["0 条不等于不存在"]
    assert second == []


def test_other_session_is_still_armed(tmp_path: Path) -> None:
    """The bug: one firing in a private chat silenced every group, cron run
    and heartbeat tick for the life of the process."""
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}

    assert engine.check("exec", args, "", session_key="feishu:a") == ["0 条不等于不存在"]
    assert engine.check("exec", args, "", session_key="cron:daily") == ["0 条不等于不存在"]
    assert engine.check("exec", args, "", session_key="feishu:group-b") == ["0 条不等于不存在"]
    # ...and each of them is now individually spent.
    assert engine.check("exec", args, "", session_key="cron:daily") == []


def test_missing_session_key_shares_one_bucket(tmp_path: Path) -> None:
    """Callers that cannot name a session keep the old de-duplication rather
    than losing it (which would turn the reminder into wallpaper)."""
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}

    assert engine.check("exec", args, "") == ["0 条不等于不存在"]
    assert engine.check("exec", args, "") == []


def test_reset_rearms_one_session_only(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}
    engine.check("exec", args, "", session_key="a")
    engine.check("exec", args, "", session_key="b")

    engine.reset("a")

    assert engine.check("exec", args, "", session_key="a") == ["0 条不等于不存在"]
    assert engine.check("exec", args, "", session_key="b") == []


def test_reset_without_key_rearms_everything(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}
    engine.check("exec", args, "", session_key="a")
    engine.check("exec", args, "", session_key="b")

    engine.reset()

    assert engine.check("exec", args, "", session_key="a")
    assert engine.check("exec", args, "", session_key="b")


def test_session_buckets_are_bounded(tmp_path: Path) -> None:
    """One process can outlive thousands of sessions; the state must not."""
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}

    for n in range(_MAX_SESSION_BUCKETS * 3):
        engine.check("exec", args, "", session_key=f"s{n}")

    assert len(engine._fired_by_session) <= _MAX_SESSION_BUCKETS


def test_recently_used_session_survives_eviction(tmp_path: Path) -> None:
    """Eviction is LRU, so the conversation currently in progress is the last
    thing to be dropped."""
    engine = _engine(tmp_path, ONE_RULE)
    args = {"command": "python query_sls.py"}
    engine.check("exec", args, "", session_key="live")

    for n in range(_MAX_SESSION_BUCKETS - 1):
        engine.check("exec", args, "", session_key=f"s{n}")
        engine.check("exec", args, "", session_key="live")  # keeps it warm

    assert engine.check("exec", args, "", session_key="live") == []


# ---- the heredoc guard ---------------------------------------------------


HEREDOC_CASES = [
    "cat << 'EOF' >> notes.md\n查 SLS 要用 query_sls.py\nEOF",
    "cat <<'PY' > s.py\n# query_sls.py 的坑\nPY",
    "python3 <<'ZZ'\nprint('query_sls.py')\nZZ",
    'bash <<"ZZ"\necho query_sls.py\nZZ',
    "cat <<-EOF > note\nquery_sls.py\nEOF",
    "cat > note.md <<EOF\nquery_sls.py\nEOF",
    "tee notes.md <<'DOC'\nquery_sls.py\nDOC",
]


@pytest.mark.parametrize("command", HEREDOC_CASES)
def test_inline_rule_does_not_fire_on_heredocs(tmp_path: Path, command: str) -> None:
    """Writing a note *about* a command must not count as running it. The
    original guard knew only `*EOF`, so `<<'PY'` and `<<'ZZ'` went through."""
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "sls.md").write_text(
        "- SLS: <!-- t: exec /query_sls\\.py/ 0 条不等于不存在 -->\n",
        encoding="utf-8",
    )
    engine = TriggerEngine()
    engine.absorb_knowledge_dir(kb)
    assert engine.rules, "inline trigger should have been picked up"

    assert engine.check("exec", {"command": command}, "", session_key="s") == []


def test_inline_rule_still_fires_on_a_real_command(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "sls.md").write_text(
        "- SLS: <!-- t: exec /query_sls\\.py/ 0 条不等于不存在 -->\n",
        encoding="utf-8",
    )
    engine = TriggerEngine()
    engine.absorb_knowledge_dir(kb)

    out = engine.check("exec", {"command": "python3 query_sls.py -n 10"}, "", session_key="s")

    assert len(out) == 1
    assert "sls.md:1" in out[0]  # the pointer is appended automatically


def test_shift_operator_is_not_mistaken_for_a_heredoc(tmp_path: Path) -> None:
    """`<<` is also a shift; a guard that swallowed those would silently
    disarm rules, which is the worst failure mode available here."""
    engine = _engine(tmp_path, ONE_RULE)

    out = engine.check(
        "exec", {"command": "python3 -c 'print(1 << 3)' && python query_sls.py"}, "",
        session_key="s",
    )

    assert out == ["0 条不等于不存在"]


@pytest.mark.parametrize("command", HEREDOC_CASES)
def test_yaml_rules_are_guarded_too(tmp_path: Path, command: str) -> None:
    """YAML rules used to be compiled verbatim: 6 of the 8 shipped rules had
    no guard at all, so any note quoting the command re-fired them."""
    engine = _engine(tmp_path, ONE_RULE)

    assert engine.check("exec", {"command": command}, "", session_key="s") == []


def test_hand_written_guard_still_works_when_wrapped(tmp_path: Path) -> None:
    """Two live rules guard themselves with `\\A(?!…)`. Wrapping those in the
    engine guard nests one `\\A` inside another; verify it still matches."""
    engine = _engine(tmp_path, """
- id: hand-guarded
  when:
    tool: exec
    arg_matches: "\\\\A(?![\\\\s\\\\S]*(cat\\\\s*>|tee\\\\s|<< *.EOF))[\\\\s\\\\S]*(dcp sls)[\\\\s\\\\S]*(-n |--limit)"
  say: -n 是拉取上限不是计数
""")

    assert engine.check("exec", {"command": "dcp sls log query -n 20"}, "", session_key="s") == [
        "-n 是拉取上限不是计数"
    ]
    assert engine.check(
        "exec", {"command": "cat <<'EOF' >> n.md\ndcp sls log query -n 20\nEOF"}, "",
        session_key="s2",
    ) == []


def test_shipped_rules_file_still_compiles_and_fires() -> None:
    """The real rules on disk, not a fixture: guarding them in the engine must
    not have broken any of them."""
    path = Path.home() / ".nanobot" / "workspace" / "rules" / "tool-triggers.yaml"
    if not path.exists():  # pragma: no cover - workspace-dependent
        pytest.skip("workspace rules file not present")

    engine = TriggerEngine.load(path)

    # 断言"规则都装载成功"，不是"永远只有 N 条" —— 写死条数会让每次新增规则
    # 都变成一次假失败，久而久之只会被改数字糊过去，守卫本身就废了。
    # 真正要守的是：文件能解析、正则能编译、声明数 == 引擎加载数。
    declared = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("- id:")
    )
    assert declared > 0, "rules file should not be empty"
    assert len(engine.rules) == declared, (
        f"declared {declared} rules but engine loaded {len(engine.rules)} — "
        "通常是某个 pf 块忘了闭合标签，后面的规则被静默吃掉"
    )

    fired = engine.check(
        "exec", {"command": "dcp ddsv deploy create -c prod.x"}, "", session_key="s",
    )
    assert fired, "deploy-already-triggered should still fire"
    assert engine.check(
        "exec", {"command": "cat <<'PY' >> note.md\ndcp ddsv deploy create\nPY"}, "",
        session_key="s2",
    ) == []


# ---- matching semantics --------------------------------------------------


def test_result_pattern_is_what_makes_this_better_than_intent(tmp_path: Path) -> None:
    engine = _engine(tmp_path, """
- id: zero
  when:
    tool: exec
    arg_matches: "query_sls"
    result_matches: "cnt: 0"
  say: 0 条不等于不存在
""")
    args = {"command": "python query_sls.py"}

    assert engine.check("exec", args, "cnt: 1234", session_key="a") == []
    assert engine.check("exec", args, "cnt: 0", session_key="a") == ["0 条不等于不存在"]


def test_result_is_only_scanned_at_the_head(tmp_path: Path) -> None:
    engine = _engine(tmp_path, """
- id: deep
  when:
    tool: exec
    result_matches: "NEEDLE"
  say: found
""")

    buried = "x" * 5000 + "NEEDLE"
    assert engine.check("exec", {}, buried, session_key="a") == []
    assert engine.check("exec", {}, "NEEDLE", session_key="b") == ["found"]


def test_tool_name_must_match_exactly(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ONE_RULE)

    assert engine.check("read_file", {"path": "query_sls.py"}, "", session_key="a") == []


def test_rule_with_no_usable_condition_is_dropped(tmp_path: Path) -> None:
    """Otherwise it degrades into "fire on every tool call"."""
    engine = _engine(tmp_path, """
- id: empty
  when: {}
  say: 什么都响
""")

    assert engine.rules == []


def test_broken_regex_does_not_arm_a_catch_all(tmp_path: Path) -> None:
    engine = _engine(tmp_path, """
- id: bad
  when:
    arg_matches: "([unclosed"
  say: nope
""")

    assert engine.rules == []


def test_broken_rules_file_is_not_fatal(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "::: not yaml [[[")

    assert engine.rules == []
    assert engine.check("exec", {}, "", session_key="a") == []


def test_burst_of_matches_is_capped(tmp_path: Path) -> None:
    """A parallel batch of tools must not bury the work under reminders."""
    body = "".join(
        f"\n- id: r{n}\n  when:\n    tool: exec\n  say: reminder {n}\n"
        for n in range(_MAX_PER_ITERATION + 3)
    )
    engine = _engine(tmp_path, body)

    out = engine.check("exec", {"command": "ls"}, "", session_key="a")

    assert len(out) == _MAX_PER_ITERATION


def test_once_per_other_than_session_repeats(tmp_path: Path) -> None:
    """Documented quirk: any value other than "session" means "every time"."""
    engine = _engine(tmp_path, """
- id: loud
  when:
    tool: exec
  say: every time
  once_per: turn
""")

    assert engine.check("exec", {}, "", session_key="a") == ["every time"]
    assert engine.check("exec", {}, "", session_key="a") == ["every time"]


def test_raising_rule_is_skipped_not_fatal(tmp_path: Path) -> None:
    class Exploding(TriggerRule):
        def matches(self, *_args, **_kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    engine = TriggerEngine(rules=[
        Exploding(rule_id="bad", say="never", tool="exec"),
        TriggerRule(rule_id="ok", say="fine", tool="exec"),
    ])

    assert engine.check("exec", {}, "", session_key="a") == ["fine"]


def test_reminder_message_is_not_disguised_as_the_user() -> None:
    msg = build_reminder_message(["点 A", "点 B"])

    assert msg is not None
    assert msg["role"] == "user"
    assert "非用户发言" in msg["content"]
    assert "• 点 A" in msg["content"]
    assert build_reminder_message([]) is None
