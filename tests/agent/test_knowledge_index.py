"""The index must stay cheap, opt-in, and must not invent entries."""

from pathlib import Path

from nanobot.agent.knowledge_index import KnowledgeIndex


def _note(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_lists_notes_that_declare_when(tmp_path: Path) -> None:
    _note(tmp_path, "knowledge/pitfalls/sls.md",
          "---\nwhen: 要查任何 SLS 日志：假 0 与假命中\n---\n# SLS\n" + "正文\n" * 50)

    entries = KnowledgeIndex(tmp_path).entries()

    assert len(entries) == 1
    assert entries[0]["name"] == "knowledge/pitfalls/sls.md"
    assert "假 0" in entries[0]["when"]
    assert entries[0]["location"].endswith("sls.md")


def test_collects_across_roots(tmp_path: Path) -> None:
    """Notes worth rereading are not all pitfalls, and not all in one folder."""
    _note(tmp_path, "knowledge/pitfalls/a.md", "---\nwhen: 场景 A\n---\n")
    _note(tmp_path, "knowledge/map.md", "---\nwhen: 场景 B\n---\n")
    _note(tmp_path, "skills/sls-log/references/playbook.md", "---\nwhen: 场景 C\n---\n")
    _note(tmp_path, "memory/LESSONS.md", "---\nwhen: 场景 D\n---\n")

    whens = {e["when"] for e in KnowledgeIndex(tmp_path).entries()}

    assert whens == {"场景 A", "场景 B", "场景 C", "场景 D"}


def test_membership_is_opt_in(tmp_path: Path) -> None:
    """Roots are broad, so files must join by declaring `when:`, never by living
    in the right folder. Otherwise the index degenerates into a directory listing
    and the entries that do carry a trigger get buried."""
    _note(tmp_path, "knowledge/no_front.md", "# 没有 frontmatter\n正文")
    _note(tmp_path, "knowledge/other_field.md", "---\ntitle: 有 frontmatter 但没 when\n---\n正文")
    _note(tmp_path, "skills/guizang/references/layouts.md", "# 一个无关的长参考文件\n" + "x\n" * 500)

    assert KnowledgeIndex(tmp_path).entries() == []
    assert KnowledgeIndex(tmp_path).build_summary() == ""


def test_summary_stays_far_smaller_than_the_notes(tmp_path: Path) -> None:
    """Progressive disclosure only pays off while the index is the cheap part."""
    for i in range(6):
        _note(tmp_path, f"knowledge/n{i}.md", f"---\nwhen: 场景 {i}\n---\n" + "正文行\n" * 400)

    summary = KnowledgeIndex(tmp_path).build_summary()
    body = sum(p.stat().st_size for p in (tmp_path / "knowledge").glob("*.md"))

    assert len(summary) < body * 0.10
    assert summary.count("<note>") == 6


def test_long_when_is_truncated(tmp_path: Path) -> None:
    _note(tmp_path, "knowledge/verbose.md", "---\nwhen: " + "长" * 900 + "\n---\n正文")

    entry = KnowledgeIndex(tmp_path).entries()[0]

    assert len(entry["when"]) < 400
    assert entry["when"].endswith("…")


def test_missing_directories_are_not_an_error(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path)

    assert index.entries() == []
    assert index.build_summary() == ""


def test_angle_brackets_are_escaped(tmp_path: Path) -> None:
    """The summary is spliced into an XML-ish prompt block; raw <> would break it."""
    _note(tmp_path, "knowledge/cmd.md", "---\nwhen: 用 <组件uid> 查 a & b\n---\n正文")

    summary = KnowledgeIndex(tmp_path).build_summary()

    assert "&lt;组件uid&gt;" in summary
    assert "a &amp; b" in summary


def test_frontmatter_only_read_from_the_head(tmp_path: Path) -> None:
    """A `when:` line buried in the body is prose, not a declaration."""
    _note(tmp_path, "knowledge/big.md", "# 标题\n" + "填充\n" * 3000 + "when: 这只是正文里提到\n")

    assert KnowledgeIndex(tmp_path).entries() == []


# ---- 读不出来的笔记要说出来，不能当成「没有」 ----------------------------
#
# 这几条针对的是一类具体的失败：查找悄悄少返回了几条，调用方分不清
# 「没有匹配」和「查找本身坏了」，于是把残缺的答案当成完整的答案用。


def test_an_unreadable_note_is_reported_not_dropped(tmp_path: Path) -> None:
    _note(tmp_path, "knowledge/ok.md", "---\nwhen: 能读\n---\n正文")
    bad = tmp_path / "knowledge" / "bad.md"
    bad.write_text("---\nwhen: 读不到\n---\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        scan = KnowledgeIndex(tmp_path).scan()
    finally:
        bad.chmod(0o644)  # 留着不改权限，pytest 清理临时目录时会失败

    assert len(scan.entries) == 1
    assert not scan.complete
    assert scan.unreadable[0][0].endswith("bad.md")


def test_the_prompt_says_the_index_is_incomplete(tmp_path: Path) -> None:
    """这句话要出现在提示词里——正要下「我没有相关笔记」结论的是模型。"""
    bad = tmp_path / "knowledge" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\nwhen: x\n---\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        summary = KnowledgeIndex(tmp_path).build_summary()
    finally:
        bad.chmod(0o644)

    assert "INCOMPLETE" in summary
    assert "bad.md" in summary


def test_a_clean_scan_reports_complete(tmp_path: Path) -> None:
    _note(tmp_path, "knowledge/ok.md", "---\nwhen: 能读\n---\n正文")

    scan = KnowledgeIndex(tmp_path).scan()

    assert scan.complete
    assert scan.unreadable == []


def test_undeclared_notes_are_not_treated_as_failures(tmp_path: Path) -> None:
    """没有 when: 的文件是主动弃权，不是读取失败——两者不能混为一谈。"""
    _note(tmp_path, "knowledge/plain.md", "# 只是一篇普通笔记")

    scan = KnowledgeIndex(tmp_path).scan()

    assert scan.entries == []
    assert scan.complete


def test_a_stray_byte_costs_only_that_line_not_the_file(tmp_path: Path) -> None:
    """编码错误按可替换处理：一个坏字节不该让整份笔记消失。"""
    path = tmp_path / "knowledge" / "mixed.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("---\nwhen: 查日志\n---\n".encode() + b"\xff\xfe " + "坏字节".encode())

    scan = KnowledgeIndex(tmp_path).scan()

    assert scan.complete
    assert scan.entries[0]["when"] == "查日志"
