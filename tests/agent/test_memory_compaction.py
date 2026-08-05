"""MEMORY.md must stay under a token cap: it is re-read into every system prompt."""

from nanobot.agent.memory import MemoryStore
from nanobot.utils.helpers import estimate_text_tokens


def _fat_entry(tag: str, size: int = 400) -> str:
    return f"- **{tag} 结案**：" + "细节 " * size


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path)


def test_compaction_is_noop_below_cap(tmp_path) -> None:
    store = _store(tmp_path)
    store.write_long_term("- **小事**：一行结论\n")

    assert store.compact_long_term(24_000) is False
    assert store.read_long_term() == "- **小事**：一行结论\n"
    assert not list(store.memory_dir.glob("MEMORY_archive_*.md"))


def test_compaction_spills_bodies_and_keeps_headlines(tmp_path) -> None:
    store = _store(tmp_path)
    store.write_long_term("\n".join(_fat_entry(f"案例{i}") for i in range(6)))
    before = estimate_text_tokens(store.read_long_term())

    assert store.compact_long_term(2000) is True

    after = store.read_long_term()
    assert estimate_text_tokens(after) <= 2000 < before
    # Every entry is still findable by headline, and its body moved rather than vanished.
    archive = next(iter(store.memory_dir.glob("MEMORY_archive_*.md"))).read_text()
    for i in range(6):
        assert f"**案例{i} 结案**" in after
        if f"**案例{i} 结案**：细节" not in after:
            assert f"**案例{i} 结案**：细节" in archive
    assert "MEMORY_archive_" in after  # pointer telling the agent where the bodies went


def test_compaction_leaves_small_entries_and_headings_alone(tmp_path) -> None:
    store = _store(tmp_path)
    store.write_long_term(
        "## 铁律\n\n- 别硬编码 open_id\n- 巡查完必须汇报\n\n## 其他\n\n" + _fat_entry("大案")
    )

    assert store.compact_long_term(300) is True

    after = store.read_long_term()
    assert "## 铁律" in after and "## 其他" in after
    assert "- 别硬编码 open_id" in after
    assert "- 巡查完必须汇报" in after
    assert "细节 细节" not in after


def test_compaction_reports_when_nothing_can_be_compacted(tmp_path) -> None:
    """A cap smaller than the irreducible content must not corrupt the file."""
    store = _store(tmp_path)
    store.write_long_term("- 一\n- 二\n- 三\n")

    assert store.compact_long_term(1) is False
    assert store.read_long_term() == "- 一\n- 二\n- 三\n"


def test_headline_falls_back_to_first_clause_without_bold(tmp_path) -> None:
    store = _store(tmp_path)
    store.write_long_term("- 无粗体标题的条目：" + "正文 " * 400)

    assert store.compact_long_term(200) is True

    after = store.read_long_term()
    assert "- 无粗体标题的条目：" in after
    assert "正文 正文" not in after
