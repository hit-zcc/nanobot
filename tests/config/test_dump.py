"""配置真相打印：值从哪来、密钥有没有漏。

这里覆盖的失败长这样：某个键改了却不生效，因为它写错了层级或大小写，
而人只能看见「文件里写了」和「行为没变」，看不见中间那棵真正生效的树。
"""

import json
from pathlib import Path

from nanobot.config.dump import build_dump, redact, render_dump
from nanobot.config.schema import Config


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestRedact:
    def test_masks_secrets_but_keeps_them_recognisable(self):
        out = redact("appSecret", "abcdefghijklmnop")
        assert "abcdefghijklmnop" not in out
        assert out.startswith("abcd")
        assert "16 chars" in out

    def test_short_secrets_reveal_nothing(self):
        assert redact("apiKey", "abc123") == "***"

    def test_leaves_ordinary_values_alone(self):
        assert redact("model", "claude-opus-4-5") == "claude-opus-4-5"

    def test_leaves_empty_values_alone(self):
        """空串本身就是信息——「这个凭据没配」需要看得见。"""
        assert redact("apiKey", "") == ""

    def test_non_strings_pass_through(self):
        """maxTokens 之类名字里带 token 的数字，不该被打码。"""
        assert redact("maxTokens", 16384) == 16384


class TestOrigin:
    def test_marks_values_that_came_from_the_file(self, tmp_path):
        path = _write(tmp_path, {"agents": {"defaults": {"temperature": 0.7}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        assert dump["settings"]["agents"]["defaults"]["temperature"]["from"] == "file"

    def test_marks_values_that_came_from_defaults(self, tmp_path):
        path = _write(tmp_path, {"agents": {"defaults": {"temperature": 0.7}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        # 同一层里没写的那个键
        assert dump["settings"]["agents"]["defaults"]["maxTokens"]["from"] == "default"

    def test_a_typo_under_channels_survives_into_the_dump(self, tmp_path):
        """`channels` 是 extra="allow"（插件要用），所以拼错的键不会报错。

        这正是「配了但没人读」能悄悄发生的地方——写错的键连个警告都没有。
        dump 至少把它原样摆出来，让人自己看见 `feshu` 旁边还有个 `feishu`。
        """
        path = _write(tmp_path, {"channels": {"feshu": {"enabled": True}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        assert "feshu" in dump["settings"]["channels"]

    def test_top_level_typos_are_rejected_outright(self, tmp_path):
        """记录一个好消息：顶层拼错不会静默失效，pydantic 当场就拒。

        写这条是因为我本来假设它会静默——实测不会，那这类 bug 就不存在，
        不必为它设计任何提示。
        """
        import pydantic
        import pytest
        with pytest.raises(pydantic.ValidationError):
            Config.model_validate({"temperature": 0.7})

    def test_unreadable_file_makes_everything_a_default(self, tmp_path):
        """配置文件坏了不该报错——「全是默认值」本身就是答案。"""
        path = tmp_path / "config.json"
        path.write_text("{ not json", encoding="utf-8")
        dump = build_dump(Config(), path)
        assert dump["settings"]["agents"]["defaults"]["maxTokens"]["from"] == "default"


class TestRender:
    def test_marks_file_set_values_with_a_bullet(self, tmp_path):
        path = _write(tmp_path, {"agents": {"defaults": {"temperature": 0.7}}})
        text = render_dump(build_dump(Config.model_validate(json.loads(path.read_text())), path))
        assert "• temperature: 0.7" in text

    def test_changed_only_hides_defaults(self, tmp_path):
        path = _write(tmp_path, {"agents": {"defaults": {"temperature": 0.7}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        text = render_dump(dump, show_defaults=False)
        assert "temperature" in text
        assert "maxTokens" not in text

    def test_changed_only_drops_sections_that_became_empty(self, tmp_path):
        """一个标题下面全被过滤光了，就不该留个空标题在那。"""
        path = _write(tmp_path, {"agents": {"defaults": {"temperature": 0.7}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        text = render_dump(dump, show_defaults=False)
        assert "providers:" not in text

    def test_secrets_never_reach_the_output(self, tmp_path):
        path = _write(tmp_path, {"providers": {"anthropic": {"apiKey": "sk-verysecret-value"}}})
        dump = build_dump(Config.model_validate(json.loads(path.read_text())), path)
        assert "sk-verysecret-value" not in render_dump(dump)

    def test_says_so_when_the_config_file_is_missing(self, tmp_path):
        text = render_dump(build_dump(Config(), tmp_path / "nope.json"))
        assert "does not exist" in text
