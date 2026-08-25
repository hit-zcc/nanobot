"""互动卡片(interactive)正文提取。

背景：`_extract_interactive_content` 里的 elements 循环原本写成双层，
等于假设 elements 是二维数组；而飞书卡片的 elements 是一维的
⇒ 外层把 element(dict) 当行、内层遍历 dict 得到键名字符串
⇒ `_extract_element_content` 判 isinstance 失败返回空
⇒ parts 恒空，最终落到兜底 `[interactive]`，正文全部丢失。

同时 schema 2.0 把正文放在 body.elements，此前根本没有递归到。

这两个 bug 都不报错，表现成"我读不到卡片内容"，容易被误解成平台限制。
"""

from nanobot.channels.feishu import (
    _extract_element_content,
    _extract_interactive_content,
    _extract_share_card_content,
)


def test_schema_1_0_body_text_is_extracted():
    """schema 1.0：elements 一维，div/markdown 正文都要读到。"""
    card = {
        "header": {"title": {"tag": "plain_text", "content": "用户反馈 #8823"}},
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": "**问题**：录音笔记读不到"}},
            {"tag": "div", "text": {"tag": "lark_md", "content": "**uid**：951685"}},
            {"tag": "hr"},
            {"tag": "markdown", "content": "用户说：转写成文字了，为什么读取不到"},
        ],
    }
    parts = _extract_interactive_content(card)

    assert "**问题**：录音笔记读不到" in parts
    assert "**uid**：951685" in parts
    assert "用户说：转写成文字了，为什么读取不到" in parts
    assert "title: 用户反馈 #8823" in parts


def test_schema_2_0_body_elements_are_extracted():
    """schema 2.0：正文在 body.elements 里。"""
    card = {
        "schema": "2.0",
        "header": {"title": {"tag": "plain_text", "content": "反馈卡片 2.0"}},
        "body": {"elements": [{"tag": "markdown", "content": "这是 2.0 卡片的正文内容"}]},
    }
    parts = _extract_interactive_content(card)

    assert "这是 2.0 卡片的正文内容" in parts
    assert "title: 反馈卡片 2.0" in parts


def test_nested_elements_still_work():
    """兼容极少数把 elements 写成二维的情况，不能因为修一维就把它弄坏。"""
    card = {"elements": [[{"tag": "markdown", "content": "嵌套写法的正文"}]]}

    assert "嵌套写法的正文" in _extract_interactive_content(card)


def test_card_wrapper_is_unwrapped():
    """外层包一层 card 的形状。"""
    card = {"card": {"elements": [{"tag": "markdown", "content": "被 card 包住的正文"}]}}

    assert "被 card 包住的正文" in _extract_interactive_content(card)


def test_share_card_no_longer_degrades_to_placeholder():
    """端到端：走 _extract_share_card_content 不应再退化成 `[interactive]`。"""
    card = {
        "header": {"title": {"tag": "plain_text", "content": "标题"}},
        "elements": [{"tag": "markdown", "content": "正文在这里"}],
    }
    text = _extract_share_card_content(card, "interactive")

    assert "正文在这里" in text
    assert text != "[interactive]"


def test_empty_card_still_falls_back():
    """真的没内容时仍要给出占位符，不能返回空串。"""
    assert _extract_share_card_content({}, "interactive") == "[interactive]"


def test_column_set_and_note_are_extracted():
    """分栏与备注里的文字也算正文。"""
    card = {
        "elements": [
            {
                "tag": "column_set",
                "columns": [
                    {"elements": [{"tag": "markdown", "content": "左栏内容"}]},
                    {"elements": [{"tag": "markdown", "content": "右栏内容"}]},
                ],
            },
            {"tag": "note", "elements": [{"tag": "plain_text", "content": "备注文字"}]},
        ]
    }
    parts = _extract_interactive_content(card)

    assert "左栏内容" in parts
    assert "右栏内容" in parts
    assert "备注文字" in parts


def test_malformed_elements_do_not_raise():
    """脏数据不能把入站链路搞崩 —— 读不到内容可以，抛异常不行。"""
    for bad in (
        {"elements": "not-a-list"},
        {"elements": [None, 123, "text"]},
        {"elements": [{"tag": "div", "text": None}]},
        {"body": None},
    ):
        assert isinstance(_extract_interactive_content(bad), list)


def test_element_content_rejects_non_dict():
    """元素级函数对非 dict 输入保持安全。"""
    assert _extract_element_content("elements") == []
    assert _extract_element_content(None) == []
