"""合并转发(merge_forward)消息展开。

背景：入站处理原本对 merge_forward 只吐一句硬编码的
`[merged forward messages]`，正文全部丢失。我据此三次（08-09 / 08-17 / 08-21）
告诉聪聪"飞书没把内容透传过来" —— 那是**读码推断**，从没验证过。

实测真相：子消息**直接躺在** `im/v1/messages/{id}` 的 items 数组里，
靠 `upper_message_id` 指回父节点串成树，一个普通 GET 就能拿全，
不需要任何额外接口。下面的 fixture 结构逐字取自 2026-08-21 22:25
聪聪转发的那条真实消息。

⚠️ 嵌套的 interactive 卡片，飞书服务端只下发「请升级至最新版本客户端」占位，
那部分确实取不到 —— 是平台行为，不是解析失败，别再混为一谈。
"""

import json
from types import SimpleNamespace

import pytest

from nanobot.channels.feishu import FeishuChannel

ROOT = "om_root"


def _msg(mid, msg_type, content, upper=None, sender=None):
    return SimpleNamespace(
        message_id=mid,
        msg_type=msg_type,
        upper_message_id=upper,
        mentions=None,
        sender=SimpleNamespace(id=sender) if sender else None,
        body=SimpleNamespace(content=json.dumps(content) if isinstance(content, dict) else content),
    )


# 真实结构：外层 merge_forward → 内含 interactive + 又一个 merge_forward → 内含 text
REAL_ITEMS = [
    _msg(ROOT, "merge_forward", "Merged and Forwarded Message"),
    _msg("om_card1", "interactive",
         {"title": None, "elements": [[
             {"tag": "img", "image_key": "img_v3_xxx"},
             {"tag": "text", "text": "请升级至最新版本客户端，以查看内容"},
         ]]},
         upper=ROOT),
    _msg("om_inner", "merge_forward", "Merged and Forwarded Message", upper=ROOT),
    _msg("om_text", "text",
         {"text": "这里不是有牛小数的id吗 你记一下 这样你就可以去问了@_user_1"},
         upper="om_inner", sender="ou_abc"),
]


class _FakeResp:
    def __init__(self, items, ok=True):
        self.data = SimpleNamespace(items=items)
        self._ok = ok
        self.code = 0 if ok else 99
        self.msg = "success" if ok else "boom"

    def success(self):
        return self._ok


def _channel(items, ok=True):
    ch = FeishuChannel.__new__(FeishuChannel)
    ch._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(
            message=SimpleNamespace(get=lambda req: _FakeResp(items, ok))
        ))
    )
    return ch


def test_nested_text_is_recovered():
    """最要紧的一条：套了两层的正文必须能读出来。"""
    out = _channel(REAL_ITEMS)._expand_merge_forward_sync(ROOT)

    assert out is not None
    assert "牛小数的id" in out, "嵌套两层的文本正文丢了"
    assert "[合并转发的聊天记录]" in out


def test_nested_structure_is_indented():
    """层级要看得出来，否则模型分不清谁包着谁。"""
    out = _channel(REAL_ITEMS)._expand_merge_forward_sync(ROOT)
    lines = out.splitlines()

    inner = next(ln for ln in lines if "合并转发]" in ln)
    text_line = next(ln for ln in lines if "牛小数" in ln)
    assert len(text_line) - len(text_line.lstrip()) > len(inner) - len(inner.lstrip())


def test_sender_id_is_prefixed_when_available():
    out = _channel(REAL_ITEMS)._expand_merge_forward_sync(ROOT)
    assert "ou_abc:" in out


def test_platform_placeholder_is_passed_through_not_swallowed():
    """飞书自己下发的降级占位要原样带出来，方便区分'平台不给'和'我没解析'。"""
    out = _channel(REAL_ITEMS)._expand_merge_forward_sync(ROOT)
    assert "请升级至最新版本客户端" in out


def test_no_children_returns_none():
    """只有外层自己时返回 None，让调用方退回占位符。"""
    only_root = [_msg(ROOT, "merge_forward", "Merged and Forwarded Message")]
    assert _channel(only_root)._expand_merge_forward_sync(ROOT) is None


def test_api_failure_returns_none():
    assert _channel(REAL_ITEMS, ok=False)._expand_merge_forward_sync(ROOT) is None


def test_api_exception_returns_none():
    ch = FeishuChannel.__new__(FeishuChannel)

    def _boom(req):
        raise RuntimeError("network down")

    ch._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(get=_boom)))
    )
    assert ch._expand_merge_forward_sync(ROOT) is None


def test_depth_guard_stops_runaway_nesting():
    """套娃不能把入站链路拖死。"""
    items = [_msg(ROOT, "merge_forward", "x")]
    parent = ROOT
    for i in range(30):
        mid = f"om_d{i}"
        items.append(_msg(mid, "merge_forward", "x", upper=parent))
        parent = mid
    items.append(_msg("om_deep", "text", {"text": "最深处"}, upper=parent))

    out = _channel(items)._expand_merge_forward_sync(ROOT)
    assert "更深的嵌套已省略" in out
    assert "最深处" not in out


def test_item_count_guard():
    """宽度也要有上限。"""
    items = [_msg(ROOT, "merge_forward", "x")]
    for i in range(200):
        items.append(_msg(f"om_w{i}", "text", {"text": f"消息{i}"}, upper=ROOT))

    out = _channel(items)._expand_merge_forward_sync(ROOT)
    assert "更多消息已省略" in out
    assert len(out.splitlines()) < 100


def test_malformed_children_do_not_raise():
    """脏数据只能降级，不能抛异常。"""
    items = [
        _msg(ROOT, "merge_forward", "x"),
        _msg("om_bad1", "text", "not-json", upper=ROOT),
        _msg("om_bad2", "text", {}, upper=ROOT),
        SimpleNamespace(message_id="om_bad3", msg_type="text",
                        upper_message_id=ROOT, mentions=None, sender=None, body=None),
    ]
    out = _channel(items)._expand_merge_forward_sync(ROOT)
    assert isinstance(out, str)


@pytest.mark.parametrize("msg_type,content,expect", [
    ("image", {"image_key": "k"}, "[image]"),
    ("audio", {"file_key": "k"}, "[audio]"),
    ("file", {"file_key": "k"}, "[file]"),
])
def test_media_children_render_as_placeholders(msg_type, content, expect):
    items = [
        _msg(ROOT, "merge_forward", "x"),
        _msg("om_m", msg_type, content, upper=ROOT),
    ]
    assert expect in _channel(items)._expand_merge_forward_sync(ROOT)


def _channel_with_refetch(items, card_bodies, calls):
    """按 message_id 分别返回不同内容 —— 模拟"父消息里是降级占位、
    单独 GET 同一条子消息拿到完整正文"这个真实行为。"""
    def _get(req):
        mid = req.message_id if hasattr(req, "message_id") else None
        # lark 的 builder 把 message_id 放在 request 上，取不到就翻 body
        mid = mid or getattr(req, "_message_id", None)
        calls.append(mid)
        if mid in card_bodies:
            return _FakeResp([_msg(mid, "interactive", card_bodies[mid])])
        return _FakeResp(items)

    ch = FeishuChannel.__new__(FeishuChannel)
    ch._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(get=_get)))
    )
    return ch


DEGRADED = {"title": None, "elements": [[
    {"tag": "img", "image_key": "k"},
    {"tag": "text", "text": "请升级至最新版本客户端，以查看内容"},
]]}

FULL_CARD = {"schema": "2.0", "body": {"elements": [
    {"tag": "markdown", "content": "牛小数 open_id ou_a18d1936…"},
]}}


def test_degraded_card_is_refetched_for_full_body():
    """核心：父消息里给的是降级占位，回查后必须拿到真正文。"""
    items = [
        _msg(ROOT, "merge_forward", "x"),
        _msg("om_card", "interactive", DEGRADED, upper=ROOT),
    ]
    calls = []
    ch = _channel_with_refetch(items, {"om_card": FULL_CARD}, calls)
    out = ch._expand_merge_forward_sync(ROOT)

    assert "牛小数 open_id" in out, "回查没生效，仍是降级占位"
    assert "请升级至最新版本客户端" not in out
    assert "om_card" in calls, "根本没发起回查"


def test_refetch_failure_falls_back_to_inline_body():
    """回查失败不能把已有的降级内容也弄丢 —— 有总比没有强。"""
    items = [
        _msg(ROOT, "merge_forward", "x"),
        _msg("om_card", "interactive", DEGRADED, upper=ROOT),
    ]

    def _get(req):
        mid = getattr(req, "message_id", None)
        if mid == "om_card":
            return _FakeResp([], ok=False)
        return _FakeResp(items)

    ch = FeishuChannel.__new__(FeishuChannel)
    ch._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(get=_get)))
    )
    out = ch._expand_merge_forward_sync(ROOT)

    assert out is not None
    assert "请升级至最新版本客户端" in out


def test_refetch_is_capped():
    """几十张卡片不能变成几十次 API 调用。"""
    items = [_msg(ROOT, "merge_forward", "x")]
    bodies = {}
    for i in range(40):
        mid = f"om_c{i}"
        items.append(_msg(mid, "interactive", DEGRADED, upper=ROOT))
        bodies[mid] = FULL_CARD
    calls = []
    ch = _channel_with_refetch(items, bodies, calls)
    ch._expand_merge_forward_sync(ROOT)

    card_calls = [c for c in calls if c and c.startswith("om_c")]
    assert len(card_calls) <= FeishuChannel._MERGE_FORWARD_MAX_REFETCH, (
        f"回查了 {len(card_calls)} 次，超过上限"
    )


def test_non_card_children_are_not_refetched():
    """只有 interactive 才回查，文本消息不该白白多打一次 API。"""
    items = [
        _msg(ROOT, "merge_forward", "x"),
        _msg("om_t", "text", {"text": "普通文本"}, upper=ROOT),
    ]
    calls = []
    ch = _channel_with_refetch(items, {}, calls)
    ch._expand_merge_forward_sync(ROOT)

    assert "om_t" not in calls


def test_expander_is_actually_wired_into_inbound_path():
    """守卫「接线」本身。

    上面所有用例测的都是 `_expand_merge_forward_sync` 这个函数，
    实测把 `_on_message` 里的调用删掉后**它们仍然全绿** —— 函数是好的，
    但没人调用它，正文照样丢。这正是我这次栽的坑的形状：
    验证了零件，没验证装配。

    ⚠️ 局限说清楚：这是源码级断言，不是行为测试。它能抓住"调用被删"，
    抓不住"调用了但结果没进 content_parts"。真正的端到端要构造完整的
    飞书事件对象走 `_on_message`，成本高很多；在那之前，这条至少让
    删接线这个动作不再是静默的。
    """
    import inspect

    src = inspect.getsource(FeishuChannel._on_message)
    assert "_expand_merge_forward_sync" in src, (
        "_on_message 里没有调用展开函数 —— 合并转发的正文会退回占位符"
    )
