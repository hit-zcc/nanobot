"""跨会话误投：reply 接口按父消息所在会话投递，不看 receive_id。

背景（2026-08-19）：往 test 群发附件，聪聪只看到文字、看不到附件。
实测消息详情：文件那条落在 p2p 私聊 oc_a276…，文字那条在群 oc_6b68…。
原因是 media 是本次调用的第一条消息、占用了 reply 名额，而 reply 的父消息
来自另一个会话。API 全程 success，所以没有任何报错——只是东西发错了地方。
"""
import asyncio
from unittest.mock import MagicMock

import pytest

try:
    from nanobot.channels import feishu
    FEISHU_AVAILABLE = getattr(feishu, "FEISHU_AVAILABLE", False)
except ImportError:  # pragma: no cover
    FEISHU_AVAILABLE = False

if not FEISHU_AVAILABLE:  # pragma: no cover
    pytest.skip("Feishu dependencies not installed (lark-oapi)", allow_module_level=True)

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig

GROUP = "oc_group_target"
P2P = "oc_p2p_other"
PARENT = "om_parent_from_p2p"


def _channel() -> FeishuChannel:
    config = FeishuConfig(
        enabled=True, app_id="cli_test", app_secret="secret",
        allow_from=["*"], reply_to_message=True,
    )
    ch = FeishuChannel(config, MessageBus())
    ch._client = MagicMock()
    ch._loop = None
    return ch


def test_reply_skipped_when_parent_belongs_to_another_chat(tmp_path):
    """父消息来自别的会话时，必须退回 create，绝不能 reply 过去。"""
    ch = _channel()
    ch._remember(ch._inbound_chats, PARENT, P2P)

    replied: list = []
    created: list = []
    ch._reply_message_with_id_sync = lambda pid, mt, c: (replied.append((pid, mt)) or "om_r")
    ch._send_message_sync = lambda rt, rid, mt, c: (created.append((rid, mt)) or "om_c")
    ch._upload_file_sync = lambda path: "file_key_test"

    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")

    asyncio.run(ch.send(OutboundMessage(
        channel="feishu", chat_id=GROUP, content="正文",
        media=[str(f)], metadata={"message_id": PARENT},
    )))

    assert replied == [], "父消息属于别的会话，不该走 reply"
    # 附件 + 正文都必须投向显式指定的群
    assert [rid for rid, _ in created] == [GROUP, GROUP]
    assert "file" in [mt for _, mt in created]


def test_reply_still_used_within_same_chat(tmp_path):
    """同一会话内的父消息，引用行为保持不变。"""
    ch = _channel()
    ch._remember(ch._inbound_chats, PARENT, GROUP)

    replied: list = []
    ch._reply_message_with_id_sync = lambda pid, mt, c: (replied.append((pid, mt)) or "om_r")
    ch._send_message_sync = lambda rt, rid, mt, c: "om_c"

    asyncio.run(ch.send(OutboundMessage(
        channel="feishu", chat_id=GROUP, content="正文",
        metadata={"message_id": PARENT},
    )))

    assert replied and replied[0][0] == PARENT


def test_message_tool_drops_message_id_for_explicit_other_chat():
    """工具层：显式指定别的 chat_id 时不得沿用当前会话的 message_id。"""
    captured: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        captured.append(msg)
        if msg.delivery and not msg.delivery.done():
            msg.delivery.set_result((True, ""))

    tool = MessageTool(send_callback=_send, default_channel="feishu",
                       default_chat_id=P2P, default_message_id=PARENT)

    asyncio.run(tool.execute(content="hi", chat_id=GROUP))
    assert captured[-1].metadata.get("message_id") is None

    # 不指定 chat_id 时仍然引用当前会话的消息
    asyncio.run(tool.execute(content="hi"))
    assert captured[-1].metadata.get("message_id") == PARENT


def test_card_markup_normalizes_at_for_streaming():
    """流式卡片走 _sanitize_card_markup，at 规范化必须在这条路上生效。"""
    from nanobot.channels.feishu import _sanitize_card_markup

    assert _sanitize_card_markup('<at user_id="ou_bob">牛小数</at> 在吗') == (
        '<at id="ou_bob"></at> 在吗'
    )
    # 已经是卡片语法则原样保留（幂等）
    assert _sanitize_card_markup('<at id="ou_bob"></at> 在吗') == '<at id="ou_bob"></at> 在吗'
    # 增量流式可能把标签劈成半截，此时不改写，等下一帧收全
    assert _sanitize_card_markup('<at user_id="ou_b') == '<at user_id="ou_b'
