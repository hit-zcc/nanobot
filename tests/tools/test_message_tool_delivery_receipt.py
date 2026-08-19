"""投递回执：message 工具必须报告**真实**投递结果，不许谎报。

背景（2026-08-19）：群里发文件反复失败，而我每次都回复"已发给你"。
根因不是判断失误 —— `publish_outbound` 只是 queue.put 就返回，真正投递在
另一个协程里，飞书失败时只 logger.error 不抛异常，所以工具在**物理上**
拿不到结果，只能返回写死的 "Message sent to ..."。

这些用例锁住修复后的行为：成功说 delivered，失败说 FAILED，等不到说没确认。
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


class _Channel(BaseChannel):
    """真实走一遍 manager 的假 channel；send 的成败由构造参数决定。

    刻意**不复刻**飞书的实现细节，只复刻契约：失败时抛异常。
    """

    name = "mock"
    display_name = "Mock"

    def __init__(self, config, bus, fail: bool = False):
        super().__init__(config, bus)
        self._fail = fail
        self.sent: list[OutboundMessage] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, msg: OutboundMessage):
        if self._fail:
            raise RuntimeError("Feishu rejected file message to chat_id=oc_x (230002)")
        self.sent.append(msg)

    async def send_delta(self, chat_id, delta, metadata=None):
        pass


def _manager(fail: bool) -> tuple[ChannelManager, MessageBus, _Channel]:
    config = Config()
    config.channels.send_max_retries = 1  # 别让重试拖慢用例
    bus = MessageBus()
    mgr = ChannelManager(config, bus)
    ch = _Channel(config, bus, fail=fail)
    mgr.channels["mock"] = ch
    return mgr, bus, ch


async def _run_once(fail: bool, media: list[str] | None = None) -> str:
    """跑一次完整链路：工具 → bus → manager → channel，返回工具的返回值。"""
    mgr, bus, ch = _manager(fail)
    tool = MessageTool(send_callback=bus.publish_outbound)
    tool.set_context("mock", "oc_group", "om_parent")

    pump = asyncio.create_task(mgr._dispatch_outbound())
    try:
        return await asyncio.wait_for(
            tool.execute(content="hi", media=media or []), timeout=10
        )
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_reports_success_only_after_real_delivery() -> None:
    result = await _run_once(fail=False)
    assert "delivered" in result.lower()
    assert "mock:oc_group" in result


@pytest.mark.asyncio
async def test_reports_failure_instead_of_claiming_sent() -> None:
    """核心用例：投递失败时**绝不能**出现"已发送"的措辞。"""
    result = await _run_once(fail=True)

    assert "FAILED" in result, f"投递失败却没报错: {result!r}"
    assert "230002" in result, "失败详情要带上原始错误，否则没法诊断"
    # 反向断言：不许出现任何会被读成"成功"的说法
    assert "delivered" not in result.lower()
    assert "Message sent" not in result


@pytest.mark.asyncio
async def test_failure_message_tells_agent_not_to_claim_sent() -> None:
    """返回值要显式指示 agent 别嘴硬 —— 这条是给模型看的，不是给人看的。"""
    result = await _run_once(fail=True)
    assert "did NOT receive" in result
    assert "Do not claim" in result


@pytest.mark.asyncio
async def test_attachment_failure_is_reported() -> None:
    """带附件时同样要真实报告 —— 这正是本次线上故障的形态。"""
    result = await _run_once(fail=True, media=["/tmp/nonexistent_probe.md"])
    assert "FAILED" in result


@pytest.mark.asyncio
async def test_unknown_channel_settles_instead_of_hanging() -> None:
    """未知 channel 以前会让调用方一直等到超时。"""
    config = Config()
    bus = MessageBus()
    mgr = ChannelManager(config, bus)  # 一个 channel 都没注册

    tool = MessageTool(send_callback=bus.publish_outbound)
    tool.set_context("nope", "chat", None)

    pump = asyncio.create_task(mgr._dispatch_outbound())
    try:
        result = await asyncio.wait_for(tool.execute(content="hi"), timeout=5)
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    assert "FAILED" in result
    assert "unknown channel" in result


@pytest.mark.asyncio
async def test_no_event_loop_context_is_explicit_about_uncertainty() -> None:
    """拿不到回执时要说"unconfirmed"，不能假装成功。"""
    sent: list[OutboundMessage] = []

    async def _cb(msg: OutboundMessage) -> None:
        sent.append(msg)
        # 故意不 settle，模拟没有 manager 在消费的场景

    tool = MessageTool(send_callback=_cb)
    tool.set_context("mock", "chat", None)

    from nanobot.agent.tools import message as message_mod

    original = message_mod._DELIVERY_TIMEOUT_S
    message_mod._DELIVERY_TIMEOUT_S = 0.2
    try:
        result = await tool.execute(content="hi")
    finally:
        message_mod._DELIVERY_TIMEOUT_S = original

    assert "did NOT confirm" in result
    assert "Do not tell the user it was sent" in result
    assert len(sent) == 1
