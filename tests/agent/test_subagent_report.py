"""The subagent's only mid-task channel to the user."""

import pytest

from nanobot.agent.subagent import SubagentTaskState
from nanobot.agent.tools.report import ReportTool
from nanobot.bus.queue import MessageBus


async def _drain(bus: MessageBus) -> list:
    out = []
    while not bus.outbound.empty():
        out.append(bus.outbound.get_nowait())
    return out


@pytest.mark.asyncio
async def test_report_reaches_the_user_even_with_progress_disabled() -> None:
    """It must not be tagged _progress: the progress switch is off in production."""
    bus = MessageBus()
    tool = ReportTool(bus=bus, channel="feishu.jarvis", chat_id="oc_g", label="重构线程池")

    await tool.execute(message="得到的 ES 索引已被删，继续跑会清空线上数据，先停一下")

    (sent,) = await _drain(bus)
    assert sent.channel == "feishu.jarvis"
    assert sent.chat_id == "oc_g"
    assert "重构线程池" in sent.content
    assert "先停一下" in sent.content
    assert sent.metadata["_important_notice"] is True
    assert "_progress" not in sent.metadata


@pytest.mark.asyncio
async def test_report_mirrors_into_task_state() -> None:
    """The main agent needs to know what the user was already told."""
    bus = MessageBus()
    state = SubagentTaskState(
        task_id="t1", label="l", task="t",
        origin_channel="feishu", origin_chat_id="c", session_key="feishu:c",
    )
    tool = ReportTool(
        bus=bus, channel="feishu", chat_id="c", label="l",
        on_report=lambda text: setattr(state, "last_report", text),
    )

    await tool.execute(message="需要你确认是否可以删索引")

    assert state.last_report == "需要你确认是否可以删索引"


@pytest.mark.asyncio
async def test_empty_report_is_rejected_without_pinging_the_user() -> None:
    bus = MessageBus()
    tool = ReportTool(bus=bus, channel="feishu", chat_id="c", label="l")

    result = await tool.execute(message="   ")

    assert "Error" in result
    assert await _drain(bus) == []


def test_description_states_the_bar() -> None:
    """The bar is the feature; a generic description turns this into a progress log."""
    description = ReportTool(bus=MessageBus(), channel="c", chat_id="i", label="l").description
    assert "ONLY" in description
    assert "Do NOT" in description
    assert "zero times" in description
