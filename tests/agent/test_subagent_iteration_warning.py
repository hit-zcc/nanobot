"""The background task must be *told* it is running out of iterations.

`iteration_warning_message` only reaches the user via `hook.on_notice`; the
subagent model never sees it. Without `iteration_warning_directive` the model
runs straight into the hard limit, `final_content` gets overwritten by the
runner, and the user is handed "被系统停止" with no summary of the work that
was in fact already done.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus


async def _captured_spec(tmp_path: Path) -> AgentRunSpec:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = SubagentManager(
        provider=provider, workspace=tmp_path, bus=MessageBus(),
    )

    captured: list[AgentRunSpec] = []

    async def _run(spec: AgentRunSpec) -> AgentRunResult:
        captured.append(spec)
        return AgentRunResult(final_content="done", messages=[])

    manager.runner.run = AsyncMock(side_effect=_run)

    await manager._run_subagent(
        "t1", "审计线上索引", "审计", {"channel": "feishu", "chat_id": "oc_g"},
    )

    assert captured, "subagent never reached runner.run"
    return captured[0]


@pytest.mark.asyncio
async def test_subagent_warns_the_model_not_just_the_user(tmp_path: Path) -> None:
    spec = await _captured_spec(tmp_path)

    assert spec.iteration_warning_remaining is not None
    assert spec.iteration_warning_message is not None
    assert spec.iteration_warning_directive is not None, (
        "only the user-facing notice is configured; the subagent model gets no "
        "in-conversation warning and will run into the hard limit"
    )


@pytest.mark.asyncio
async def test_directive_tells_it_to_land_the_work_itself(tmp_path: Path) -> None:
    """A subagent has nobody to hand off to — copying the foreground wording
    ("发送后到此为止，等待用户确认") would strand the deliverable."""
    spec = await _captured_spec(tmp_path)
    directive = spec.iteration_warning_directive or ""

    assert directive.startswith("[自动提醒 · 非用户发言]")
    assert "等待用户确认" not in directive
    assert "最终" in directive


@pytest.mark.asyncio
async def test_warning_can_actually_fire_within_the_budget(tmp_path: Path) -> None:
    """A threshold >= max_iterations is silently ignored by the runner."""
    spec = await _captured_spec(tmp_path)

    assert spec.max_iterations > spec.iteration_warning_remaining
