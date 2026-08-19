"""The loop must tell the reminder engines *which conversation* it is in.

Both engines are built once in ``AgentLoop.__init__`` and therefore live as
long as the process. If the loop does not hand them a session key, their
"once per session" state degrades into "once per process": a rule that fired
in one chat stays silent in every group, cron run and heartbeat tick until a
restart. That is the whole bug, so it is worth an end-to-end check rather
than trusting the argument to be passed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    return loop


class _Recorder:
    """Stands in for either engine; records the session key it was given."""

    def __init__(self) -> None:
        self.keys: list[str | None] = []

    def check(self, *_args, session_key=None, **_kwargs):
        self.keys.append(session_key)
        return []


async def _one_tool_turn(loop, **kwargs):
    calls = {"n": 0}

    async def chat_with_retry(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={"command": "ls"})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    loop.provider.chat_with_retry = chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")
    return await loop._run_agent_loop([], **kwargs)


@pytest.mark.asyncio
async def test_session_key_reaches_both_engines(tmp_path):
    loop = _make_loop(tmp_path)
    triggers, preflight = _Recorder(), _Recorder()
    loop.tool_triggers, loop.tool_preflight = triggers, preflight

    await _one_tool_turn(loop, session_key="feishu:oc_123")

    assert preflight.keys == ["feishu:oc_123"]
    assert triggers.keys == ["feishu:oc_123"]


@pytest.mark.asyncio
async def test_falls_back_to_the_origin_when_no_key_is_given(tmp_path):
    """Older call sites (and tests) pass no key; they must still land in a
    per-conversation bucket rather than one global one."""
    loop = _make_loop(tmp_path)
    triggers, preflight = _Recorder(), _Recorder()
    loop.tool_triggers, loop.tool_preflight = triggers, preflight

    await _one_tool_turn(loop, channel="feishu", chat_id="oc_9")

    assert preflight.keys == ["feishu:oc_9"]
    assert triggers.keys == ["feishu:oc_9"]


@pytest.mark.asyncio
async def test_two_chats_each_get_the_reminder(tmp_path):
    """The real symptom, reproduced through the loop with the real engine."""
    from nanobot.agent.tool_triggers import TriggerEngine

    loop = _make_loop(tmp_path)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        '- id: r\n  when:\n    tool: exec\n    arg_matches: "ls"\n  say: 提醒\n',
        encoding="utf-8",
    )
    loop.tool_triggers = TriggerEngine.load(rules)

    def reminded(msgs: list[dict]) -> bool:
        return any("提醒" in str(m.get("content", "")) and m.get("role") == "user"
                   for m in msgs)

    _, _, first = await _one_tool_turn(loop, session_key="feishu:a")
    _, _, again = await _one_tool_turn(loop, session_key="feishu:a")
    _, _, other = await _one_tool_turn(loop, session_key="cron:nightly")

    assert reminded(first), "first turn in a chat should get the reminder"
    assert not reminded(again), "same chat, second turn: already said"
    assert reminded(other), "a different session must still be armed"
