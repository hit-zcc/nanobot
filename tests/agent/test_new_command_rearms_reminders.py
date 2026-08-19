"""`/new` keeps the session key but throws the history away.

Since the reminder engines de-duplicate by session key, a fresh conversation
would otherwise inherit "already said that" from the one it replaced — which
is exactly the state the user just asked to be rid of.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.agent.tool_preflight import PreflightEngine, Playbook
from nanobot.agent.tool_triggers import TriggerEngine
from nanobot.command.builtin import cmd_new


class _Session:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.last_consolidated = 0

    def clear(self) -> None:
        self.messages = []


def _ctx(tmp_path):
    import re

    rules = tmp_path / "rules.yaml"
    rules.write_text(
        '- id: r\n  when:\n    tool: exec\n    arg_matches: "ls"\n  say: 提醒\n',
        encoding="utf-8",
    )
    loop = SimpleNamespace(
        tool_triggers=TriggerEngine.load(rules),
        tool_preflight=PreflightEngine(playbooks=[
            Playbook("p.md", body="配方", arg_matches=re.compile("ls")),
        ]),
        sessions=SimpleNamespace(save=lambda s: None, invalidate=lambda k: None,
                                 get_or_create=lambda k: _Session(k)),
        _schedule_background=lambda coro: coro.close(),
    )
    session = _Session("feishu:a")
    msg = SimpleNamespace(channel="feishu", chat_id="a")
    return SimpleNamespace(loop=loop, session=session, key=session.key, msg=msg), loop


@pytest.mark.asyncio
async def test_new_rearms_the_engines_for_that_session(tmp_path):
    ctx, loop = _ctx(tmp_path)
    args = {"command": "ls"}
    assert loop.tool_triggers.check("exec", args, "", session_key="feishu:a")
    assert loop.tool_preflight.check("exec", args, session_key="feishu:a")

    await cmd_new(ctx)

    assert loop.tool_triggers.check("exec", args, "", session_key="feishu:a")
    assert loop.tool_preflight.check("exec", args, session_key="feishu:a")


@pytest.mark.asyncio
async def test_new_does_not_rearm_other_sessions(tmp_path):
    """/new is scoped to one chat; it must not make every other conversation
    repeat its reminders."""
    ctx, loop = _ctx(tmp_path)
    args = {"command": "ls"}
    loop.tool_triggers.check("exec", args, "", session_key="feishu:other")

    await cmd_new(ctx)

    assert loop.tool_triggers.check("exec", args, "", session_key="feishu:other") == []


@pytest.mark.asyncio
async def test_new_survives_a_loop_without_engines(tmp_path):
    """Test doubles and older loops may not carry the engines at all."""
    ctx, loop = _ctx(tmp_path)
    del loop.tool_triggers
    loop.tool_preflight = None

    out = await cmd_new(ctx)

    assert "New session started." in out.content
