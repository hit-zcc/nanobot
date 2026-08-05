"""A request must end with a user turn.

Background-task results used to be injected as an *assistant* message, which
makes the request a prefill. Providers reject that outright — Claude answers
400 "This model does not support assistant message prefill" — so every
completed background task blew up instead of being reported to the user.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    with patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        return AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)


def _subagent_msg(content: str = "[Subagent 'audit' completed]\n\nResult:\nall good"):
    return InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="feishu:c1",
        content=content,
        session_key_override="feishu:c1",
    )


class TestContextBuilder:
    def test_built_request_always_ends_with_user(self, tmp_path):
        from nanobot.agent.context import ContextBuilder

        builder = ContextBuilder(workspace=tmp_path)
        messages = builder.build_messages(
            history=[{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}],
            current_message="[Subagent 'audit' completed]",
        )

        assert messages[-1]["role"] == "user"
        assert messages[0]["role"] == "system"

    def test_no_caller_can_override_the_trailing_role(self, tmp_path):
        """The old `current_role` knob existed only to produce this bug."""
        from nanobot.agent.context import ContextBuilder

        builder = ContextBuilder(workspace=tmp_path)
        with pytest.raises(TypeError):
            builder.build_messages(
                history=[], current_message="x", current_role="assistant",
            )


class TestSubagentResultTurn:
    @pytest.mark.asyncio
    async def test_result_is_injected_as_user_turn(self, tmp_path):
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="feishu:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        seen: dict[str, list] = {}

        async def chat_with_retry(**kwargs):
            seen["messages"] = list(kwargs["messages"])
            return LLMResponse(content="reported", tool_calls=[], usage={})

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None

        out = await loop._process_message(_subagent_msg())

        sent = seen["messages"]
        assert sent[-1]["role"] == "user", (
            f"background result entered as {sent[-1]['role']!r}; "
            "an assistant tail is a prefill and gets rejected with 400"
        )
        assert "Subagent" in str(sent[-1]["content"])
        assert out.channel == "feishu" and out.chat_id == "c1"

    @pytest.mark.asyncio
    async def test_holds_for_failed_tasks_too(self, tmp_path):
        """Failures are announced through the same path, so they broke too."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="feishu:c1")
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        seen: dict[str, list] = {}

        async def chat_with_retry(**kwargs):
            seen["messages"] = list(kwargs["messages"])
            return LLMResponse(content="reported the failure", tool_calls=[], usage={})

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None

        await loop._process_message(
            _subagent_msg("[Subagent 'audit' failed]\n\nResult:\nhit the tool limit")
        )

        assert seen["messages"][-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_survives_a_history_ending_in_assistant(self, tmp_path):
        """Mid-run checkpoints can leave history ending on an assistant turn."""
        from nanobot.session.manager import Session

        loop = _make_loop(tmp_path)
        session = Session(key="feishu:c1")
        session.messages = [
            {"role": "user", "content": "run the audit"},
            {"role": "assistant", "content": "starting"},
        ]
        loop.sessions.get_or_create = MagicMock(return_value=session)
        loop.sessions.save = MagicMock()
        loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()

        seen: dict[str, list] = {}

        async def chat_with_retry(**kwargs):
            seen["messages"] = list(kwargs["messages"])
            return LLMResponse(content="ok", tool_calls=[], usage={})

        loop.provider.chat_with_retry = chat_with_retry
        loop.tools = MagicMock()
        loop.tools.get_definitions.return_value = []
        loop.tools.get.return_value = None

        await loop._process_message(_subagent_msg())

        assert seen["messages"][-1]["role"] == "user"
