import pytest

from nanobot.agent.router import AgentRouter
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import BindingConfig


class FakeCommands:
    def is_priority(self, _raw: str) -> bool:
        return False


class FakeAgent:
    def __init__(self) -> None:
        self.commands = FakeCommands()
        self._coalesce_window = 3.0
        self.buffered: list[InboundMessage] = []
        self.spawned: list[InboundMessage] = []

    def _buffer_inbound(self, msg: InboundMessage) -> None:
        self.buffered.append(msg)

    def _spawn_dispatch(self, msg: InboundMessage) -> None:
        self.spawned.append(msg)


def test_resolve_uses_channel_binding() -> None:
    life = FakeAgent()
    work = FakeAgent()
    router = AgentRouter(
        MessageBus(),
        {"jarvis": work, "life-buddy": life},
        [
            BindingConfig(agentId="life-buddy", channel="feishu"),
            BindingConfig(agentId="jarvis", channel="feishu.jarvis"),
        ],
        "jarvis",
    )

    assert router.resolve("feishu", "life-user") is life
    assert router.resolve("feishu.jarvis", "work-user") is work
    assert router.resolve("telegram", "other") is work


def test_rejects_binding_to_unknown_agent() -> None:
    with pytest.raises(ValueError, match="unknown agents"):
        AgentRouter(
            MessageBus(),
            {"jarvis": FakeAgent()},
            [BindingConfig(agentId="missing", channel="feishu")],
            "jarvis",
        )


@pytest.mark.asyncio
async def test_accept_preserves_agent_coalescing_and_system_dispatch() -> None:
    agent = FakeAgent()
    router = AgentRouter(MessageBus(), {"jarvis": agent}, [], "jarvis")
    user_msg = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="c1",
        content="hello",
    )
    system_msg = InboundMessage(
        channel="system",
        sender_id="system",
        chat_id="feishu:c1",
        content="done",
    )

    await router._accept(agent, user_msg)
    await router._accept(agent, system_msg)

    assert agent.buffered == [user_msg]
    assert agent.spawned == [system_msg]
