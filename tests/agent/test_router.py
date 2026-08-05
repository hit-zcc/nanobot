import pytest

from nanobot.agent.router import AgentRouter
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import command_text
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


@pytest.mark.parametrize(
    ("content", "metadata", "expected"),
    [
        ("/stop", None, "/stop"),
        ("  /stop  ", None, "/stop"),
        # Group channels leave the "@bot" text in the body.
        ("@wuw /stop", {"mentions": [{"name": "wuw"}]}, "/stop"),
        # Fallback path: no mention metadata from the channel.
        ("@wuw /stop", None, "/stop"),
        # Display names may contain spaces, so the generic regex is not enough.
        ("@zhang san /team foo", {"mentions": [{"name": "zhang san"}]}, "/team foo"),
        ("@wuw @bob /status", {"mentions": [{"name": "wuw"}, {"name": "bob"}]}, "/status"),
        ("@wuw", {"mentions": [{"name": "wuw"}]}, ""),
        # Non-commands are only normalized for matching, never for the model.
        ("@wuw hello", {"mentions": [{"name": "wuw"}]}, "hello"),
        ("mail me at a@b.com", None, "mail me at a@b.com"),
        (None, None, ""),
    ],
)
def test_command_text_strips_leading_mentions(content, metadata, expected) -> None:
    assert command_text(content, metadata) == expected


@pytest.mark.asyncio
async def test_accept_dispatches_priority_command_mentioned_in_group() -> None:
    """"@bot /stop" in a group must hit the priority path, not the model."""
    seen: list[str] = []

    class PriorityCommands:
        def is_priority(self, raw: str) -> bool:
            return raw == "/stop"

        async def dispatch_priority(self, ctx):
            seen.append(ctx.raw)
            return None

    agent = FakeAgent()
    agent.commands = PriorityCommands()
    router = AgentRouter(MessageBus(), {"jarvis": agent}, [], "jarvis")

    await router._accept(agent, InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="oc_group",
        content="@wuw /stop",
        metadata={"mentions": [{"name": "wuw"}], "chat_type": "group"},
    ))

    assert seen == ["/stop"]
    assert agent.buffered == []
    assert agent.spawned == []
