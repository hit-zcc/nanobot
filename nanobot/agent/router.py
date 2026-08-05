"""Route inbound messages to independently configured agents."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import BindingConfig


class AgentRouter:
    """Consume the shared inbound bus and dispatch each message by binding."""

    def __init__(
        self,
        bus: MessageBus,
        agents: dict[str, AgentLoop],
        bindings: list[BindingConfig],
        default_agent_id: str,
    ) -> None:
        if default_agent_id not in agents:
            raise ValueError(f"Unknown default agent: {default_agent_id}")
        unknown = sorted({binding.agent_id for binding in bindings} - agents.keys())
        if unknown:
            raise ValueError(f"Bindings reference unknown agents: {', '.join(unknown)}")
        self.bus = bus
        self.agents = agents
        self.bindings = bindings
        self.default_agent_id = default_agent_id
        self._running = False

    def resolve(self, channel: str, chat_id: str) -> AgentLoop:
        """Resolve the first matching binding, then fall back to the default."""
        for binding in self.bindings:
            if binding.channel is not None and binding.channel != channel:
                continue
            if binding.chat_id is not None and binding.chat_id != chat_id:
                continue
            return self.agents[binding.agent_id]
        return self.agents[self.default_agent_id]

    async def _accept(self, agent: AgentLoop, msg: InboundMessage) -> None:
        # Delegate rather than re-implement: admission rules (priority commands,
        # mid-turn interjections, burst coalescing) must not drift between the
        # single-agent loop and this router.
        await agent.accept_inbound(msg)

    async def run(self) -> None:
        """Connect all agents and route messages until stopped."""
        self._running = True
        for agent_id, agent in self.agents.items():
            logger.info("Connecting MCP for agent '{}'...", agent_id)
            await agent._connect_mcp()
            agent._running = True

        logger.info(
            "Agent router started with {} agent(s): {}",
            len(self.agents),
            ", ".join(self.agents),
        )

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as exc:
                logger.warning("Error consuming inbound message: {}, continuing...", exc)
                continue

            agent = self.resolve(msg.channel, msg.chat_id)
            logger.debug("Routed {}:{} to agent '{}'", msg.channel, msg.chat_id, self._agent_id(agent))
            await self._accept(agent, msg)

    def _agent_id(self, target: AgentLoop) -> str:
        return next(
            (agent_id for agent_id, agent in self.agents.items() if agent is target),
            self.default_agent_id,
        )

    def stop(self) -> None:
        self._running = False
        for agent in self.agents.values():
            agent.stop()

    async def close_mcp(self) -> None:
        for agent in self.agents.values():
            await agent.close_mcp()
