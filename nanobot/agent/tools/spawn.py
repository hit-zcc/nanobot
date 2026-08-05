"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Run a task in the background and report back when it is done. "
            "PREFER THIS over doing the work yourself whenever the task is likely to take "
            "more than a handful of tool calls -- for example: building or testing a project, "
            "tracing a bug across files or repositories, log/data investigation, bulk edits, "
            "research that needs several fetches, or anything the user described as a chunk of "
            "work rather than a question. Doing that inline blocks the conversation for minutes "
            "and floods it with intermediate steps; a subagent runs silently and delivers one "
            "report at the end. Answer directly only when you already know the answer or it is "
            "one or two quick lookups. "
            "Give the subagent the full context it needs -- it cannot see this conversation, "
            "and it cannot ask you follow-up questions. State the goal, the relevant paths, and "
            "what 'done' looks like. For deliverables or existing projects, inspect the "
            "workspace first and use a dedicated subdirectory when helpful."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
