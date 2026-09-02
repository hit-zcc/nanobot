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
            "Run a genuinely large task in the background and report back when it is done. "
            "Do NOT use this merely because a task needs several tool calls, because background "
            "execution is quieter, or because more investigation might be useful. Before spawning, "
            "write down the user's exact question and the minimum evidence that would answer it. "
            "If one focused code read, one targeted query, or at most 20 tool calls can answer that "
            "question, do the work yourself and stop as soon as that evidence is obtained. Do not "
            "silently expand a code-location question into log sampling, impact analysis, a full "
            "audit, or implementation planning; those are separate tasks and require a user request. "
            "Use this tool only when the requested deliverable itself requires more than 20 tool "
            "calls, crosses repositories or evidence systems, performs bulk edits, or runs a long "
            "build/evaluation. If you already know the relevant files and have the required context, "
            "that is evidence against spawning. "
            "Give the subagent the full context it needs -- it cannot see this conversation, and it "
            "cannot ask follow-up questions. State the exact question, explicit scope exclusions, "
            "the minimum sufficient evidence, and the stopping condition. For deliverables or "
            "existing projects, inspect the workspace first and use a dedicated subdirectory when "
            "helpful. Once spawned the task is fully delegated: do not also do it yourself in this "
            "turn (that duplicates the work) -- acknowledge briefly and stop, then wait for its report."
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
