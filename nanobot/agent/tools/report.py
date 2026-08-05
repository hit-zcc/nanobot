"""Mid-task reporting channel for subagents."""

from typing import TYPE_CHECKING, Any, Callable

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus


class ReportTool(Tool):
    """The one way a running subagent may interrupt the user.

    Subagents otherwise have no outbound channel: they run silently and the
    user only sees the final report.  That is deliberate -- step-by-step
    narration ("first I'll X, now Y, that didn't work") is noise, and the user
    can ask for status at any time because the manager keeps live state.  So
    this tool exists strictly for the cases where waiting until the end would
    be worse than interrupting, and the description below is written to make
    the model feel that bar rather than treat it as a progress log.
    """

    def __init__(
        self,
        bus: "MessageBus",
        channel: str,
        chat_id: str,
        label: str,
        on_report: Callable[[str], None] | None = None,
    ):
        self._bus = bus
        self._channel = channel
        self._chat_id = chat_id
        self._label = label
        self._on_report = on_report

    @property
    def name(self) -> str:
        return "report"

    @property
    def description(self) -> str:
        return (
            "Interrupt the user with ONE short message while the task is still running. "
            "This pings their phone, so the bar is high. Use it ONLY when: "
            "(a) you are blocked and cannot continue without a decision from them; "
            "(b) you found something that changes what the task should be, or that they "
            "would want to act on before you finish; or "
            "(c) you are about to do something risky or irreversible and they should know first. "
            "Do NOT use it to narrate steps, to say what you are about to do, to report that a "
            "step succeeded, or to say you are still working -- they get a full report when the "
            "task ends, and they can ask for status whenever they want. "
            "A task that finishes normally should typically call this zero times. "
            "When unsure, do not call it; put it in the final report instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "What the user needs to know, in their language. Lead with the point. "
                        "If you are blocked, state the decision you need and the options."
                    ),
                },
            },
            "required": ["message"],
        }

    async def execute(self, message: str, **kwargs: Any) -> str:
        text = (message or "").strip()
        if not text:
            return "Error: report message is empty."

        from nanobot.bus.events import OutboundMessage

        if self._on_report:
            self._on_report(text)
        await self._bus.publish_outbound(OutboundMessage(
            channel=self._channel,
            chat_id=self._chat_id,
            content=f"📌 后台任务「{self._label}」：{text}",
            # Not "_progress": this must survive the progress switch being off,
            # which is exactly the point of having a high bar for calling it.
            metadata={"_important_notice": True},
        ))
        return "Reported to the user. Continue the task; do not repeat this in the next few steps."
