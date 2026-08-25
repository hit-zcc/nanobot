"""Message tool for sending messages to users."""

import asyncio
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

# 等投递回执的上限。附件要先上传再发消息，比纯文本慢得多，
# 所以给得比较宽；超时只是"没等到回执"，不代表一定失败。
_DELIVERY_TIMEOUT_S = 60.0


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user, optionally with file attachments. "
            "This is the ONLY way to deliver files (images, documents, audio, video) to the user. "
            "Use the 'media' parameter with file paths to attach files. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        explicit_chat = bool(chat_id) and chat_id != self._default_chat_id
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        # 🔴 显式指定了别的 chat_id，就不能再沿用当前会话的 message_id。
        # 飞书 reply 接口按 **父消息所在会话** 投递，chat_id 被完全忽略：
        # 2026-08-19 实测，指定群 oc_6b68…（test 群）发附件，文件却落进了
        # 与父消息同源的 p2p 私聊 oc_a276…，而 API 全程返回成功。
        # 表现就是"文字看得到、附件不见了"——因为文字是后发的、走的 create。
        if explicit_chat:
            message_id = None

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        # 🔴 挂一个投递回执 Future。没有它，publish_outbound 只是入队就返回，
        # 工具永远返回"已发送"——哪怕消息根本没投出去（飞书 230002 等）。
        # 这是"假完成态"的机制来源：不是忘了检查，是回执本身在撒谎。
        try:
            delivery: asyncio.Future | None = asyncio.get_running_loop().create_future()
        except RuntimeError:
            delivery = None

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            },
            delivery=delivery,
        )

        media_info = f" with {len(media)} attachments" if media else ""
        try:
            await self._send_callback(msg)
        except Exception as e:
            return f"Error sending message: {str(e)}"

        if delivery is None:
            # 没有事件循环（同步测试等），退回旧行为并说清楚。
            self._mark_sent(channel, chat_id)
            return f"Message queued to {channel}:{chat_id}{media_info} (delivery unconfirmed)"

        try:
            ok, detail = await asyncio.wait_for(delivery, timeout=_DELIVERY_TIMEOUT_S)
        except asyncio.TimeoutError:
            return (
                f"⚠️ Delivery to {channel}:{chat_id}{media_info} did NOT confirm "
                f"within {_DELIVERY_TIMEOUT_S:.0f}s. Do not tell the user it was sent."
            )
        except Exception as e:
            return f"Error awaiting delivery to {channel}:{chat_id}: {e}"

        if ok:
            self._mark_sent(channel, chat_id)
            return f"Message delivered to {channel}:{chat_id}{media_info}"
        return (
            f"❌ Delivery FAILED to {channel}:{chat_id}{media_info}: {detail}. "
            f"The user did NOT receive this. Do not claim it was sent; "
            f"check the target id and report the failure."
        )

    def _mark_sent(self, channel: str, chat_id: str) -> None:
        """记下本轮已经主动发过消息（用于抑制重复的收尾回复）。"""
        if channel == self._default_channel and chat_id == self._default_chat_id:
            self._sent_in_turn = True
