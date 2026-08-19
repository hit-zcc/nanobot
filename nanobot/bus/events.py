"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    session_key_override: str | None = None  # Optional override for thread-scoped sessions

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # 投递回执。发送方（如 message 工具）可以挂一个 Future 上来，
    # ChannelManager 投递完成后写入结果，让调用方知道**真的发出去了没有**。
    #
    # 为什么需要它：publish_outbound 只是 queue.put，立刻返回；真正的投递在
    # 另一个协程里，失败时 channel 层只 logger.error。没有这个字段，工具层
    # 无论如何都拿不到投递结果，只能谎报"已发送"。
    # 用 Any 而不是 asyncio.Future 是为了不让 dataclass 依赖 asyncio。
    delivery: Any = None


