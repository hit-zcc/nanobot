"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import html
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from pydantic import Field

import importlib.util

FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None


class _ThreadLocalLoopProxy:
    """Thread-local proxy for asyncio event loops.

    Replaces lark_oapi.ws.client's module-level ``loop`` variable so that
    multiple Feishu channel threads each get their own event loop without
    overwriting each other.
    """

    _local = threading.local()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._local.loop = loop

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        loop = getattr(self._local, "loop", None)
        if loop is None:
            raise RuntimeError("No event loop set for this thread")
        return loop

    def __getattr__(self, name: str):
        return getattr(self._get_loop(), name)


# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


def _extract_share_card_content(content_json: dict, msg_type: str) -> str:
    """Extract text representation from share cards and interactive messages."""
    parts = []

    if msg_type == "share_chat":
        parts.append(f"[shared chat: {content_json.get('chat_id', '')}]")
    elif msg_type == "share_user":
        parts.append(f"[shared user: {content_json.get('user_id', '')}]")
    elif msg_type == "interactive":
        parts.extend(_extract_interactive_content(content_json))
    elif msg_type == "share_calendar_event":
        parts.append(f"[shared calendar event: {content_json.get('event_key', '')}]")
    elif msg_type == "system":
        parts.append("[system message]")
    elif msg_type == "merge_forward":
        parts.append("[merged forward messages]")

    return "\n".join(parts) if parts else f"[{msg_type}]"


def _extract_interactive_content(content: dict) -> list[str]:
    """Recursively extract text and links from interactive card content."""
    parts = []

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []

    if not isinstance(content, dict):
        return parts

    if "title" in content:
        title = content["title"]
        if isinstance(title, dict):
            title_content = title.get("content", "") or title.get("text", "")
            if title_content:
                parts.append(f"title: {title_content}")
        elif isinstance(title, str):
            parts.append(f"title: {title}")

    for elements in content.get("elements", []) if isinstance(content.get("elements"), list) else []:
        for element in elements:
            parts.extend(_extract_element_content(element))

    card = content.get("card", {})
    if card:
        parts.extend(_extract_interactive_content(card))

    header = content.get("header", {})
    if header:
        header_title = header.get("title", {})
        if isinstance(header_title, dict):
            header_text = header_title.get("content", "") or header_title.get("text", "")
            if header_text:
                parts.append(f"title: {header_text}")

    return parts


def _extract_element_content(element: dict) -> list[str]:
    """Extract content from a single card element."""
    parts = []

    if not isinstance(element, dict):
        return parts

    tag = element.get("tag", "")

    if tag in ("markdown", "lark_md"):
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "div":
        text = element.get("text", {})
        if isinstance(text, dict):
            text_content = text.get("content", "") or text.get("text", "")
            if text_content:
                parts.append(text_content)
        elif isinstance(text, str):
            parts.append(text)
        for field in element.get("fields", []):
            if isinstance(field, dict):
                field_text = field.get("text", {})
                if isinstance(field_text, dict):
                    c = field_text.get("content", "")
                    if c:
                        parts.append(c)

    elif tag == "a":
        href = element.get("href", "")
        text = element.get("text", "")
        if href:
            parts.append(f"link: {href}")
        if text:
            parts.append(text)

    elif tag == "button":
        text = element.get("text", {})
        if isinstance(text, dict):
            c = text.get("content", "")
            if c:
                parts.append(c)
        url = element.get("url", "") or element.get("multi_url", {}).get("url", "")
        if url:
            parts.append(f"link: {url}")

    elif tag == "img":
        alt = element.get("alt", {})
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")

    elif tag == "note":
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    elif tag == "column_set":
        for col in element.get("columns", []):
            for ce in col.get("elements", []):
                parts.extend(_extract_element_content(ce))

    elif tag == "plain_text":
        content = element.get("content", "")
        if content:
            parts.append(content)

    else:
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    return parts


def _extract_post_content(content_json: dict) -> tuple[str, list[str]]:
    """Extract text and image keys from Feishu post (rich text) message.

    Handles three payload shapes:
    - Direct:    {"title": "...", "content": [[...]]}
    - Localized: {"zh_cn": {"title": "...", "content": [...]}}
    - Wrapped:   {"post": {"zh_cn": {"title": "...", "content": [...]}}}
    """

    def _parse_block(block: dict) -> tuple[str | None, list[str]]:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return None, []
        texts, images = [], []
        if title := block.get("title"):
            texts.append(title)
        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for el in row:
                if not isinstance(el, dict):
                    continue
                tag = el.get("tag")
                if tag in ("text", "a"):
                    texts.append(el.get("text", ""))
                elif tag == "at":
                    texts.append(f"@{el.get('user_name', 'user')}")
                elif tag == "code_block":
                    lang = el.get("language", "")
                    code_text = el.get("text", "")
                    texts.append(f"\n```{lang}\n{code_text}\n```\n")
                elif tag == "img" and (key := el.get("image_key")):
                    images.append(key)
        return (" ".join(texts).strip() or None), images

    # Unwrap optional {"post": ...} envelope
    root = content_json
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return "", []

    # Direct format
    if "content" in root:
        text, imgs = _parse_block(root)
        if text or imgs:
            return text or "", imgs

    # Localized: prefer known locales, then fall back to any dict child
    for key in ("zh_cn", "en_us", "ja_jp"):
        if key in root:
            text, imgs = _parse_block(root[key])
            if text or imgs:
                return text or "", imgs
    for val in root.values():
        if isinstance(val, dict):
            text, imgs = _parse_block(val)
            if text or imgs:
                return text or "", imgs

    return "", []


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message content.

    Legacy wrapper for _extract_post_content, returns only text.
    """
    text, _ = _extract_post_content(content_json)
    return text


def _resolve_text_mentions(
    text: str,
    mentions: list[Any] | None,
) -> tuple[str, list[dict[str, str]]]:
    """Replace Feishu ``@_user_N`` placeholders with readable display names."""
    resolved: list[dict[str, str]] = []
    replacements: list[tuple[str, str]] = []

    for mention in mentions or []:
        key = str(getattr(mention, "key", None) or "")
        mention_id = getattr(mention, "id", None)
        name = str(getattr(mention, "name", None) or "").lstrip("@")
        open_id = str(getattr(mention_id, "open_id", None) or "")
        user_id = str(getattr(mention_id, "user_id", None) or "")
        union_id = str(getattr(mention_id, "union_id", None) or "")

        label = name or open_id or user_id or union_id
        if key and label:
            replacements.append((key, f"@{label}"))
        resolved.append({
            key_name: value
            for key_name, value in {
                "key": key,
                "name": name,
                "open_id": open_id,
                "user_id": user_id,
                "union_id": union_id,
            }.items()
            if value
        })

    # Replace longer keys first so @_user_1 cannot partially alter @_user_10.
    for key, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(key, label)
    return text, resolved


def _sanitize_unresolved_mentions(text: str) -> str:
    """Prevent internal Feishu mention placeholders from reaching users."""
    return re.sub(r"@_user_\d+\b", "@用户", text)


_LEADING_DISPLAY_MENTION_RE = re.compile(r"^\s*@([^\s,，:：]+)[\s,，:：]*")


def _split_leading_display_mention(text: str) -> tuple[str, str]:
    """Extract an LLM-rendered leading ``@name`` for use as an at-label."""
    match = _LEADING_DISPLAY_MENTION_RE.match(text)
    if not match:
        return "用户", text.lstrip()
    return match.group(1), text[match.end():].lstrip()


class FeishuConfig(Base):
    """Feishu/Lark channel configuration using WebSocket long connection."""

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    react_emoji: str = "THUMBSUP"
    group_policy: Literal["open", "mention"] = "mention"
    reply_to_message: bool = False  # If True, bot replies quote the user's original message
    streaming: bool = True
    progress_notify_interval_s: int = Field(
        default=120,
        ge=0,
        le=3600,
        description=(
            "Create a fresh Feishu message this often while a tool remains active. "
            "Intermediate heartbeats still update in place; 0 disables fresh notices."
        ),
    )
    group_context_messages: int = Field(
        default=20,
        ge=0,
        le=200,
        description=(
            "How many recent un-addressed group messages to keep per chat and hand "
            "to the agent when it is next mentioned. 0 disables the buffer."
        ),
    )


_STREAM_ELEMENT_ID = "streaming_md"


@dataclass
class _FeishuStreamBuf:
    """Per-chat streaming accumulator using CardKit streaming API."""
    text: str = ""
    mention_target: str | None = None
    mention_pending: bool = False
    card_id: str | None = None
    sequence: int = 0
    last_edit: float = 0.0


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.

    Uses WebSocket to receive events - no public IP or webhook required.

    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """

    name = "feishu"
    display_name = "Feishu"

    _STREAM_EDIT_INTERVAL = 0.5  # throttle between CardKit streaming updates
    _STREAM_LEADING_MENTION_RE = re.compile(
        r"^\s*@([^\s,，:：]+)[\s,，:：]+"
    )

    @classmethod
    def _finalize_stream_mention(
        cls,
        buf: _FeishuStreamBuf,
        *,
        force: bool = False,
    ) -> bool:
        """Wait for a complete streamed ``@name`` token before removing it."""
        if not buf.mention_pending or not buf.mention_target:
            return True

        stripped = buf.text.lstrip()
        if not stripped:
            return False
        if stripped.startswith("@"):
            match = cls._STREAM_LEADING_MENTION_RE.match(buf.text)
            if not match and not force:
                return False
            if match:
                remainder = buf.text[match.end():].lstrip()
            else:
                _, remainder = _split_leading_display_mention(buf.text)
        else:
            remainder = stripped

        buf.text = f'<at id="{buf.mention_target}"></at> {remainder}'
        buf.mention_pending = False
        return True

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return FeishuConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = FeishuConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._processed_card_actions: OrderedDict[str, None] = OrderedDict()
        self._reply_targets: OrderedDict[str, str] = OrderedDict()
        # Card-click session routing.  A card.action.trigger callback only
        # reports ``open_chat_id`` (always ``oc_…``), but p2p sessions are keyed
        # by the user's ``ou_…`` open_id.  Without these maps a click on a card
        # sent in a private chat lands in a different agent session than the one
        # that produced the card.  Keyed by bot message_id and by open_chat_id.
        self._outbound_routes: OrderedDict[str, str] = OrderedDict()
        self._chat_routes: OrderedDict[str, str] = OrderedDict()
        # Inbound message ids whose reply already carried an @mention, so a turn
        # that spans several outbound messages only notifies the sender once.
        self._mentioned_turns: OrderedDict[str, None] = OrderedDict()
        # Recent group messages that did not address the bot, per chat.  They are
        # not answered, but the next mention hands them over as context — people
        # routinely post the question first and @ the bot in a follow-up message.
        self._group_context: dict[str, deque[tuple[str, int, str]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream_bufs: dict[str, _FeishuStreamBuf] = {}
        self._tool_progress_messages: dict[str, str] = {}
        self._tool_progress_last_notified: dict[str, float] = {}
        # Staleness guard: drop messages created before this channel started
        # (restart replays) or older than the threshold (in-process redelivery).
        # Feishu's at-least-once delivery can re-push old events with a fresh
        # message_id, which message_id dedup alone cannot catch.
        # NANOBOT_FEISHU_STALE_SECONDS=0 disables the age check.
        self._start_time_ms: int = int(time.time() * 1000)
        self._stale_threshold_ms: int = int(
            float(os.environ.get("NANOBOT_FEISHU_STALE_SECONDS", "120")) * 1000
        )

    @staticmethod
    def _remember(cache: OrderedDict[str, Any], key: str, value: Any, limit: int = 1000) -> None:
        """Insert into a bounded FIFO cache, evicting the oldest entries."""
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _register_optional_event(builder: Any, method_name: str, handler: Any) -> Any:
        """Register an event handler only when the SDK supports it."""
        method = getattr(builder, method_name, None)
        return method(handler) if callable(method) else builder

    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return

        import lark_oapi as lark
        self._running = True
        self._loop = asyncio.get_running_loop()

        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        builder = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        )
        builder = self._register_optional_event(
            builder, "register_p2_im_message_reaction_created_v1", self._on_reaction_created
        )
        builder = self._register_optional_event(
            builder, "register_p2_im_message_message_read_v1", self._on_message_read
        )
        builder = self._register_optional_event(
            builder,
            "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
            self._on_bot_p2p_chat_entered,
        )
        register_card_action = getattr(builder, "register_p2_card_action_trigger", None)
        if callable(register_card_action):
            builder = register_card_action(self._on_card_action_sync)
        else:
            logger.warning(
                "Installed lark-oapi SDK does not support card.action.trigger callbacks"
            )
        event_handler = builder.build()

        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )

        # Start WebSocket client in a separate thread with reconnect loop.
        # A dedicated event loop is created for this thread so that lark_oapi's
        # module-level `loop = asyncio.get_event_loop()` picks up an idle loop
        # instead of the already-running main asyncio loop, which would cause
        # "This event loop is already running" errors.
        #
        # When multiple Feishu channels run in the same process, each thread
        # needs its own event loop.  We install a thread-local proxy object as
        # the module-level ``loop`` so that every thread transparently gets its
        # own loop without overwriting the others.
        def run_ws():
            import time
            import lark_oapi.ws.client as _lark_ws_client

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)

            # Install a thread-local proxy (once) so multiple Feishu channels
            # in the same process don't overwrite each other's event loop.
            if not isinstance(_lark_ws_client.loop, _ThreadLocalLoopProxy):
                _lark_ws_client.loop = _ThreadLocalLoopProxy()
            _lark_ws_client.loop.set_loop(ws_loop)

            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as e:
                        logger.warning("Feishu WebSocket error: {}", e)
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """
        Stop the Feishu bot.

        Notice: lark.ws.Client does not expose stop method， simply exiting the program will close the client.

        Reference: https://github.com/larksuite/oapi-sdk-python/blob/v2_main/lark_oapi/ws/client.py#L86
        """
        self._running = False
        logger.info("Feishu bot stopped")

    def _is_bot_mentioned(self, message: Any) -> bool:
        """Check if the bot is @mentioned in the message."""
        raw_content = message.content or ""
        if "@_all" in raw_content:
            return True

        for mention in getattr(message, "mentions", None) or []:
            mid = getattr(mention, "id", None)
            if not mid:
                continue
            # Bot mentions have no user_id (None or "") but a valid open_id
            if not getattr(mid, "user_id", None) and (getattr(mid, "open_id", None) or "").startswith("ou_"):
                return True
        return False

    def _is_group_message_for_bot(self, message: Any) -> bool:
        """Allow group messages when policy is open or bot is @mentioned."""
        if self.config.group_policy == "open":
            return True
        return self._is_bot_mentioned(message)

    def _buffer_group_context(self, message: Any, sender_id: str) -> None:
        """Remember an un-addressed group message as context for the next mention.

        Only a compact text rendering is kept — no media is downloaded and no
        reaction is added, because the bot is not answering this message.
        """
        limit = self.config.group_context_messages
        if limit <= 0:
            return
        chat_id = getattr(message, "chat_id", None)
        if not chat_id:
            return

        msg_type = getattr(message, "message_type", "")
        try:
            content_json = json.loads(message.content) if message.content else {}
        except (json.JSONDecodeError, TypeError):
            content_json = {}

        if msg_type == "text":
            text, _ = _resolve_text_mentions(
                content_json.get("text", ""), getattr(message, "mentions", None)
            )
        elif msg_type == "post":
            text, _ = _extract_post_content(content_json)
        elif msg_type in ("share_chat", "share_user", "interactive"):
            text = _extract_share_card_content(content_json, msg_type)
        else:
            text = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

        text = " ".join((text or "").split())
        if not text:
            return
        if len(text) > 500:
            text = text[:500] + "…"

        buf = self._group_context.get(chat_id)
        if buf is None or buf.maxlen != limit:
            buf = deque(buf or (), maxlen=limit)
            self._group_context[chat_id] = buf
        created_ms = int(getattr(message, "create_time", 0) or 0)
        buf.append((sender_id, created_ms, text))

    def _drain_group_context(self, chat_id: str, current_sender: str) -> str:
        """Render and clear the buffered group messages for *chat_id*."""
        buf = self._group_context.pop(chat_id, None)
        if not buf:
            return ""
        lines = []
        for sender_id, created_ms, text in buf:
            who = "same sender" if sender_id == current_sender else sender_id
            when = (
                time.strftime("%H:%M", time.localtime(created_ms / 1000))
                if created_ms else "?"
            )
            lines.append(f"[{when}] ({who}) {text}")
        return (
            "[Recent group messages that did not address this bot — context only, "
            "already seen by everyone in the chat]\n"
            + "\n".join(lines)
        )

    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        from lark_oapi.api.im.v1 import CreateMessageReactionRequest, CreateMessageReactionRequestBody, Emoji
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()

            response = self._client.im.v1.message_reaction.create(request)

            if not response.success():
                logger.warning("Failed to add reaction: code={}, msg={}", response.code, response.msg)
            else:
                logger.debug("Added {} reaction to message {}", emoji_type, message_id)
        except Exception as e:
            logger.warning("Error adding reaction: {}", e)

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).

        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client:
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)

    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

    # Markdown formatting patterns that should be stripped from plain-text
    # surfaces like table cells and heading text.
    _MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _MD_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__")
    _MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    _MD_STRIKE_RE = re.compile(r"~~(.+?)~~")

    @classmethod
    def _strip_md_formatting(cls, text: str) -> str:
        """Strip markdown formatting markers from text for plain display.

        Feishu table cells do not support markdown rendering, so we remove
        the formatting markers to keep the text readable.
        """
        # Remove bold markers
        text = cls._MD_BOLD_RE.sub(r"\1", text)
        text = cls._MD_BOLD_UNDERSCORE_RE.sub(r"\1", text)
        # Remove italic markers
        text = cls._MD_ITALIC_RE.sub(r"\1", text)
        # Remove strikethrough markers
        text = cls._MD_STRIKE_RE.sub(r"\1", text)
        return text

    @classmethod
    def _parse_md_table(cls, table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [_line.strip() for _line in table_text.strip().split("\n") if _line.strip()]
        if len(lines) < 3:
            return None
        def split(_line: str) -> list[str]:
            return [c.strip() for c in _line.strip("|").split("|")]
        headers = [cls._strip_md_formatting(h) for h in split(lines[0])]
        rows = [[cls._strip_md_formatting(c) for c in split(_line)] for _line in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    @staticmethod
    def _split_elements_by_table_limit(elements: list[dict], max_tables: int = 1) -> list[list[dict]]:
        """Split card elements into groups with at most *max_tables* table elements each.

        Feishu cards have a hard limit of one table per card (API error 11310).
        When the rendered content contains multiple markdown tables each table is
        placed in a separate card message so every table reaches the user.
        """
        if not elements:
            return [[]]
        groups: list[list[dict]] = []
        current: list[dict] = []
        table_count = 0
        for el in elements:
            if el.get("tag") == "table":
                if table_count >= max_tables:
                    if current:
                        groups.append(current)
                    current = []
                    table_count = 0
                current.append(el)
                table_count += 1
            else:
                current.append(el)
        if current:
            groups.append(current)
        return groups or [[]]

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(m.group(1), f"\x00CODE{len(code_blocks)-1}\x00", 1)

        elements = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            text = self._strip_md_formatting(m.group(2).strip())
            display_text = f"**{text}**" if text else ""
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": display_text,
                },
            })
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    # ── Smart format detection ──────────────────────────────────────────
    # Patterns that indicate "complex" markdown needing card rendering
    _COMPLEX_MD_RE = re.compile(
        r"```"                        # fenced code block
        r"|^\|.+\|.*\n\s*\|[-:\s|]+\|"  # markdown table (header + separator)
        r"|^#{1,6}\s+"                # headings
        , re.MULTILINE,
    )

    # Simple markdown patterns (bold, italic, strikethrough)
    _SIMPLE_MD_RE = re.compile(
        r"\*\*.+?\*\*"               # **bold**
        r"|__.+?__"                   # __bold__
        r"|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"  # *italic* (single *)
        r"|~~.+?~~"                   # ~~strikethrough~~
        , re.DOTALL,
    )

    # Markdown link: [text](url)
    _MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")

    # Unordered list items
    _LIST_RE = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)

    # Ordered list items
    _OLIST_RE = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)

    # Max length for plain text format
    _TEXT_MAX_LEN = 200

    # Max length for post (rich text) format; beyond this, use card
    _POST_MAX_LEN = 2000

    @classmethod
    def _detect_msg_format(cls, content: str) -> str:
        """Determine the optimal Feishu message format for *content*.

        Returns one of:
        - ``"text"``        – plain text, short and no markdown
        - ``"post"``        – rich text (links only, moderate length)
        - ``"interactive"`` – card with full markdown rendering
        """
        stripped = content.strip()

        # Complex markdown (code blocks, tables, headings) → always card
        if cls._COMPLEX_MD_RE.search(stripped):
            return "interactive"

        # Long content → card (better readability with card layout)
        if len(stripped) > cls._POST_MAX_LEN:
            return "interactive"

        # Has bold/italic/strikethrough → card (post format can't render these)
        if cls._SIMPLE_MD_RE.search(stripped):
            return "interactive"

        # Has list items → card (post format can't render list bullets well)
        if cls._LIST_RE.search(stripped) or cls._OLIST_RE.search(stripped):
            return "interactive"

        # Has links → post format (supports <a> tags)
        if cls._MD_LINK_RE.search(stripped):
            return "post"

        # Short plain text → text format
        if len(stripped) <= cls._TEXT_MAX_LEN:
            return "text"

        # Medium plain text without any formatting → post format
        return "post"

    def _outbound_mention_target(self, metadata: dict[str, Any] | None) -> str | None:
        """Return the real Feishu open_id to mention for a group reply.

        Only the *first* outbound message of a turn mentions the sender.  A turn
        often spans several messages (media, chunked cards, follow-up sends), and
        repeating the ``<at>`` re-notifies the group for what is one reply.
        """
        meta = metadata or {}
        if meta.get("_progress") or meta.get("_tool_hint"):
            return None
        if meta.get("sender_type") == "bot":
            return None

        target = ""
        if meta.get("chat_type") == "group":
            target = str(meta.get("sender_open_id") or "")
        if not target and meta.get("message_id"):
            target = self._reply_targets.get(str(meta["message_id"]), "")
        if not re.fullmatch(r"ou_[A-Za-z0-9_-]+", target):
            return None

        turn_key = str(meta.get("message_id") or "")
        if turn_key:
            if turn_key in self._mentioned_turns:
                return None
            self._remember(self._mentioned_turns, turn_key, None)
        return target

    @classmethod
    def _markdown_to_post(
        cls,
        content: str,
        *,
        mention_target: str | None = None,
        mention_label: str = "用户",
    ) -> str:
        """Convert markdown content to Feishu post message JSON.

        Handles links ``[text](url)`` as ``a`` tags; everything else as ``text`` tags.
        Each line becomes a paragraph (row) in the post body.
        """
        lines = content.strip().split("\n")
        paragraphs: list[list[dict]] = []

        for line in lines:
            elements: list[dict] = []
            last_end = 0

            for m in cls._MD_LINK_RE.finditer(line):
                # Text before this link
                before = line[last_end:m.start()]
                if before:
                    elements.append({"tag": "text", "text": before})
                elements.append({
                    "tag": "a",
                    "text": m.group(1),
                    "href": m.group(2),
                })
                last_end = m.end()

            # Remaining text after last link
            remaining = line[last_end:]
            if remaining:
                elements.append({"tag": "text", "text": remaining})

            # Empty line → empty paragraph for spacing
            if not elements:
                elements.append({"tag": "text", "text": ""})

            paragraphs.append(elements)

        if mention_target:
            paragraphs[0][0:0] = [
                {
                    "tag": "at",
                    "user_id": mention_target,
                    "user_name": mention_label or "用户",
                },
                {"tag": "text", "text": " "},
            ]

        post_body = {
            "zh_cn": {
                "content": paragraphs,
            }
        }
        return json.dumps(post_body, ensure_ascii=False)

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}
    _AUDIO_EXTS = {".opus"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi"}
    _FILE_TYPE_MAP = {
        ".opus": "opus", ".mp4": "mp4", ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
        ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
    }

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody
        try:
            with open(file_path, "rb") as f:
                request = CreateImageRequest.builder() \
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    ).build()
                response = self._client.im.v1.image.create(request)
                if response.success():
                    image_key = response.data.image_key
                    logger.debug("Uploaded image {}: {}", os.path.basename(file_path), image_key)
                    return image_key
                else:
                    logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading image {}: {}", file_path, e)
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody
        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                request = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(f)
                        .build()
                    ).build()
                response = self._client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    logger.debug("Uploaded file {}: {}", file_name, file_key)
                    return file_key
                else:
                    logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading file {}: {}", file_path, e)
            return None

    def _download_image_sync(self, message_id: str, image_key: str) -> tuple[bytes | None, str | None]:
        """Download an image from Feishu message by message_id and image_key."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest
        try:
            request = GetMessageResourceRequest.builder() \
                .message_id(message_id) \
                .file_key(image_key) \
                .type("image") \
                .build()
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                # GetMessageResourceRequest returns BytesIO, need to read bytes
                if hasattr(file_data, 'read'):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.error("Failed to download image: code={}, msg={}", response.code, response.msg)
                return None, None
        except Exception as e:
            logger.error("Error downloading image {}: {}", image_key, e)
            return None, None

    def _download_file_sync(
        self, message_id: str, file_key: str, resource_type: str = "file"
    ) -> tuple[bytes | None, str | None]:
        """Download a file/audio/media from a Feishu message by message_id and file_key."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        # Feishu API only accepts 'image' or 'file' as type parameter
        # Convert 'audio' to 'file' for API compatibility
        if resource_type == "audio":
            resource_type = "file"

        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                if hasattr(file_data, "read"):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.error("Failed to download {}: code={}, msg={}", resource_type, response.code, response.msg)
                return None, None
        except Exception:
            logger.exception("Error downloading {} {}", resource_type, file_key)
            return None, None

    async def _download_and_save_media(
        self,
        msg_type: str,
        content_json: dict,
        message_id: str | None = None
    ) -> tuple[str | None, str]:
        """
        Download media from Feishu and save to local disk.

        Returns:
            (file_path, content_text) - file_path is None if download failed
        """
        loop = asyncio.get_running_loop()
        media_dir = get_media_dir("feishu")

        data, filename = None, None

        if msg_type == "image":
            image_key = content_json.get("image_key")
            if image_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_image_sync, message_id, image_key
                )
                if not filename:
                    filename = f"{image_key[:16]}.jpg"

        elif msg_type in ("audio", "file", "media"):
            file_key = content_json.get("file_key")
            if file_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_file_sync, message_id, file_key, msg_type
                )
                if not filename:
                    filename = file_key[:16]
                if msg_type == "audio" and not filename.endswith(".opus"):
                    filename = f"{filename}.opus"

        if data and filename:
            file_path = media_dir / filename
            file_path.write_bytes(data)
            logger.debug("Downloaded {} to {}", msg_type, file_path)
            return str(file_path), f"[{msg_type}: {filename}]"

        return None, f"[{msg_type}: download failed]"

    _REPLY_CONTEXT_MAX_LEN = 10_000

    def _get_chat_mode_sync(self, chat_id: str) -> str | None:
        """Fetch a chat's mode ("p2p" or "group"), or None when unavailable."""
        from lark_oapi.api.im.v1 import GetChatRequest
        try:
            request = GetChatRequest.builder().chat_id(chat_id).build()
            response = self._client.im.v1.chat.get(request)
            if not response.success():
                logger.debug(
                    "Feishu: could not fetch chat {}: code={}, msg={}",
                    chat_id, response.code, response.msg,
                )
                return None
            return getattr(response.data, "chat_mode", None)
        except Exception as e:
            logger.debug("Feishu: error fetching chat {}: {}", chat_id, e)
            return None

    def _get_message_content_sync(self, message_id: str) -> str | None:
        """Fetch the text content of a Feishu message by ID (synchronous).

        Returns a "[Reply to: ...]" context string, or None on failure.
        """
        from lark_oapi.api.im.v1 import GetMessageRequest
        try:
            request = GetMessageRequest.builder().message_id(message_id).build()
            response = self._client.im.v1.message.get(request)
            if not response.success():
                logger.debug(
                    "Feishu: could not fetch parent message {}: code={}, msg={}",
                    message_id, response.code, response.msg,
                )
                return None
            items = getattr(response.data, "items", None)
            if not items:
                return None
            msg_obj = items[0]
            raw_content = getattr(msg_obj, "body", None)
            raw_content = getattr(raw_content, "content", None) if raw_content else None
            if not raw_content:
                return None
            try:
                content_json = json.loads(raw_content)
            except (json.JSONDecodeError, TypeError):
                return None
            msg_type = getattr(msg_obj, "msg_type", "")
            if msg_type == "text":
                text = content_json.get("text", "").strip()
                text, _ = _resolve_text_mentions(
                    text,
                    getattr(msg_obj, "mentions", None),
                )
            elif msg_type == "post":
                text, _ = _extract_post_content(content_json)
                text = text.strip()
            else:
                text = ""
            if not text:
                return None
            if len(text) > self._REPLY_CONTEXT_MAX_LEN:
                text = text[: self._REPLY_CONTEXT_MAX_LEN] + "..."
            return f"[Reply to: {text}]"
        except Exception as e:
            logger.debug("Feishu: error fetching parent message {}: {}", message_id, e)
            return None

    def _reply_message_with_id_sync(
        self, parent_message_id: str, msg_type: str, content: str,
    ) -> str | None:
        """Reply to a Feishu message and return the new message ID."""
        from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
        try:
            request = ReplyMessageRequest.builder() \
                .message_id(parent_message_id) \
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.reply(request)
            if not response.success():
                logger.error(
                    "Failed to reply to Feishu message {}: code={}, msg={}, log_id={}",
                    parent_message_id, response.code, response.msg, response.get_log_id()
                )
                return None
            msg_id = getattr(response.data, "message_id", None)
            logger.debug("Feishu reply sent to message {}: {}", parent_message_id, msg_id)
            return msg_id
        except Exception as e:
            logger.error("Error replying to Feishu message {}: {}", parent_message_id, e)
            return None

    def _reply_message_sync(self, parent_message_id: str, msg_type: str, content: str) -> bool:
        """Reply to an existing Feishu message using the Reply API (synchronous)."""
        return bool(self._reply_message_with_id_sync(parent_message_id, msg_type, content))

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        """Send a single message and return the message_id on success."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu {} message: code={}, msg={}, log_id={}",
                    msg_type, response.code, response.msg, response.get_log_id()
                )
                return None
            msg_id = getattr(response.data, "message_id", None)
            logger.debug("Feishu {} message sent to {}: {}", msg_type, receive_id, msg_id)
            return msg_id
        except Exception as e:
            logger.error("Error sending Feishu {} message: {}", msg_type, e)
            return None

    def _update_message_sync(self, message_id: str, content: str) -> bool:
        """Replace the text content of an existing Feishu message."""
        from lark_oapi.api.im.v1 import UpdateMessageRequest, UpdateMessageRequestBody

        try:
            body = json.dumps({"text": content}, ensure_ascii=False)
            request = UpdateMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    UpdateMessageRequestBody.builder()
                    .msg_type("text")
                    .content(body)
                    .build()
                ).build()
            response = self._client.im.v1.message.update(request)
            if not response.success():
                logger.warning(
                    "Failed to update Feishu progress message {}: code={}, msg={}",
                    message_id, response.code, response.msg,
                )
                return False
            return True
        except Exception as e:
            logger.warning("Error updating Feishu progress message {}: {}", message_id, e)
            return False

    async def _send_tool_progress(
        self,
        msg: OutboundMessage,
        receive_id_type: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Create once, then update one progress message for an agent turn."""
        progress_id = str(msg.metadata.get("_tool_progress_id") or "")
        if not progress_id:
            return

        existing_message_id = self._tool_progress_messages.get(progress_id)
        done = bool(msg.metadata.get("_tool_progress_done"))
        if done:
            if not existing_message_id:
                return
            self._tool_progress_messages.pop(progress_id, None)
            self._tool_progress_last_notified.pop(progress_id, None)
            await loop.run_in_executor(
                None, self._update_message_sync, existing_message_id, msg.content,
            )
            return

        now = time.monotonic()
        if existing_message_id:
            notify_interval = self.config.progress_notify_interval_s
            last_notified = self._tool_progress_last_notified.get(progress_id, now)
            if notify_interval > 0 and now - last_notified >= notify_interval:
                text_body = json.dumps({"text": msg.content.strip()}, ensure_ascii=False)
                parent_message_id = None
                if msg.metadata.get("thread_id"):
                    parent_message_id = (
                        msg.metadata.get("root_id") or msg.metadata.get("message_id")
                    )
                if parent_message_id:
                    sent_message_id = await loop.run_in_executor(
                        None,
                        self._reply_message_with_id_sync,
                        parent_message_id,
                        "text",
                        text_body,
                    )
                else:
                    sent_message_id = await loop.run_in_executor(
                        None,
                        self._send_message_sync,
                        receive_id_type,
                        msg.chat_id,
                        "text",
                        text_body,
                    )
                if sent_message_id:
                    self._tool_progress_messages[progress_id] = sent_message_id
                    self._tool_progress_last_notified[progress_id] = now
                    return
            await loop.run_in_executor(
                None, self._update_message_sync, existing_message_id, msg.content,
            )
            return

        text_body = json.dumps({"text": msg.content.strip()}, ensure_ascii=False)
        parent_message_id = None
        if msg.metadata.get("thread_id"):
            parent_message_id = msg.metadata.get("root_id") or msg.metadata.get("message_id")

        if parent_message_id:
            sent_message_id = await loop.run_in_executor(
                None,
                self._reply_message_with_id_sync,
                parent_message_id,
                "text",
                text_body,
            )
        else:
            sent_message_id = await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                msg.chat_id,
                "text",
                text_body,
            )
        if sent_message_id:
            self._tool_progress_messages[progress_id] = sent_message_id
            self._tool_progress_last_notified[progress_id] = now

    def _create_streaming_card_sync(
        self, receive_id_type: str, chat_id: str, reply_message_id: str | None = None
    ) -> str | None:
        """Create a CardKit streaming card, send it to chat, return card_id.

        When reply_message_id is given, the card message is sent as a reply so it
        quotes the user's original message; otherwise it is sent as a new message.
        """
        from lark_oapi.api.cardkit.v1 import CreateCardRequest, CreateCardRequestBody
        card_json = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True, "update_multi": True, "streaming_mode": True},
            "body": {"elements": [{"tag": "markdown", "content": "", "element_id": _STREAM_ELEMENT_ID}]},
        }
        try:
            request = CreateCardRequest.builder().request_body(
                CreateCardRequestBody.builder()
                .type("card_json")
                .data(json.dumps(card_json, ensure_ascii=False))
                .build()
            ).build()
            response = self._client.cardkit.v1.card.create(request)
            if not response.success():
                logger.warning("Failed to create streaming card: code={}, msg={}", response.code, response.msg)
                return None
            card_id = getattr(response.data, "card_id", None)
            if card_id:
                card_content = json.dumps({"type": "card", "data": {"card_id": card_id}})
                if reply_message_id:
                    sent = self._reply_message_sync(reply_message_id, "interactive", card_content)
                    if not sent:
                        # Reply failed (e.g. original message gone); fall back to plain send.
                        sent = bool(self._send_message_sync(
                            receive_id_type, chat_id, "interactive", card_content,
                        ))
                else:
                    sent = bool(self._send_message_sync(
                        receive_id_type, chat_id, "interactive", card_content,
                    ))
                if sent:
                    return card_id
                logger.warning("Created streaming card {} but failed to send it to {}", card_id, chat_id)
            return None
        except Exception as e:
            logger.warning("Error creating streaming card: {}", e)
            return None

    def _stream_update_text_sync(self, card_id: str, content: str, sequence: int) -> bool:
        """Stream-update the markdown element on a CardKit card (typewriter effect)."""
        from lark_oapi.api.cardkit.v1 import ContentCardElementRequest, ContentCardElementRequestBody
        try:
            request = ContentCardElementRequest.builder() \
                .card_id(card_id) \
                .element_id(_STREAM_ELEMENT_ID) \
                .request_body(
                    ContentCardElementRequestBody.builder()
                    .content(content).sequence(sequence).build()
                ).build()
            response = self._client.cardkit.v1.card_element.content(request)
            if not response.success():
                logger.warning("Failed to stream-update card {}: code={}, msg={}", card_id, response.code, response.msg)
                return False
            return True
        except Exception as e:
            logger.warning("Error stream-updating card {}: {}", card_id, e)
            return False

    def _close_streaming_mode_sync(self, card_id: str, sequence: int) -> bool:
        """Turn off CardKit streaming_mode so the chat list preview exits the streaming placeholder.

        Per Feishu docs, streaming cards keep a generating-style summary in the session list until
        streaming_mode is set to false via card settings (after final content update).
        Sequence must strictly exceed the previous card OpenAPI operation on this entity.
        """
        from lark_oapi.api.cardkit.v1 import SettingsCardRequest, SettingsCardRequestBody
        settings_payload = json.dumps({"config": {"streaming_mode": False}}, ensure_ascii=False)
        try:
            request = SettingsCardRequest.builder() \
                .card_id(card_id) \
                .request_body(
                    SettingsCardRequestBody.builder()
                    .settings(settings_payload)
                    .sequence(sequence)
                    .uuid(str(uuid.uuid4()))
                    .build()
                ).build()
            response = self._client.cardkit.v1.card.settings(request)
            if not response.success():
                logger.warning(
                    "Failed to close streaming on card {}: code={}, msg={}",
                    card_id, response.code, response.msg,
                )
                return False
            return True
        except Exception as e:
            logger.warning("Error closing streaming on card {}: {}", card_id, e)
            return False

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """Progressive streaming via CardKit: create card on first delta, stream-update on subsequent."""
        if not self._client:
            return
        meta = metadata or {}
        loop = asyncio.get_running_loop()
        rid_type = "chat_id" if chat_id.startswith("oc_") else "open_id"

        # Decide whether the streaming card should quote the user's message.
        # Mirrors the routing in send(): explicit reply_to_message config, or an
        # implicit thread reply to keep topic-group context in-thread.
        reply_message_id: str | None = None
        if self.config.reply_to_message and meta.get("message_id"):
            reply_message_id = meta.get("message_id")
        elif meta.get("thread_id"):
            reply_message_id = meta.get("root_id") or meta.get("message_id") or None

        # --- stream end: final update or fallback ---
        if meta.get("_stream_end"):
            buf = self._stream_bufs.pop(chat_id, None)
            if not buf or not buf.text:
                return
            self._finalize_stream_mention(buf, force=True)
            if buf.card_id:
                buf.sequence += 1
                await loop.run_in_executor(
                    None, self._stream_update_text_sync, buf.card_id, buf.text, buf.sequence,
                )
                # Required so the chat list preview exits the streaming placeholder (Feishu streaming card docs).
                buf.sequence += 1
                await loop.run_in_executor(
                    None, self._close_streaming_mode_sync, buf.card_id, buf.sequence,
                )
            else:
                first_chunk = True
                for chunk in self._split_elements_by_table_limit(self._build_card_elements(buf.text)):
                    card = json.dumps({"config": {"wide_screen_mode": True}, "elements": chunk}, ensure_ascii=False)
                    if reply_message_id and first_chunk:
                        first_chunk = False
                        ok = await loop.run_in_executor(
                            None, self._reply_message_sync, reply_message_id, "interactive", card,
                        )
                        if ok:
                            continue
                        # Fall back to a plain send if the reply fails.
                    await loop.run_in_executor(None, self._send_message_sync, rid_type, chat_id, "interactive", card)
            return

        # --- accumulate delta ---
        buf = self._stream_bufs.get(chat_id)
        if buf is None:
            buf = _FeishuStreamBuf()
            self._stream_bufs[chat_id] = buf
            mention_target = self._outbound_mention_target(meta)
            if mention_target:
                buf.mention_target = mention_target
                buf.mention_pending = True
        buf.text += delta
        if not buf.text.strip():
            return
        if not self._finalize_stream_mention(buf):
            return

        now = time.monotonic()
        if buf.card_id is None:
            card_id = await loop.run_in_executor(
                None, self._create_streaming_card_sync, rid_type, chat_id, reply_message_id,
            )
            if card_id:
                buf.card_id = card_id
                buf.sequence = 1
                await loop.run_in_executor(None, self._stream_update_text_sync, card_id, buf.text, 1)
                buf.last_edit = now
        elif (now - buf.last_edit) >= self._STREAM_EDIT_INTERVAL:
            buf.sequence += 1
            await loop.run_in_executor(None, self._stream_update_text_sync, buf.card_id, buf.text, buf.sequence)
            buf.last_edit = now

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu, including media (images/files) if present."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            sanitized_content = _sanitize_unresolved_mentions(msg.content)
            if sanitized_content != msg.content:
                logger.warning("Feishu: sanitized unresolved mention placeholder in outbound message")
                msg.content = sanitized_content

            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
            loop = asyncio.get_running_loop()

            if msg.metadata.get("_tool_progress_id"):
                await self._send_tool_progress(msg, receive_id_type, loop)
                return

            # Handle tool hint messages as code blocks in interactive cards.
            # These are progress-only messages and should bypass normal reply routing.
            if msg.metadata.get("_tool_hint"):
                if msg.content and msg.content.strip():
                    await self._send_tool_hint_card(
                        receive_id_type, msg.chat_id, msg.content.strip()
                    )
                return

            # Determine whether the first message should quote the user's message.
            # Only the very first send (media or text) in this call uses reply; subsequent
            # chunks/media fall back to plain create to avoid redundant quote bubbles.
            reply_message_id: str | None = None
            if (
                self.config.reply_to_message
                and not msg.metadata.get("_progress", False)
            ):
                reply_message_id = msg.metadata.get("message_id") or None
            # For topic group messages, always reply to keep context in thread
            elif msg.metadata.get("thread_id"):
                reply_message_id = msg.metadata.get("root_id") or msg.metadata.get("message_id") or None

            first_send = True  # tracks whether the reply has already been used

            def _do_send(m_type: str, content: str) -> None:
                """Send via reply (first message) or create (subsequent)."""
                nonlocal first_send
                sent_id: str | None = None
                if reply_message_id and first_send:
                    first_send = False
                    sent_id = self._reply_message_with_id_sync(reply_message_id, m_type, content)
                if not sent_id:
                    # Regular send, also the fallback when the reply failed.
                    sent_id = self._send_message_sync(receive_id_type, msg.chat_id, m_type, content)
                if sent_id:
                    # A click on this message reports only its message_id, so
                    # remember which session sent it (see _on_card_action).
                    self._remember(self._outbound_routes, sent_id, msg.chat_id)

            for file_path in msg.media:
                if not os.path.isfile(file_path):
                    logger.warning("Media file not found: {}", file_path)
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    key = await loop.run_in_executor(None, self._upload_image_sync, file_path)
                    if key:
                        await loop.run_in_executor(
                            None, _do_send,
                            "image", json.dumps({"image_key": key}, ensure_ascii=False),
                        )
                else:
                    key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                    if key:
                        # Use msg_type "audio" for audio, "video" for video, "file" for documents.
                        # Feishu requires these specific msg_types for inline playback.
                        # Note: "media" is only valid as a tag inside "post" messages, not as a standalone msg_type.
                        if ext in self._AUDIO_EXTS:
                            media_type = "audio"
                        elif ext in self._VIDEO_EXTS:
                            media_type = "video"
                        else:
                            media_type = "file"
                        await loop.run_in_executor(
                            None, _do_send,
                            media_type, json.dumps({"file_key": key}, ensure_ascii=False),
                        )

            if msg.content and msg.content.strip():
                fmt = self._detect_msg_format(msg.content)
                mention_target = self._outbound_mention_target(msg.metadata)
                mention_label = "用户"
                content = msg.content
                if mention_target:
                    # The leading name usually came from the inbound mention of
                    # this bot (for example ``@Jarvis``), not from the sender.
                    # Never reuse it as the sender's outbound at-label.
                    _, content = _split_leading_display_mention(content)

                if fmt == "text":
                    # Short plain text – send as simple text message
                    if mention_target:
                        label = html.escape(mention_label, quote=True)
                        content = (
                            f'<at user_id="{mention_target}">{label}</at> '
                            f"{content}"
                        )
                    text_body = json.dumps({"text": content.strip()}, ensure_ascii=False)
                    await loop.run_in_executor(None, _do_send, "text", text_body)

                elif fmt == "post":
                    # Medium content with links – send as rich-text post
                    post_body = self._markdown_to_post(
                        content,
                        mention_target=mention_target,
                        mention_label=mention_label,
                    )
                    await loop.run_in_executor(None, _do_send, "post", post_body)

                else:
                    # Complex / long content – send as interactive card
                    if mention_target:
                        content = f'<at id="{mention_target}"></at> {content}'
                    elements = self._build_card_elements(content)
                    for chunk in self._split_elements_by_table_limit(elements):
                        card = {"config": {"wide_screen_mode": True}, "elements": chunk}
                        await loop.run_in_executor(
                            None, _do_send,
                            "interactive", json.dumps(card, ensure_ascii=False),
                        )

        except Exception as e:
            logger.error("Error sending Feishu message: {}", e)
            raise

    def _on_message_sync(self, data: Any) -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    @staticmethod
    def _card_action_response(content: str, toast_type: str = "info") -> Any:
        """Build the immediate callback response required by Feishu."""
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        return P2CardActionTriggerResponse({
            "toast": {"type": toast_type, "content": content},
        })

    @staticmethod
    def _card_action_key(data: Any) -> str:
        """Return a stable key for callback retry/click deduplication."""
        event = getattr(data, "event", None)
        token = getattr(event, "token", None)
        if token:
            return str(token)
        operator = getattr(event, "operator", None)
        context = getattr(event, "context", None)
        action = getattr(event, "action", None)
        value = getattr(action, "value", None) or {}
        return "|".join((
            str(getattr(context, "open_message_id", None) or ""),
            str(getattr(operator, "open_id", None) or ""),
            str(getattr(action, "tag", None) or ""),
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
        ))

    def _on_card_action_sync(self, data: Any) -> Any:
        """Acknowledge a card click immediately and dispatch it asynchronously."""
        event = getattr(data, "event", None)
        operator = getattr(event, "operator", None)
        sender_id = str(getattr(operator, "open_id", None) or "unknown")

        if not self.is_allowed(sender_id):
            logger.warning("Rejected Feishu card action from unauthorized user {}", sender_id)
            return self._card_action_response("你没有权限执行这个操作", "error")

        action_key = self._card_action_key(data)
        if action_key in self._processed_card_actions:
            return self._card_action_response("该操作已提交，请勿重复点击", "warning")

        if not self._loop or not self._loop.is_running():
            logger.error("Cannot dispatch Feishu card action: main event loop is not running")
            return self._card_action_response("机器人暂时不可用，请稍后重试", "error")

        self._processed_card_actions[action_key] = None
        while len(self._processed_card_actions) > 1000:
            self._processed_card_actions.popitem(last=False)

        future = asyncio.run_coroutine_threadsafe(self._on_card_action(data), self._loop)

        def _log_result(done: Any) -> None:
            try:
                done.result()
            except Exception as exc:
                logger.error("Error processing Feishu card action: {}", exc)

        future.add_done_callback(_log_result)
        return self._card_action_response("操作已收到，正在处理", "success")

    async def _resolve_card_action_chat_id(
        self, sender_id: str, open_chat_id: str, message_id: str
    ) -> str:
        """Map a card click back to the session that produced the card.

        Card callbacks only carry ``open_chat_id``, which is an ``oc_…`` id even
        for private chats, while p2p sessions are keyed by the user's ``ou_…``
        open_id (see ``_on_message``).  Using ``open_chat_id`` verbatim would
        split the clicker off into a second session, so resolve it against what
        we already know about this chat, and only ask Feishu as a last resort.
        """
        if message_id:
            route = self._outbound_routes.get(message_id)
            if route:
                return route
        if not open_chat_id:
            return sender_id
        route = self._chat_routes.get(open_chat_id)
        if route:
            return route
        if self._client:
            loop = asyncio.get_running_loop()
            chat_mode = await loop.run_in_executor(
                None, self._get_chat_mode_sync, open_chat_id
            )
            if chat_mode:
                route = sender_id if chat_mode == "p2p" else open_chat_id
                self._remember(self._chat_routes, open_chat_id, route)
                return route
        logger.warning(
            "Feishu: could not resolve session for card click in {}; "
            "falling back to the chat id",
            open_chat_id,
        )
        return open_chat_id

    async def _on_card_action(self, data: Any) -> None:
        """Convert a Feishu card interaction into an inbound agent instruction."""
        event = data.event
        operator = getattr(event, "operator", None)
        context = getattr(event, "context", None)
        action = getattr(event, "action", None)

        sender_id = str(getattr(operator, "open_id", None) or "unknown")
        open_chat_id = str(getattr(context, "open_chat_id", None) or "")
        message_id = str(getattr(context, "open_message_id", None) or "")
        chat_id = await self._resolve_card_action_chat_id(
            sender_id, open_chat_id, message_id
        )
        value = getattr(action, "value", None) or {}
        form_value = getattr(action, "form_value", None) or {}

        details = {
            "tag": getattr(action, "tag", None),
            "name": getattr(action, "name", None),
            "value": value,
            "form_value": form_value,
            "option": getattr(action, "option", None),
            "input_value": getattr(action, "input_value", None),
            "options": getattr(action, "options", None),
            "checked": getattr(action, "checked", None),
        }
        details = {key: val for key, val in details.items() if val not in (None, "", {}, [])}
        details_json = json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
        if len(details_json) > 8000:
            details_json = details_json[:8000] + "…"

        content = (
            "[Feishu card action]\n"
            "The user explicitly clicked an interactive card control. "
            "Handle the selected action as the user's current instruction.\n"
            f"Action payload: {details_json}"
        )
        # A card click routed to the chat id itself is a group session; a p2p
        # session routes to the clicker's open_id.
        chat_type = "group" if chat_id == open_chat_id else "p2p"
        logger.info(
            "Feishu card action from {} in {} (session {}): {}",
            sender_id,
            open_chat_id or chat_id,
            chat_id,
            details_json[:500],
        )
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata={
                "message_id": message_id or None,
                "chat_type": chat_type,
                "msg_type": "card_action",
                "sender_open_id": sender_id,
                "card_action": details,
                "card_action_token": getattr(event, "token", None),
            },
        )

    async def _on_message(self, data: Any) -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return

            # Staleness guard — drop replayed/old messages that dedup-by-id can't
            # catch (Feishu can re-push an old message as a new event after a
            # reconnect or process restart, with a fresh message_id).
            create_time_ms = int(message.create_time) if getattr(message, "create_time", None) else 0
            if create_time_ms:
                if create_time_ms < self._start_time_ms:
                    logger.warning(
                        "Feishu: dropping pre-startup message {} (created {}s before channel start)",
                        message_id, (self._start_time_ms - create_time_ms) // 1000,
                    )
                    return
                if self._stale_threshold_ms > 0:
                    age_ms = int(time.time() * 1000) - create_time_ms
                    if age_ms > self._stale_threshold_ms:
                        logger.warning(
                            "Feishu: dropping stale message {} (age {}s > {}s)",
                            message_id, age_ms // 1000, self._stale_threshold_ms // 1000,
                        )
                        return

            self._processed_message_ids[message_id] = None

            # Trim cache
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type
            sender_type = sender.sender_type

            if sender_type == "bot":
                # Feishu only delivers these with
                # im:message.group_at_msg.include_bot:readonly. Keep an explicit
                # mention requirement even when groupPolicy=open so two open bots
                # cannot accidentally enter an unbounded reply loop.
                if chat_type != "group" or not self._is_bot_mentioned(message):
                    logger.debug("Feishu: skipping non-directed bot message")
                    return
            elif chat_type == "group" and not self._is_group_message_for_bot(message):
                # Not answered, but kept: people often post the question and then
                # @ the bot in a separate message, which on its own says nothing.
                self._buffer_group_context(message, sender_id)
                logger.debug("Feishu: buffering group message (not mentioned)")
                return

            # Add reaction
            await self._add_reaction(message_id, self.config.react_emoji)

            # Parse content
            content_parts = []
            media_paths = []

            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}
            resolved_mentions: list[dict[str, str]] = []

            if msg_type == "text":
                text = content_json.get("text", "")
                if text:
                    text, resolved_mentions = _resolve_text_mentions(
                        text,
                        getattr(message, "mentions", None),
                    )
                    content_parts.append(text)

            elif msg_type == "post":
                text, image_keys = _extract_post_content(content_json)
                if text:
                    content_parts.append(text)
                # Download images embedded in post
                for img_key in image_keys:
                    file_path, content_text = await self._download_and_save_media(
                        "image", {"image_key": img_key}, message_id
                    )
                    if file_path:
                        media_paths.append(file_path)
                    content_parts.append(content_text)

            elif msg_type in ("image", "audio", "file", "media"):
                file_path, content_text = await self._download_and_save_media(msg_type, content_json, message_id)
                if file_path:
                    media_paths.append(file_path)

                if msg_type == "audio" and file_path:
                    transcription = await self.transcribe_audio(file_path)
                    if transcription:
                        content_text = f"[transcription: {transcription}]"

                content_parts.append(content_text)

            elif msg_type in ("share_chat", "share_user", "interactive", "share_calendar_event", "system", "merge_forward"):
                # Handle share cards and interactive messages
                text = _extract_share_card_content(content_json, msg_type)
                if text:
                    content_parts.append(text)

            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            # Extract reply context (parent/root message IDs)
            parent_id = getattr(message, "parent_id", None) or None
            root_id = getattr(message, "root_id", None) or None
            thread_id = getattr(message, "thread_id", None) or None

            # Prepend quoted message text when the user replied to another message
            if parent_id and self._client:
                loop = asyncio.get_running_loop()
                reply_ctx = await loop.run_in_executor(
                    None, self._get_message_content_sync, parent_id
                )
                if reply_ctx:
                    content_parts.insert(0, reply_ctx)

            content = "\n".join(content_parts) if content_parts else ""

            if not content and not media_paths:
                return
            if sender_type == "bot":
                content = (
                    "[Message from another Feishu bot; explicitly addressed to this bot]\n"
                    + content
                )

            # Hand over anything said in this group since the bot was last
            # addressed, so a bare "@bot" still carries what it refers to.
            if chat_type == "group":
                if group_context := self._drain_group_context(chat_id, sender_id):
                    content = f"{group_context}\n\n{content}" if content else group_context

            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            # Card clicks only report open_chat_id; remember how this chat maps
            # to a session so the callback routes back to the same session.
            if chat_id:
                self._remember(self._chat_routes, chat_id, reply_to)
            if chat_type == "group" and sender_type != "bot" and sender_id.startswith("ou_"):
                self._remember(self._reply_targets, message_id, sender_id)
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "sender_type": sender_type,
                    "sender_open_id": sender_id,
                    "mentions": resolved_mentions,
                    "parent_id": parent_id,
                    "root_id": root_id,
                    "thread_id": thread_id,
                }
            )

        except Exception as e:
            logger.error("Error processing Feishu message: {}", e)

    def _on_reaction_created(self, data: Any) -> None:
        """Ignore reaction events so they do not generate SDK noise."""
        pass

    def _on_message_read(self, data: Any) -> None:
        """Ignore read events so they do not generate SDK noise."""
        pass

    def _on_bot_p2p_chat_entered(self, data: Any) -> None:
        """Ignore p2p-enter events when a user opens a bot chat."""
        logger.debug("Bot entered p2p chat (user opened chat window)")
        pass

    @staticmethod
    def _format_tool_hint_lines(tool_hint: str) -> str:
        """Split tool hints across lines on top-level call separators only."""
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        in_string = False
        quote_char = ""
        escaped = False

        for i, ch in enumerate(tool_hint):
            buf.append(ch)

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote_char:
                    in_string = False
                continue

            if ch in {'"', "'"}:
                in_string = True
                quote_char = ch
                continue

            if ch == "(":
                depth += 1
                continue

            if ch == ")" and depth > 0:
                depth -= 1
                continue

            if ch == "," and depth == 0:
                next_char = tool_hint[i + 1] if i + 1 < len(tool_hint) else ""
                if next_char == " ":
                    parts.append("".join(buf).rstrip())
                    buf = []

        if buf:
            parts.append("".join(buf).strip())

        return "\n".join(part for part in parts if part)

    async def _send_tool_hint_card(self, receive_id_type: str, receive_id: str, tool_hint: str) -> None:
        """Send tool hint as an interactive card with formatted code block.

        Args:
            receive_id_type: "chat_id" or "open_id"
            receive_id: The target chat or user ID
            tool_hint: Formatted tool hint string (e.g., 'web_search("q"), read_file("path")')
        """
        loop = asyncio.get_running_loop()

        # Put each top-level tool call on its own line without altering commas inside arguments.
        formatted_code = self._format_tool_hint_lines(tool_hint)

        card = {
            "config": {"wide_screen_mode": True},
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"**Tool Calls**\n\n```text\n{formatted_code}\n```"
                }
            ]
        }

        await loop.run_in_executor(
            None, self._send_message_sync,
            receive_id_type, receive_id, "interactive",
            json.dumps(card, ensure_ascii=False),
        )
