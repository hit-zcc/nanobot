"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, command_text, register_builtin_commands
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _BACKGROUND_STATUS_QUERIES = {
        "咋样了",
        "怎么样了",
        "进度",
        "进度咋样",
        "进度怎么样",
        "任务咋样了",
        "任务怎么样了",
        "后台任务咋样了",
        "后台任务怎么样了",
        "看看任务咋样了",
        "看看任务怎么样了",
        "看看进度",
        "现在咋样了",
        "现在怎么样了",
        "嗯咋样了",
        "到哪了",
        "好了吗",
        "完成了吗",
    }
    _TOOL_DISPLAY_NAMES = {
        "exec": "执行命令",
        "read_file": "读取文件",
        "write_file": "写入文件",
        "edit_file": "编辑文件",
        "list_dir": "浏览目录",
        "web_search": "搜索网页",
        "web_fetch": "获取网页",
        "message": "发送消息",
        "spawn": "启动子任务",
        "cron": "管理定时任务",
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 80,
        max_subagent_iterations: int = 80,
        context_window_tokens: int = 65_536,
        memory_max_tokens: int = 24_000,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.memory_max_tokens = memory_max_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}

        self.context = ContextBuilder(workspace, timezone=timezone)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            max_iterations=max_subagent_iterations,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Coalesce bursts of messages in the same session into a single turn so
        # e.g. an image followed quickly by a question gets one combined reply
        # instead of two. NANOBOT_COALESCE_WINDOW seconds of quiet ends a burst
        # (<=0 disables coalescing).
        self._coalesce_window: float = float(
            os.environ.get("NANOBOT_COALESCE_WINDOW", "3.0")
        )
        self._inbound_buffer: dict[str, list[InboundMessage]] = {}
        self._debounce_tasks: dict[str, asyncio.Task] = {}
        # Messages that arrived while a turn was already running. They are
        # spliced into that run at its next tool boundary rather than queued
        # behind it, so the user can steer or interrupt long jobs.
        self._pending_injections: dict[str, list[InboundMessage]] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            memory_max_tokens=memory_max_tokens,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    def _refusal_fallback_model(self) -> str | None:
        """Stronger model to retry on when the primary model safety-refuses.

        Opus 4.8 lacks the extra Fable/Mythos safety layer, so it recovers most
        spurious refusals. Only applies to Anthropic-family models; returns None
        when already on Opus 4.8 or on a non-Claude provider.
        """
        model = self.model or ""
        prefix, _, name = model.rpartition("/")
        target = name or model
        markers = ("claude", "fable", "mythos", "opus", "sonnet", "haiku")
        if not any(m in target for m in markers):
            return None
        if "opus-4-8" in target:
            return None
        fallback = "claude-opus-4-8"
        return f"{prefix}/{fallback}" if prefix else fallback

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours} 小时")
        if minutes:
            parts.append(f"{minutes} 分")
        if secs or not parts:
            parts.append(f"{secs} 秒")
        return " ".join(parts)

    @classmethod
    def _format_tool_names(cls, names: list[str]) -> list[str]:
        return [cls._TOOL_DISPLAY_NAMES.get(name, f"`{name}`") for name in names]

    @classmethod
    def _format_tool_progress(cls, names: list[str], elapsed: float, *, done: bool) -> str:
        labels = cls._format_tool_names(names)
        duration = cls._format_duration(elapsed)
        if done:
            subject = labels[0] if len(labels) == 1 else "、".join(labels)
            suffix = "完成" if len(labels) == 1 else "已完成"
            return f"✅ {subject}{suffix} · 共用时 {duration}"
        if len(labels) == 1:
            return f"⏳ 正在{labels[0]} · 已用时 {duration}"
        return f"⏳ 正在处理：{'、'.join(labels)} · 已用时 {duration}"

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_notice: Callable[[str], Awaitable[None]] | None = None,
        on_tool_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        run_state: dict[str, Any] | None = None,
        message_sink: list[dict] | None = None,
        injection_key: str | None = None,
        on_checkpoint: Callable[[list[dict]], None] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop.

        *message_sink*: when given, receives the live working message list so a
        cancelled turn can still be persisted by the caller.
        *injection_key*: session key whose queued interjections are spliced in
        at tool boundaries, letting the user steer a run already in flight.
        *on_checkpoint*: called at each iteration boundary with the live message
        list so the caller can persist progress before the turn completes.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_self = self

        class _LoopHook(AgentHook):
            def __init__(self) -> None:
                self._stream_buf = ""
                self._tool_progress_id = f"tool-progress:{time.time_ns()}"
                self._tool_progress_started: float | None = None
                self._tool_progress_tools: list[str] = []

            def wants_streaming(self) -> bool:
                return on_stream is not None

            async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                from nanobot.utils.helpers import strip_think

                prev_clean = strip_think(self._stream_buf)
                self._stream_buf += delta
                new_clean = strip_think(self._stream_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental and on_stream:
                    await on_stream(incremental)

            async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
                if on_stream_end:
                    await on_stream_end(resuming=resuming)
                self._stream_buf = ""

            async def before_execute_tools(self, context: AgentHookContext) -> None:
                if on_progress:
                    if not on_stream:
                        thought = loop_self._strip_think(context.response.content if context.response else None)
                        if thought:
                            await on_progress(thought)
                    tool_hint = loop_self._strip_think(loop_self._tool_hint(context.tool_calls))
                    await on_progress(tool_hint, tool_hint=True)
                for tc in context.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])
                loop_self._set_tool_context(channel, chat_id, message_id)

            async def on_notice(self, context: AgentHookContext, message: str) -> None:
                if on_notice:
                    await on_notice(message)
                elif on_progress:
                    await on_progress(message)

            async def take_injections(self, context: AgentHookContext) -> list[dict[str, Any]]:
                if not injection_key:
                    return []
                pending = loop_self._take_injections(injection_key)
                if not pending:
                    return []
                built = loop_self._build_injection_messages(pending)
                if not built:
                    return []
                logger.info(
                    "Injecting {} interjection(s) into running turn for {}",
                    len(pending), injection_key,
                )
                await self.on_notice(
                    context, "📨 收到你的新消息，已插进当前任务，我马上把它一起考虑。",
                )
                return built

            async def on_tool_heartbeat(
                self, context: AgentHookContext, *, elapsed: float, pending: list[str],
            ) -> None:
                # Sent as plain progress (not tool_hint) on purpose: this is a
                # liveness signal for a job already running long, so it must not
                # be gated behind sendToolHints.
                if self._tool_progress_started is None:
                    self._tool_progress_started = time.monotonic() - elapsed
                for name in pending:
                    if name not in self._tool_progress_tools:
                        self._tool_progress_tools.append(name)
                total_elapsed = time.monotonic() - self._tool_progress_started
                content = loop_self._format_tool_progress(pending, total_elapsed, done=False)
                if on_tool_progress:
                    await on_tool_progress(
                        content, progress_id=self._tool_progress_id, done=False,
                    )
                elif on_progress:
                    await on_progress(content)

            async def finish_tool_progress(self) -> None:
                if self._tool_progress_started is None or not on_tool_progress:
                    return
                elapsed = time.monotonic() - self._tool_progress_started
                content = loop_self._format_tool_progress(
                    self._tool_progress_tools, elapsed, done=True,
                )
                await on_tool_progress(
                    content, progress_id=self._tool_progress_id, done=True,
                )

            async def after_iteration(self, context: AgentHookContext) -> None:
                if on_checkpoint:
                    on_checkpoint(context.messages)

            def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
                return loop_self._strip_think(content)

        hook = _LoopHook()
        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            max_iterations_message=(
                "⚠️ 当前任务已达到工具调用硬限制（{max_iterations}轮），"
                "系统因此停止了本轮；这不代表任务已经完成。"
                "你可以让我继续，或把任务拆成更小的步骤。"
            ),
            iteration_warning_remaining=10,
            iteration_warning_message=(
                "⚠️ 当前任务已使用 {used_iterations}/{max_iterations} 轮模型/工具循环，"
                "距离硬限制只剩 {remaining_iterations} 轮。"
                "我会开始收敛并优先汇报结果；若仍未完成，触顶时会明确说明。"
            ),
            refusal_message=(
                "这条请求触发了模型的安全限制，我没法继续生成。"
                "可以换个说法、拆成几步分开发，或改用其他模型再试。"
            ),
            refusal_fallback_model=loop_self._refusal_fallback_model(),
            concurrent_tools=True,
            message_sink=message_sink,
        ))
        await hook.finish_tool_progress()
        if run_state is not None:
            run_state["stop_reason"] = result.stop_reason
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        elif result.stop_reason == "refusal":
            logger.warning("LLM refused (safety stop): {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            await self.accept_inbound(msg)

    async def accept_inbound(self, msg: InboundMessage) -> None:
        """Route one inbound message to the right entry path.

        Single source of truth for admission, shared by this loop and by
        AgentRouter in multi-agent mode. Duplicating it once already meant the
        router silently missed interjection support.
        """
        raw = command_text(msg.content, msg.metadata)
        if self.commands.is_priority(raw):
            ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
            if result := await self.commands.dispatch_priority(ctx):
                await self.bus.publish_outbound(result)
            return

        # A message landing mid-turn steers that turn instead of queueing behind
        # it. System notifications are excluded: they are independent turns, and
        # must neither interrupt the user's turn nor merge into their text burst.
        if msg.channel != "system" and self._has_active_run(msg):
            self._queue_injection(msg)
        elif self._coalesce_window > 0 and msg.channel != "system":
            self._buffer_inbound(msg)
        else:
            self._spawn_dispatch(msg)

    @staticmethod
    def _dispatch_session_key(msg: InboundMessage) -> str:
        """Resolve the conversation key used for locking and task tracking."""
        if msg.session_key_override:
            return msg.session_key_override
        if msg.channel == "system" and ":" in msg.chat_id:
            # Defensive compatibility for system producers that encode their
            # target as "channel:chat_id" but omit an explicit override.
            return msg.chat_id
        return msg.session_key

    def _spawn_dispatch(self, msg: InboundMessage) -> None:
        """Dispatch a (possibly merged) message as a tracked, cancellable task."""
        task = asyncio.create_task(self._dispatch(msg))
        key = self._dispatch_session_key(msg)
        self._active_tasks.setdefault(key, []).append(task)
        task.add_done_callback(lambda t, k=key: self._on_dispatch_done(k, t))

    def _on_dispatch_done(self, key: str, task: asyncio.Task) -> None:
        """Untrack a finished dispatch and rescue any last-moment interjection.

        A message can be queued as an interjection in the window between the
        run's final boundary check and this callback, when the task is still
        listed as active. Nobody would ever consume it, so once the session is
        truly idle, replay whatever is left.
        """
        tasks = self._active_tasks.get(key, [])
        if task in tasks:
            tasks.remove(task)
        if not tasks:
            self._active_tasks.pop(key, None)
        if task.cancelled():
            self._take_injections(key)  # /stop must not restart work
        elif not any(not t.done() for t in self._active_tasks.get(key, [])):
            self._replay_unconsumed_injections(key)

    def _has_active_run(self, msg: InboundMessage) -> bool:
        """True if a turn for this session is currently executing."""
        key = self._dispatch_session_key(msg)
        return any(not t.done() for t in self._active_tasks.get(key, []))

    def _queue_injection(self, msg: InboundMessage) -> None:
        """Hold a message for the running turn to pick up at its next boundary."""
        key = self._dispatch_session_key(msg)
        self._pending_injections.setdefault(key, []).append(msg)
        logger.info("Queued interjection for running turn in session {}", key)

    def _take_injections(self, key: str) -> list[InboundMessage]:
        """Remove and return the messages queued for *key*."""
        return self._pending_injections.pop(key, [])

    def _buffer_inbound(self, msg: InboundMessage) -> None:
        """Buffer a message and (re)start the per-session coalesce timer.

        Each new message in the same session slides the window, so a burst is
        only flushed once `self._coalesce_window` seconds of quiet have passed.
        """
        key = self._dispatch_session_key(msg)
        self._inbound_buffer.setdefault(key, []).append(msg)
        if old := self._debounce_tasks.get(key):
            old.cancel()
        self._debounce_tasks[key] = asyncio.create_task(self._flush_after_quiet(key))

    async def _flush_after_quiet(self, key: str) -> None:
        """Wait for the quiet window, then dispatch the buffered burst as one turn."""
        try:
            await asyncio.sleep(self._coalesce_window)
        except asyncio.CancelledError:
            return  # superseded by a newer message; that timer will flush
        self._debounce_tasks.pop(key, None)
        batch = self._inbound_buffer.pop(key, [])
        if not batch:
            return
        merged = batch[0] if len(batch) == 1 else self._merge_inbound(batch)
        self._spawn_dispatch(merged)

    @staticmethod
    def _merge_inbound(batch: list[InboundMessage]) -> InboundMessage:
        """Merge a burst of messages into one: text joined, media concatenated.

        The last message supplies channel/chat/sender/metadata so reactions and
        replies target the most recent message of the burst.
        """
        last = batch[-1]
        contents = [m.content.strip() for m in batch if m.content and m.content.strip()]
        media: list[str] = []
        for m in batch:
            media.extend(m.media)
        return InboundMessage(
            channel=last.channel,
            sender_id=last.sender_id,
            chat_id=last.chat_id,
            content="\n".join(contents),
            media=media,
            metadata=last.metadata,
            session_key_override=last.session_key_override,
        )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        dispatch_key = self._dispatch_session_key(msg)
        lock = self._session_locks.setdefault(dispatch_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    # Carry the original message's reply-routing keys into the
                    # stream deltas so channels that quote/reply (e.g. Feishu
                    # streaming cards) can target the user's original message.
                    reply_meta = {
                        k: msg.metadata[k]
                        for k in ("message_id", "root_id", "thread_id")
                        if msg.metadata.get(k)
                    }

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta,
                            metadata={
                                **reply_meta,
                                "_stream_delta": True,
                                "_stream_id": _current_stream_id(),
                            },
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="",
                            metadata={
                                "_stream_end": True,
                                "_resuming": resuming,
                                "_stream_id": _current_stream_id(),
                            },
                        ))
                        stream_segment += 1

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", dispatch_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    def _replay_unconsumed_injections(self, key: str) -> None:
        """Dispatch interjections the finished turn never picked up.

        A turn that ended without hitting a tool boundary (or ended right after
        one) leaves them unconsumed; they become their own turn rather than
        being dropped.
        """
        pending = self._take_injections(key)
        if not pending:
            return
        logger.info(
            "Replaying {} unconsumed interjection(s) as a new turn for {}",
            len(pending), key,
        )
        merged = pending[0] if len(pending) == 1 else self._merge_inbound(pending)
        self._spawn_dispatch(merged)

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        for task in self._debounce_tasks.values():
            task.cancel()
        self._debounce_tasks.clear()
        self._inbound_buffer.clear()
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
            )
            live_msgs: list[dict] = []
            cursor = [1 + len(history)]

            def _checkpoint(msgs: list[dict]) -> None:
                cursor[0] = self._checkpoint_turn(session, msgs, cursor[0], key)

            try:
                final_content, _, all_msgs = await self._run_agent_loop(
                    messages, channel=channel, chat_id=chat_id,
                    message_id=msg.metadata.get("message_id"),
                    message_sink=live_msgs,
                    on_checkpoint=_checkpoint,
                )
            except asyncio.CancelledError:
                self._save_cancelled_turn(session, live_msgs, cursor[0], key)
                raise
            self._save_turn(session, all_msgs, cursor[0])
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = command_text(msg.content, msg.metadata)
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        normalized_raw = "".join(raw.split()).rstrip("？?！!。")
        if normalized_raw in self._BACKGROUND_STATUS_QUERIES:
            status = self.subagents.format_session_status(key)
            if isinstance(status, str) and status:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=status,
                    metadata=dict(msg.metadata or {}),
                )

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        background_context = self.subagents.format_session_context(key)
        if not isinstance(background_context, str):
            background_context = None
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            runtime_metadata=background_context,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        async def _bus_notice(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_important_notice"] = True
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        async def _bus_tool_progress(
            content: str, *, progress_id: str, done: bool,
        ) -> None:
            if msg.channel.split(".", 1)[0] != "feishu":
                if not done:
                    await _bus_progress(content)
                return
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_progress_id"] = progress_id
            meta["_tool_progress_done"] = done
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # Only suppress the final outbound message when the final, successful
        # stream segment already delivered it. Earlier tool-call commentary must
        # not hide a later error or max-iteration explanation.
        streamed_final_segment = False
        wrapped_on_stream = None
        wrapped_on_stream_end = on_stream_end
        if on_stream is not None:
            _orig_on_stream = on_stream

            async def wrapped_on_stream(delta: str) -> None:
                nonlocal streamed_final_segment
                if delta.strip():
                    streamed_final_segment = True
                await _orig_on_stream(delta)

            if on_stream_end is not None:
                _orig_on_stream_end = on_stream_end

                async def wrapped_on_stream_end(*, resuming: bool = False) -> None:
                    nonlocal streamed_final_segment
                    await _orig_on_stream_end(resuming=resuming)
                    if resuming:
                        streamed_final_segment = False

        run_state: dict[str, Any] = {}
        live_msgs: list[dict] = []
        cursor = [1 + len(history)]

        def _checkpoint(msgs: list[dict]) -> None:
            cursor[0] = self._checkpoint_turn(session, msgs, cursor[0], key)

        try:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress or _bus_progress,
                on_notice=_bus_notice,
                on_tool_progress=_bus_tool_progress if on_progress is None else None,
                on_stream=wrapped_on_stream,
                on_stream_end=wrapped_on_stream_end,
                channel=msg.channel, chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
                run_state=run_state,
                message_sink=live_msgs,
                injection_key=self._dispatch_session_key(msg),
                on_checkpoint=_checkpoint,
            )
        except asyncio.CancelledError:
            # A stopped run must not silently swallow whatever the user sent
            # while it ran: persist those interjections instead of replaying
            # them, so /stop really stops rather than restarting work.
            orphaned = self._build_injection_messages(
                self._take_injections(self._dispatch_session_key(msg))
            )
            self._save_cancelled_turn(
                session, live_msgs, cursor[0], key, extra=orphaned,
            )
            raise

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, cursor[0])
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if (
            streamed_final_segment
            and run_state.get("stop_reason") == "completed"
        ):
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    _INTERJECTION_PREFIX = (
        "[用户在你执行任务的过程中发来了新消息。请先读懂它：如果是纠正或改变方向，"
        "立即调整后续步骤；如果只是补充信息，就并入当前任务继续。不要重新开始已完成的工作。]"
    )

    def _build_injection_messages(
        self, pending: list[InboundMessage],
    ) -> list[dict[str, Any]]:
        """Turn queued interjections into a single user message for the run."""
        texts = [m.content.strip() for m in pending if m.content and m.content.strip()]
        media: list[str] = []
        for m in pending:
            media.extend(m.media)
        if not texts and not media:
            return []
        body = "\n".join(texts) if texts else "（用户发来了附件）"
        return [self.context.build_user_message(
            f"{self._INTERJECTION_PREFIX}\n{body}", media or None,
        )]

    _CANCELLED_TOOL_RESULT = (
        "Error: tool execution was interrupted by the user (/stop). "
        "The result is unknown — re-run it if you still need the output."
    )

    @staticmethod
    def _seal_dangling_tool_calls(messages: list[dict]) -> list[dict]:
        """Append synthetic results for tool calls left unanswered by a cancel.

        A turn cancelled mid-tool leaves an assistant message whose
        ``tool_calls`` have no matching ``tool`` results.  Persisting that as-is
        makes the next request illegal for most providers, so every orphan call
        gets an explicit "interrupted" result instead of dropping the message
        (which would hide from the model that work was already attempted).
        """
        answered = {
            str(m.get("tool_call_id"))
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        sealed = list(messages)
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tid = str(tc.get("id") or "")
                if not tid or tid in answered:
                    continue
                answered.add(tid)
                sealed.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "name": (tc.get("function") or {}).get("name", ""),
                    "content": AgentLoop._CANCELLED_TOOL_RESULT,
                })
        return sealed

    def _checkpoint_turn(
        self, session: Session, messages: list[dict], cursor: int, key: str,
    ) -> int:
        """Persist the turn so far, mid-run. Returns the new cursor.

        Saving only at the end of a turn means anything that kills the process
        outright — ``os.execv`` on restart, SIGKILL, a crash — takes the whole
        turn with it, including the messages describing what was just done.
        Checkpointing at iteration boundaries bounds that loss to the current
        step. Boundaries are also the only safe place: every tool call already
        has its result, so the persisted history stays legal.
        """
        if len(messages) <= cursor:
            return cursor
        try:
            self._save_turn(session, messages, cursor)
            self.sessions.save(session)
            return len(messages)
        except Exception:
            logger.exception("Mid-run checkpoint failed for {}", key)
            return cursor

    def _save_cancelled_turn(
        self, session: Session, messages: list[dict], skip: int, key: str,
        extra: list[dict] | None = None,
    ) -> None:
        """Persist the work completed before a /stop so context is not lost.

        *extra* is appended after tool results are sealed, so interjections that
        never got consumed still reach the history in a legal position.
        """
        try:
            if len(messages) <= skip and not extra:
                return
            sealed = self._seal_dangling_tool_calls(messages)
            if extra:
                sealed = sealed + extra
            self._save_turn(session, sealed, skip)
            self.sessions.save(session)
            logger.info(
                "Session saved after cancellation for {} ({} new messages)",
                key, len(messages) - skip,
            )
        except Exception:
            logger.exception("Failed to save session after cancellation for {}", key)

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )
