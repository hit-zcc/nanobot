"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


def _flush_sessions(ctx: CommandContext) -> None:
    """Best-effort persist of cached sessions before the process is replaced."""
    try:
        sessions = getattr(ctx.loop, "sessions", None)
        for session in list(getattr(sessions, "_cache", {}).values()):
            try:
                sessions.save(session)
            except Exception:
                logger.exception("Failed to flush session {} before restart", session.key)
    except Exception:
        logger.exception("Session flush before restart failed")


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        # execv replaces the process image, so no atexit/finally handler ever
        # runs: anything still only in memory is gone. Flush first.
        _flush_sessions(ctx)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    # A new conversation re-arms the reminders. Their once-per-session state
    # is keyed by session key, and /new keeps the key while discarding the
    # history — so without this the fresh session inherits "already said
    # that" from the conversation it replaced.
    for engine, method in ((getattr(loop, "tool_triggers", None), "reset"),
                           (getattr(loop, "tool_preflight", None), "reset_session")):
        reset = getattr(engine, method, None)
        if callable(reset):
            try:
                reset(session.key)
            except Exception:  # never let a nicety break /new
                logger.warning("/new: could not reset {}", method)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


def _text_reply(ctx: CommandContext, content: str) -> OutboundMessage:
    """Reply with plain text, bypassing any channel-side markdown rendering."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={"render_as": "text"},
    )


def _describe_preset(name: str, preset: Any) -> str:
    """One list line for a preset: name, model, and its own description."""
    suffix = f" — {preset.description}" if preset.description else ""
    return f"  {name} → {preset.model}{suffix}"


def _persist_defaults(path: Path, overrides: dict[str, Any]) -> str | None:
    """Patch `agents.defaults` in the config file. Returns an error, or None.

    The file is edited key by key rather than re-serialized from the loaded
    Config: a round-trip would rewrite every channel and provider section, so a
    single unrecognized key anywhere in the user's file would be dropped by a
    model switch.
    """
    from pydantic.alias_generators import to_camel

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        defaults = raw.setdefault("agents", {}).setdefault("defaults", {})
        for key, value in overrides.items():
            # Written back under whichever spelling the file already uses, so a
            # snake_case config does not grow a second camelCase copy of a key.
            defaults[to_camel(key) if key not in defaults else key] = value
        path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # a live switch must not fail over a save
        logger.warning("/model: could not persist to {}: {}", path, exc)
        return str(exc)
    return None


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show the current model, or switch this agent to another one.

    `/model` lists what is available, `/model <preset>` switches to a preset
    from ``agents.presets``, and `/model <provider>/<name>` switches to any
    model the registry can route without needing a preset.
    """
    from nanobot.config.loader import get_config_path, load_config
    from nanobot.providers.factory import ProviderConfigError, build_provider

    loop = ctx.loop
    arg = (ctx.args or "").strip()
    config = load_config()
    presets = config.agents.presets

    if not arg:
        lines = [f"\U0001f9e0 Model: {loop.model}"]
        if presets:
            lines.append("Presets:")
            lines += [_describe_preset(name, p) for name, p in presets.items()]
            lines.append("Switch with: /model <preset> or /model <provider>/<model>")
        else:
            lines.append("No presets configured (agents.presets in the config file).")
            lines.append("Switch with: /model <provider>/<model>")
        return _text_reply(ctx, "\n".join(lines))

    matched = next(
        ((name, p) for name, p in presets.items() if name.lower() == arg.lower()), None
    )
    if matched is None and "/" not in arg:
        lines = [f"Unknown model preset: {arg}"]
        if presets:
            lines.append("Available:")
            lines += [_describe_preset(name, p) for name, p in presets.items()]
        lines.append("Or give a full id: /model <provider>/<model>")
        return _text_reply(ctx, "\n".join(lines))

    if matched is not None:
        preset_name, preset = matched
        overrides = preset.overrides()
    else:
        # A fully-qualified id carries its own provider prefix. Any forced
        # `provider` from the previous model would override that prefix and
        # send the new model name to the old backend, so it is cleared.
        preset_name, overrides = None, {"model": arg, "provider": "auto"}

    config.agents.defaults = config.agents.defaults.model_copy(update=overrides)
    defaults = config.agents.defaults
    try:
        provider = build_provider(config)
    except ProviderConfigError as exc:
        return _text_reply(ctx, "\n".join(
            [f"Cannot switch to {defaults.model}: {exc.message}", *exc.hints]
        ))

    previous = loop.model
    loop.switch_model(provider, defaults.model, defaults.context_window_tokens)

    label = f"{defaults.model} (preset: {preset_name})" if preset_name else defaults.model
    lines = [f"\U0001f9e0 Model: {label}", f"Was: {previous}"]
    if error := _persist_defaults(get_config_path(), overrides):
        lines.append(f"⚠️ Active now, but not saved to the config file: {error}")
    elif len(config.agents.agents) > 1:
        lines.append("Saved as the default; other agents switch on next restart.")
    else:
        lines.append("Saved as the default for the next restart.")
    return _text_reply(ctx, "\n".join(lines))


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/model — Show or switch the model (e.g. /model claude)",
        "/help — Show available commands",
    ]
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
