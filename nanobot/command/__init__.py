"""Slash command routing and built-in handlers."""

from nanobot.command.builtin import register_builtin_commands
from nanobot.command.router import CommandContext, CommandRouter, command_text

__all__ = ["CommandContext", "CommandRouter", "command_text", "register_builtin_commands"]
