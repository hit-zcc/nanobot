"""Claude Code tool — interact with Claude Code CLI from nanobot agent."""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

from nanobot.agent.tools.base import Tool


class ClaudeCodeTool(Tool):
    """Run Claude Code as a sub-agent and return its response.

    Allows the nanobot agent to delegate tasks to Claude Code,
    which has full coding capabilities (file editing, shell access,
    web search, etc.) backed by the user's Max subscription.
    """

    @property
    def name(self) -> str:
        return "claude_code"

    @property
    def description(self) -> str:
        return (
            "Delegate a task to Claude Code (a powerful coding agent with file editing, "
            "shell access, and web search). Use this for complex coding tasks, code review, "
            "refactoring, or when you need Claude Code's built-in tools. "
            "You can optionally resume a previous session by providing a session_id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question to send to Claude Code.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to resume a previous conversation. "
                    "Use 'claude_code_sessions' tool to list available sessions.",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (e.g. 'opus', 'sonnet'). Default: opus.",
                    "enum": ["opus", "sonnet", "haiku"],
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for Claude Code. Default: current workspace.",
                },
            },
            "required": ["prompt"],
        }

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(
        self,
        prompt: str,
        session_id: str | None = None,
        model: str = "opus",
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        claude_bin = shutil.which("claude")
        if not claude_bin:
            return "Error: Claude Code CLI not found. Install it: npm install -g @anthropic-ai/claude-code"

        cmd = [
            claude_bin, "-p",
            "--output-format", "json",
            "--model", model,
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence" if not session_id else None,
        ]
        # Remove None entries
        cmd = [c for c in cmd if c is not None]

        if session_id:
            cmd.extend(["--resume", session_id])

        cmd.append(prompt)

        cwd = working_dir or (str(self._workspace) if self._workspace else None)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=300  # 5 min timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: Claude Code timed out after 5 minutes."
        except Exception as e:
            return f"Error running Claude Code: {e}"

        output = stdout.decode("utf-8", errors="replace").strip()

        if not output:
            err = stderr.decode("utf-8", errors="replace").strip()
            return f"Error: Claude Code returned no output. stderr: {err[:500]}"

        # Parse JSON result
        try:
            result = json.loads(output)
            text = result.get("result", "")
            session = result.get("session_id", "")
            cost = result.get("total_cost_usd", 0)
            turns = result.get("num_turns", 0)
            is_error = result.get("is_error", False)

            parts = []
            if is_error:
                parts.append(f"[Claude Code Error] {text}")
            else:
                parts.append(text)

            meta = []
            if session:
                meta.append(f"session: {session}")
            if turns > 1:
                meta.append(f"turns: {turns}")
            if cost:
                meta.append(f"cost: ${cost:.4f}")
            if meta:
                parts.append(f"\n---\n_[{', '.join(meta)}]_")

            return "\n".join(parts)
        except json.JSONDecodeError:
            # Return raw output if not valid JSON
            return output[:5000]


class ClaudeCodeSessionsTool(Tool):
    """List available Claude Code sessions for resuming."""

    @property
    def name(self) -> str:
        return "claude_code_sessions"

    @property
    def description(self) -> str:
        return (
            "List recent Claude Code sessions. Use this to find a session_id "
            "to pass to the claude_code tool for resuming a conversation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_filter": {
                    "type": "string",
                    "description": "Filter sessions by project name (substring match).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max sessions to return. Default: 10.",
                },
            },
        }

    async def execute(
        self,
        project_filter: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        import glob
        from datetime import datetime
        from pathlib import Path

        projects_dir = Path.home() / ".claude" / "projects"
        if not projects_dir.exists():
            return "No Claude Code sessions found."

        import os

        sessions = []
        for proj in os.listdir(projects_dir):
            proj_path = projects_dir / proj
            if not proj_path.is_dir():
                continue
            if project_filter and project_filter.lower() not in proj.lower():
                continue

            for f in glob.glob(str(proj_path / "*.jsonl")):
                fp = Path(f)
                session_id = fp.stem
                mtime = fp.stat().st_mtime
                lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()

                # Get first user message
                summary = ""
                for line in lines[:20]:
                    try:
                        msg = json.loads(line)
                        if msg.get("type") == "user":
                            inner = msg.get("message", msg)
                            content = inner.get("content", "")
                            if isinstance(content, str):
                                summary = content.replace("\n", " ").strip()[:80]
                            elif isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        summary = c["text"].replace("\n", " ").strip()[:80]
                                        break
                            if summary:
                                break
                    except Exception:
                        pass

                project_name = proj.replace("-Users-zcc-", "~/").replace("-", "/")
                sessions.append({
                    "id": session_id,
                    "project": project_name,
                    "messages": len(lines),
                    "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                    "summary": summary,
                })

        sessions.sort(key=lambda x: x["modified"], reverse=True)
        sessions = sessions[:limit]

        if not sessions:
            return "No matching sessions found."

        lines_out = ["| # | Session ID | Modified | Msgs | Project | Summary |",
                     "|---|-----------|----------|------|---------|---------|"]
        for i, s in enumerate(sessions, 1):
            lines_out.append(
                f"| {i} | `{s['id'][:8]}…` | {s['modified']} | {s['messages']} | {s['project'][-25:]} | {s['summary'][:40]} |"
            )

        return "\n".join(lines_out)
