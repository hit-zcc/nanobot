"""Generate a local-only Claude Code reference vs nanobot request snapshot.

Disconnected from provider registry and production call paths. It never loads
credentials. HTTP capture uses MockTransport after a localhost-only URL guard.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx

from nanobot.providers import claude_oauth_provider as oauth_module
from nanobot.providers.claude_oauth_provider import ClaudeOAuthProvider

DISCLAIMER = "仅本地研究快照，不代表完整 Claude Code 实现，不得用于真实 OAuth 请求。"
MOCK_TOKEN = "mock-token-not-for-network"
LOCAL_CAPTURE_URL = "http://127.0.0.1/v1/messages"
TEST_SESSION_UUID = "123e4567-e89b-42d3-a456-426614174000"
CC_VERSION = "2.1.241"
IDENTITIES = {
    "interactive": "You are Claude Code, Anthropic's official CLI for Claude.",
    "noninteractive_append": (
        "You are Claude Code, Anthropic's official CLI for Claude, running within "
        "the Claude Agent SDK."
    ),
    "noninteractive_plain": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
}
REFERENCE_HEADERS = {
    "anthropic-beta": "oauth-2025-04-20",
    "x-app": "cli",
    "User-Agent": f"claude-cli/{CC_VERSION} (external, cli)",
    "X-Claude-Code-Session-Id": TEST_SESSION_UUID,
    "Authorization": f"Bearer {MOCK_TOKEN}",
}
FIXTURE_MESSAGES = [
    {"role": "system", "content": "snapshot-test-system-content"},
    {"role": "user", "content": "snapshot-test-user-content"},
    {"role": "assistant", "content": "snapshot-test-assistant-content"},
    {"role": "user", "content": "snapshot-test-follow-up"},
]
FIXTURE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "snapshot_test_tool",
        "description": "Local snapshot fixture only.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}]


def guard_local_url(url: str) -> None:
    """Reject every target except plain HTTP on an explicit loopback host."""
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"network guard rejected non-local target: {parsed.hostname or url}")
    if parsed.username or parsed.password:
        raise ValueError("network guard rejected URL credentials")


def extract_current_provider_headers() -> dict[str, str]:
    """Statically extract AsyncAnthropic(default_headers=...) from current source."""
    tree = ast.parse(inspect.getsource(oauth_module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "AsyncAnthropic":
            continue
        for keyword in node.keywords:
            if keyword.arg != "default_headers" or not isinstance(keyword.value, ast.Dict):
                continue
            headers: dict[str, str] = {}
            pairs = zip(keyword.value.keys, keyword.value.values, strict=True)
            for key_node, value_node in pairs:
                key = ast.literal_eval(key_node)
                if not isinstance(key, str):
                    raise RuntimeError("non-string default header key")
                if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                    headers[key] = value_node.value
                elif key == "X-Claude-Code-Session-Id":
                    headers[key] = "<32-hex session id>"
                else:
                    raise RuntimeError(f"unsupported dynamic default header: {key}")
            return headers
    raise RuntimeError("could not locate Claude OAuth default_headers")


def build_nanobot_body() -> dict[str, Any]:
    """Call the real _build_kwargs without running provider __init__."""
    provider = object.__new__(ClaudeOAuthProvider)
    provider.default_model = "claude-sonnet-4-20250514"
    provider.extra_headers = {}
    return provider._build_kwargs(
        messages=deepcopy(FIXTURE_MESSAGES), tools=deepcopy(FIXTURE_TOOLS),
        model="claude-oauth/claude-sonnet-4-20250514", max_tokens=8192,
        temperature=0.2, reasoning_effort="medium", tool_choice="auto",
        supports_caching=True,
    )


def capture_with_mock_transport(
    body: dict[str, Any], headers: dict[str, str], url: str = LOCAL_CAPTURE_URL,
) -> dict[str, Any]:
    """Capture a request in-memory; MockTransport creates no socket."""
    guard_local_url(url)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        guard_local_url(str(request.url))
        wanted = {name.lower() for name in headers}
        captured.update({
            "method": request.method,
            "url": str(request.url),
            "target_host": request.url.host,
            "headers": {k: v for k, v in request.headers.items() if k.lower() in wanted},
            "body": json.loads(request.content),
            "transport": "httpx.MockTransport (no socket)",
        })
        return httpx.Response(200, json={"captured": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        client.post(url, headers=headers, json=body).raise_for_status()
    return captured


def _session_shape(value: str) -> str:
    if value == "<32-hex session id>" or re.fullmatch(r"[0-9a-f]{32}", value):
        return "32 lowercase hex characters"
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return "other"
    return "UUID"


def _first_system_text(body: dict[str, Any]) -> str | None:
    system = body.get("system")
    if isinstance(system, list) and system and isinstance(system[0], dict):
        return system[0].get("text")
    return system if isinstance(system, str) else None


def _cache_locations(body: dict[str, Any]) -> list[str]:
    return [
        section for section in ("system", "messages", "tools")
        if "cache_control" in json.dumps(body.get(section), sort_keys=True)
    ]


def build_snapshot(modes: list[str] | None = None) -> dict[str, Any]:
    selected = modes or list(IDENTITIES)
    unknown = set(selected) - set(IDENTITIES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    body = build_nanobot_body()
    provider_headers = extract_current_provider_headers()
    capture = capture_with_mock_transport(
        body, {**provider_headers, "Authorization": f"Bearer {MOCK_TOKEN}"}
    )
    dimensions = {
        "system_first_block": _first_system_text(body),
        "headers": provider_headers,
        "authorization": "Bearer <hard-coded mock token>",
        "model": body.get("model"), "messages": body.get("messages"),
        "tools": body.get("tools"),
        "session_id_shape": _session_shape(provider_headers["X-Claude-Code-Session-Id"]),
        "thinking": body.get("thinking"),
        "cache_control_locations": _cache_locations(body),
    }
    references: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for mode in selected:
        reference = {
            "scope": "verified static reference subset from local Claude Code 2.1.241 facts",
            "completeness": "not a complete Claude Code request",
            "identity": IDENTITIES[mode], "headers": REFERENCE_HEADERS,
            "session_id_shape": _session_shape(TEST_SESSION_UUID),
            "model": "not captured", "messages": "not captured", "tools": "not captured",
            "thinking": "not captured", "cache_control": "not captured",
        }
        references[mode] = reference
        checks = {
            "system_first_block": dimensions["system_first_block"] == reference["identity"],
            "x_app": provider_headers.get("x-app") == reference["headers"].get("x-app"),
            "user_agent": provider_headers.get("User-Agent") == reference["headers"].get("User-Agent"),
            "oauth_beta": provider_headers.get("anthropic-beta") == reference["headers"].get("anthropic-beta"),
            "session_id_shape": dimensions["session_id_shape"] == reference["session_id_shape"],
        }
        comparisons[mode] = {
            "checks": checks,
            "stats": {"compared": len(checks), "matches": sum(checks.values()),
                      "differences": len(checks) - sum(checks.values()),
                      "unavailable_dimensions": 5},
        }
    return {
        "schema_version": 1, "disclaimer": DISCLAIMER,
        "safety": {
            "network_policy": "localhost only; MockTransport opens no socket",
            "capture_url": LOCAL_CAPTURE_URL, "capture_target_host": capture["target_host"],
            "credential_files_read": [], "token_refresh_or_login": False,
            "mock_token_literal": MOCK_TOKEN,
        },
        "fixture": {"contains_real_user_content": False, "contains_real_session_id": False,
                    "test_session_uuid": TEST_SESSION_UUID},
        "claude_code_static_references": references,
        "nanobot_current": {
            "source": "current source + real _build_kwargs; __init__ bypassed",
            "credentials": "not read; hard-coded mock only",
            "request": capture, "dimensions": dimensions,
        },
        "comparisons": comparisons,
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    d = snapshot["nanobot_current"]["dimensions"]
    lines = [
        "# Claude Code 请求快照 / mock 对比实验", "",
        f"> **{snapshot['disclaimer']}**", "", "## 安全边界", "",
        "- 未读取 Keychain、`~/.claude/.credentials.json` 或 `~/.nanobot/auth`。",
        f"- 鉴权仅使用明显假的 `{MOCK_TOKEN}`。",
        "- 未执行登录、token refresh 或 `claude -p`。",
        "- HTTP 捕获目标为 `127.0.0.1`，由 `httpx.MockTransport` 内存截获，未启 socket。",
        "- 实验模块未接入 provider registry 或生产调用链。", "",
        "## nanobot 当前快照", "",
        f"- system 首块：`{d['system_first_block']}`", f"- model：`{d['model']}`",
        f"- session-id 形态：`{d['session_id_shape']}`",
        "- thinking：`" + json.dumps(d["thinking"], ensure_ascii=False) + "`",
        "- cache-control 分布：`" + ", ".join(d["cache_control_locations"]) + "`", "",
        "## 静态参考子集对比", "",
        "| 模式 | system | x-app | UA | OAuth beta | session形态 | 匹配/对比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, comparison in snapshot["comparisons"].items():
        c, stats = comparison["checks"], comparison["stats"]
        mark = lambda value: "✓" if value else "✗"  # noqa: E731
        lines.append(
            f"| `{mode}` | {mark(c['system_first_block'])} | {mark(c['x_app'])} | "
            f"{mark(c['user_agent'])} | {mark(c['oauth_beta'])} | "
            f"{mark(c['session_id_shape'])} | {stats['matches']}/{stats['compared']} |"
        )
    lines += [
        "", "每种模式另有 5 个维度（model、messages、tools、thinking、cache-control）因本机静态事实",
        "未覆盖而标记为 unavailable；它们仍记录在 nanobot 快照中，但不臆造 Claude Code 值。",
        "表内统计全部由生成程序计算。", "", "## 局限", "",
        "三种模式仅是本机 2.1.241 已核实的身份与基础元数据静态参考子集，不是 Claude Code",
        "完整请求、协议复刻或兼容性证明。结果只能用于离线结构研究，不得用于真实网络请求。", "",
    ]
    return "\n".join(lines)


def write_outputs(snapshot: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    report_path.write_text(render_markdown(snapshot))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_dir = Path.home() / ".nanobot" / "workspace" / "docs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", *IDENTITIES], default="all")
    parser.add_argument("--json-output", type=Path,
                        default=default_dir / "claude-request-snapshot.json")
    parser.add_argument("--report-output", type=Path,
                        default=default_dir / "claude-request-diff.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = build_snapshot(None if args.mode == "all" else [args.mode])
    write_outputs(snapshot, args.json_output, args.report_output)
    print(f"wrote {args.json_output}")
    print(f"wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
