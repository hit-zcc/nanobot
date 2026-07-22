# Feishu Tool Progress Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show one readable Feishu tool-progress message per agent turn, update it in place, and convert it to a completed state when the turn finishes.

**Architecture:** `AgentLoop` owns user-facing tool labels, elapsed-time formatting, and per-turn progress lifecycle. It publishes progress events with a stable ID; `FeishuChannel` owns the Feishu message ID cache and uses the IM Patch Message API for in-place updates. Existing non-Feishu progress delivery remains unchanged.

**Tech Stack:** Python 3.11+, asyncio, pytest, lark-oapi Feishu SDK

## Global Constraints

- Keep the existing 20-second heartbeat grace period and backoff schedule.
- Keep active tool names visible using readable Chinese labels; unknown names remain visible as inline code.
- Create no progress message for a task that finishes before its first heartbeat.
- Do not create a replacement message when a Feishu patch fails.
- Scope in-place editing to Feishu; preserve existing behavior for other channels.

---

### Task 1: Format and publish one logical progress lifecycle

**Files:**
- Modify: `nanobot/agent/loop.py`
- Test: `tests/agent/test_runner.py`

**Interfaces:**
- Produces: `AgentLoop._format_duration(seconds: float) -> str`
- Produces: `AgentLoop._format_tool_names(names: list[str]) -> list[str]`
- Produces: optional `_run_agent_loop(..., on_tool_progress: Callable[..., Awaitable[None]] | None = None)` callback receiving `content`, `progress_id`, and `done`.
- Consumes: existing `AgentHook.on_tool_heartbeat(..., elapsed, pending)` callbacks from `AgentRunner`.

- [ ] **Step 1: Write failing formatter and lifecycle tests**

Add tests asserting:

```python
def test_tool_progress_formatting(tmp_path):
    loop = _make_loop(tmp_path)
    assert loop._format_duration(20) == "20 秒"
    assert loop._format_duration(263) == "4 分 23 秒"
    assert loop._format_duration(3661) == "1 小时 1 分 1 秒"
    assert loop._format_tool_names(["exec", "read_file"]) == ["执行命令", "读取文件"]
    assert loop._format_tool_names(["custom_tool"]) == ["`custom_tool`"]
```

Extend the existing slow-tool loop test with an `on_tool_progress` recorder. Assert the first event is `done=False`, the final event is `done=True`, both share one non-empty `progress_id`, and the completion text uses `✅`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/agent/test_runner.py -k 'tool_progress or tool_heartbeat' -v`

Expected: FAIL because the formatter methods and `on_tool_progress` argument do not exist.

- [ ] **Step 3: Implement formatting and lifecycle publishing**

In `AgentLoop`, add a focused mapping and helpers:

```python
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
```

Add `on_tool_progress` to `_run_agent_loop`. The nested hook stores one UUID-like progress ID, the first heartbeat start time (`time.monotonic() - elapsed`), and the union of announced tools. Heartbeats use `正在… · 已用时…`; after `runner.run(...)` returns, emit one `done=True` event only if a heartbeat was previously emitted.

In `_process_message`, publish these events as `OutboundMessage` metadata:

```python
{
    **msg.metadata,
    "_progress": True,
    "_tool_progress_id": progress_id,
    "_tool_progress_done": done,
}
```

For non-Feishu channels, discard the `done=True` event so their current repeated-heartbeat behavior is unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/agent/test_runner.py -k 'tool_progress or tool_heartbeat' -v`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the logical progress lifecycle**

```bash
git add nanobot/agent/loop.py tests/agent/test_runner.py
git commit -m "feat(agent): format tool progress lifecycle"
```

---

### Task 2: Update one Feishu progress message in place

**Files:**
- Modify: `nanobot/channels/feishu.py`
- Test: `tests/channels/test_feishu_reply.py`
- Test: `tests/channels/test_feishu_streaming.py`

**Interfaces:**
- Consumes metadata `_tool_progress_id: str` and `_tool_progress_done: bool` from Task 1.
- Produces: `FeishuChannel._patch_message_sync(message_id: str, content: str) -> bool`.
- Produces: `FeishuChannel._reply_message_with_id_sync(parent_message_id: str, msg_type: str, content: str) -> str | None` while preserving `_reply_message_sync(...) -> bool`.
- Maintains: `self._tool_progress_messages: dict[str, str]`, mapping logical progress IDs to Feishu message IDs.

- [ ] **Step 1: Write failing Feishu create/update/completion tests**

Add tests that construct `OutboundMessage` instances with `_tool_progress_id` and verify:

```python
await channel.send(first_heartbeat)
await channel.send(second_heartbeat)
await channel.send(completion)

assert channel._client.im.v1.message.create.call_count == 1
assert channel._client.im.v1.message.patch.call_count == 2
assert "progress-1" not in channel._tool_progress_messages
```

Also assert completion without cached state creates and patches nothing, patch failure does not call create again, and a topic-thread first heartbeat uses Reply Message while still caching the returned reply message ID.

- [ ] **Step 2: Run focused Feishu tests and verify RED**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/channels/test_feishu_reply.py tests/channels/test_feishu_streaming.py -k 'progress or reply_message_with_id' -v`

Expected: FAIL because the cache, patch helper, and specialized send path do not exist.

- [ ] **Step 3: Implement Feishu message creation and patching**

Initialize `self._tool_progress_messages = {}`. Add a Reply API helper that returns `response.data.message_id`, retain the old boolean wrapper, and add:

```python
def _patch_message_sync(self, message_id: str, content: str) -> bool:
    from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody
    body = json.dumps({"text": content}, ensure_ascii=False)
    request = (
        PatchMessageRequest.builder()
        .message_id(message_id)
        .request_body(PatchMessageRequestBody.builder().content(body).build())
        .build()
    )
    response = self._client.im.v1.message.patch(request)
    return bool(response.success())
```

At the beginning of `send`, route messages with `_tool_progress_id` to a new async helper. It creates and caches the first text message, patches later messages, pops state on completion, respects topic-thread reply routing, and never creates a replacement after patch failure.

- [ ] **Step 4: Run focused Feishu tests and verify GREEN**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/channels/test_feishu_reply.py tests/channels/test_feishu_streaming.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit Feishu in-place updates**

```bash
git add nanobot/channels/feishu.py tests/channels/test_feishu_reply.py tests/channels/test_feishu_streaming.py
git commit -m "feat(feishu): refresh tool progress in place"
```

---

### Task 3: Verify integration and restart the gateway

**Files:**
- No production file changes expected.

**Interfaces:**
- Consumes the completed Task 1 and Task 2 behavior.
- Produces a running gateway using the current checkout.

- [ ] **Step 1: Run diff and focused verification**

Run: `git diff --check`

Expected: exit 0 with no output.

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/agent/test_runner.py tests/channels/test_feishu_reply.py tests/channels/test_feishu_streaming.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Run the complete test suite**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/`

Expected: all tests PASS; existing optional-dependency skips and third-party deprecation warnings may remain.

- [ ] **Step 3: Restart the gateway**

Run: `PATH="$PWD/.venv/bin:$PATH" /Users/zhangcongcong/.nanobot/workspace/scripts/restart-gateway.sh`

Expected: restart script exits 0 and starts one gateway process.

- [ ] **Step 4: Verify the restarted process and fresh log tail**

Run: `ps -ax -o pid=,ppid=,lstart=,command=` and inspect the nanobot gateway process.

Run: `rg -n "Starting nanobot gateway|Channels enabled|Heartbeat started|ERROR|CRITICAL|Traceback" /Users/zhangcongcong/.nanobot/gateway.log | tail -n 40`

Expected: a fresh startup sequence after the restart with Feishu enabled and no new startup error or traceback.

- [ ] **Step 5: Report the result**

Report changed behavior, commit IDs, focused/full test counts, and restart evidence. Do not push unless explicitly requested.
