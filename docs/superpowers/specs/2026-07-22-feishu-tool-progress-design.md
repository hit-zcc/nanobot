# Feishu Tool Progress Message Design

## Goal

Replace repeated long-running tool heartbeat messages in Feishu with one message that is updated in place. Keep the active tool names visible, but present them as readable Chinese labels instead of raw internal identifiers.

## User Experience

- Do not show progress for tasks that finish before the existing 20-second grace period.
- On the first heartbeat, create one progress message:
  - One tool: `⏳ 正在执行命令 · 已用时 20 秒`
  - Multiple tools: `⏳ 正在处理：读取文件、执行命令 · 已用时 20 秒`
- On later heartbeats, update that same message with the new elapsed time and current pending tools.
- When the task finishes, update the same message:
  - One tool: `✅ 执行命令完成 · 共用时 4 分 23 秒`
  - Multiple tools: `✅ 读取文件、执行命令已完成 · 共用时 4 分 23 秒`
- Never create a completion message if no heartbeat message was created.

## Tool Labels

Use a focused mapping for common built-in tools:

- `exec` → `执行命令`
- `read_file` → `读取文件`
- `write_file` → `写入文件`
- `edit_file` → `编辑文件`
- `list_dir` → `浏览目录`
- `web_search` → `搜索网页`
- `web_fetch` → `获取网页`
- `message` → `发送消息`
- `spawn` → `启动子任务`
- `cron` → `管理定时任务`

Unknown tool names remain visible and are rendered as inline code, so plugin and MCP tools are not hidden or mislabeled.

## Message Lifecycle

The agent loop assigns a stable progress ID to each inbound turn and attaches it to heartbeat metadata. The Feishu channel stores the Feishu `message_id` for that progress ID:

1. First heartbeat: create a normal text message and cache its `message_id`.
2. Later heartbeat: patch the cached message instead of creating another message.
3. Completion event: patch the message into its completed state and remove the cache entry.
4. No cached entry: completion is a no-op.

Progress state is keyed by task ID rather than chat ID so concurrent tasks cannot overwrite one another.

## Failure Handling

- If initial message creation fails, log the failure and retain no progress state.
- If an update fails, log it and do not create a replacement message, preventing duplicate-message spam.
- Completion cleanup removes in-memory state even if the final update fails.
- Other channels continue using the existing progress behavior; this change is intentionally scoped to Feishu.

## Testing

Add tests that prove:

- tool names and elapsed time are formatted readably;
- the first heartbeat creates one Feishu message;
- later heartbeats patch that message without creating another;
- completion patches and clears the cached state;
- completion without a prior heartbeat sends nothing;
- a patch failure does not fall back to creating another message;
- existing runner heartbeat timing and the full test suite remain green.
