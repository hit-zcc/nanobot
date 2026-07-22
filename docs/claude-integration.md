# nanobot 调用 Claude 的完整实现说明

> 目的：完整描述 nanobot 是如何调用 Claude 的，包括认证（API Key / Claude OAuth 订阅）、
> 消息格式转换、prompt caching、extended thinking、流式输出、重试和 agentic 工具循环，
> 供在其他项目中复刻这套实现。
>
> 对应源码：
> - `nanobot/providers/anthropic_provider.py` — Anthropic SDK 封装（核心）
> - `nanobot/providers/claude_oauth_provider.py` — Claude Pro/Max 订阅 OAuth 认证
> - `nanobot/providers/base.py` — 抽象基类、重试、日志
> - `nanobot/agent/runner.py` — agentic 工具循环
> - `nanobot/providers/registry.py` — provider 路由

---

## 1. 总体架构

```
用户消息（OpenAI chat 格式，项目内部统一格式）
        │
        ▼
AgentRunner.run()                    ← agentic 循环：LLM ↔ 工具，直到没有 tool_calls
        │  provider.chat_stream_with_retry(...)
        ▼
LLMProvider (base.py)                ← 重试（1s/2s/4s）、去图降级、调用日志
        │  self.chat_stream(...)
        ▼
AnthropicProvider                    ← OpenAI 格式 → Anthropic Messages API 格式转换
  └─ ClaudeOAuthProvider（子类）      ← 换认证方式：Bearer OAuth token + Claude Code 伪装头
        │  AsyncAnthropic().messages.create / .stream
        ▼
api.anthropic.com  /v1/messages
```

关键设计决策：

1. **内部统一用 OpenAI chat 格式**（`role: system/user/assistant/tool` + `tool_calls`），
   在 provider 层做一次性转换成 Anthropic 格式。好处是 agent 循环、会话存储、其他
   provider（OpenAI/DeepSeek/…）全部共用一套消息结构。
2. **OAuth 是 `AnthropicProvider` 的子类**，只改三件事：SDK client 的认证方式、
   模型名前缀剥离、以及强制注入一个 system 前缀（见 §3.3）。请求构建/解析逻辑完全复用。
3. **错误不抛异常**，统一收敛成 `LLMResponse(finish_reason="error", content="Error calling LLM: ...")`，
   由上层根据错误文本判断是否可重试。

依赖：`anthropic`（官方 SDK，用 `AsyncAnthropic`）、`httpx`（OAuth token 交换）、
`json_repair`（容错解析 tool 参数 JSON）。

---

## 2. 数据结构（`base.py`）

```python
@dataclass
class ToolCallRequest:
    id: str                      # Anthropic 的 tool_use id（toolu_xxx）
    name: str
    arguments: dict[str, Any]    # 已解析成 dict
    def to_openai_tool_call(self) -> dict: ...   # 回写历史时序列化为 OpenAI 格式

@dataclass
class LLMResponse:
    content: str | None                  # 所有 text block 拼接
    tool_calls: list[ToolCallRequest]
    finish_reason: str                   # "stop" | "tool_calls" | "length" | "refusal" | "error"
    usage: dict[str, int]                # prompt/completion/total + cache_*_input_tokens
    thinking_blocks: list[dict] | None   # Anthropic thinking block 原样保存（含 signature）
    response_model: str | None           # API 回显的真实模型名
```

`thinking_blocks` 是关键：**Anthropic 要求多轮对话中把上一轮 assistant 的 thinking block
原封不动传回**（含 `signature` 字段），否则带工具调用的 thinking 会话会报 400。
所以响应解析时保存，回写历史时塞进 assistant 消息，下次转换时再还原成 thinking block。

---

## 3. 认证

### 3.1 API Key 方式（`AnthropicProvider.__init__`）

标准做法，直接给 SDK：

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    api_key=api_key,              # ANTHROPIC_API_KEY
    base_url=api_base,            # 可选，默认 https://api.anthropic.com
    default_headers=extra_headers # 可选
)
```

### 3.2 Claude OAuth 方式（Pro/Max 订阅，不用 API Key）

复用 Claude Code 的公开 OAuth client 走 **PKCE 授权码流程**。常量：

```python
CLAUDE_CLIENT_ID    = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"   # Claude Code 公开 client
CLAUDE_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URL     = "https://platform.claude.com/v1/oauth/token"
CLAUDE_API_BASE      = "https://api.anthropic.com"
CLAUDE_SCOPES = "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
```

**登录流程（`login_claude_oauth`）：**

1. 生成 PKCE：`code_verifier = secrets.token_urlsafe(64)`，
   `code_challenge = base64url(sha256(verifier))`（S256），再加一个随机 `state`。
2. 在 `127.0.0.1` 随机端口起一个一次性 HTTP server 接收回调
   （`redirect_uri = http://localhost:{port}/callback`）。
3. 打开浏览器访问 `{AUTHORIZE_URL}?response_type=code&client_id=...&redirect_uri=...&scope=...&state=...&code_challenge=...&code_challenge_method=S256`。
4. 回调里校验 `state`，拿到 `code`。
5. POST `CLAUDE_TOKEN_URL`（JSON body）换 token：

   ```json
   {
     "grant_type": "authorization_code",
     "code": "...", "redirect_uri": "...",
     "client_id": "...", "code_verifier": "...", "state": "..."
   }
   ```

6. 持久化到 `~/.nanobot/auth/claude-oauth.json`：

   ```json
   {"access_token": "...", "refresh_token": "...", "expires_at": 1751234567.0, "token_type": "Bearer"}
   ```

**token 读取有一个兜底**：如果本项目没登录过，会去读 Claude Code 自己的凭据文件
`~/.claude/.credentials.json`（格式 `{"claudeAiOauth": {"accessToken", "refreshToken", "expiresAt(毫秒)"}}`），
这样已经在本机登录过 `claude` 的用户可以免登录直接用。

**刷新（`_refresh_access_token` + `_ensure_token`）：**

- 过期前 120 秒（`_EXPIRY_SKEW_SECONDS`）就提前刷新；
- 刷新是 POST 同一个 token URL：`{"grant_type": "refresh_token", "refresh_token": ..., "client_id": ..., "scope": CLAUDE_SCOPES}`；
- `_ensure_token()` 在**每次请求前**被调用（`_inject_token`），把有效 token 写回
  `self._client.auth_token`——SDK 会把它放到 `Authorization: Bearer <token>` 头。

**SDK client 的构造（这是 OAuth 能被 api.anthropic.com 接受的关键）：**

```python
self._client = AsyncAnthropic(
    auth_token=access_token,           # 走 Authorization: Bearer，而不是 x-api-key
    base_url="https://api.anthropic.com",
    default_headers={
        "anthropic-beta": "oauth-2025-04-20",        # 必须：OAuth beta 头
        "x-app": "cli",
        "User-Agent": "claude-code/2.1.90 nanobot",
        "X-Claude-Code-Session-Id": secrets.token_hex(16),
    },
)
```

### 3.3 OAuth 的 system 前缀约束（最容易踩的坑）

api.anthropic.com **只在第一个 system block 恰好是这句话时才接受 Pro/Max OAuth token**：

```
You are Claude Code, Anthropic's official CLI for Claude.
```

实现方式：`AnthropicProvider` 留了一个钩子 `_system_prefix()`（默认 `None`），
OAuth 子类返回上面这个字符串；`_build_kwargs` 里用 `_prepend_system_prefix` 把它插到
system block 列表最前面（幂等——重试不会叠加重复前缀），你自己的 system prompt
作为第二个 block 跟在后面。API Key 路径完全不受影响。

### 3.4 OAuth 专属的 429 退避

OAuth 订阅额度更容易撞限流，所以 `ClaudeOAuthProvider._call_with_retry` 在 base 层
通用重试之外，又包了一层针对 429 的指数退避：5s → 10s → 20s → 40s → 60s，最多 5 次，
每次重试前都重新 `_inject_token()`（token 可能在等待期间过期）。

---

## 4. 请求构建（`_build_kwargs`）

入参是 OpenAI 格式的 `messages` + `tools`，出参是可以直接
`client.messages.create(**kwargs)` 的字典。步骤依次是：

### 4.1 消息清洗（`_sanitize_empty_content`，base.py）

- 空字符串 content：assistant 且带 tool_calls → `None`；否则填 `"(empty)"`
  （Anthropic 不接受空 text block）；
- content 列表里的空 text block 直接删掉；
- 剥掉项目内部的 `_meta` 字段。

### 4.2 OpenAI → Anthropic 消息转换（`_convert_messages`）

| OpenAI 输入 | Anthropic 输出 |
|---|---|
| `role: "system"` | 提出来放到顶层 `system` 参数（字符串或 block 列表） |
| `role: "user"`，content 里的 `image_url` block | `{"type":"image","source":{...}}`：`data:image/png;base64,...` 正则拆成 base64 source；普通 URL 用 url source |
| `role: "assistant"` | block 列表，顺序：先还原 `thinking_blocks`（thinking + signature 原样）→ `text` → 每个 `tool_calls` 转成 `{"type":"tool_use","id","name","input"}`（arguments 字符串用 `json_repair.loads` 容错解析；缺 id 时生成 `toolu_` + 22 位随机串） |
| `role: "tool"` | `{"type":"tool_result","tool_use_id": msg["tool_call_id"],"content":...}`，**并入上一条 user 消息的 content**（没有就新建一条 user） |

最后 `_merge_consecutive`：Anthropic 要求 user/assistant 严格交替，
把连续同角色消息的 content 合并成一个 block 列表。

### 4.3 工具定义转换（`_convert_tools`）

OpenAI 的 `{"type":"function","function":{name,description,parameters}}` →
Anthropic 的 `{"name","description","input_schema"}`，`parameters` 直接改名
`input_schema`（JSON Schema 本身兼容）。

`tool_choice` 映射：`"auto"`/None → `{"type":"auto"}`；`"required"` → `{"type":"any"}`；
`"none"` → 不传；`{"function":{"name":X}}` → `{"type":"tool","name":X}`。
**thinking 开启时强制 `{"type":"auto"}`**（Anthropic 限制：extended thinking 不兼容
强制 tool_choice）。

### 4.4 Prompt caching（`_apply_cache_control`）

用 3 个 `cache_control: {"type":"ephemeral"}` 断点（API 上限 4 个）：

1. **system 最后一个 block** —— 缓存 tools + system（渲染顺序 tools→system→messages，
   断点在 system 上会连 tools 一起盖住）；
2. **倒数第二条消息的最后一个 block**（消息数 ≥3 时）—— 多轮对话里让上一轮的完整
   历史前缀命中缓存；
3. **tools 列表最后一个**。

命中效果可以从响应 `usage.cache_read_input_tokens` / `cache_creation_input_tokens` 验证。

### 4.5 Extended thinking（`reasoning_effort`）

项目自己的 `reasoning_effort: "low"|"medium"|"high"` 映射成 thinking 预算：

```python
budget_map = {"low": 1024, "medium": 4096, "high": max(8192, max_tokens)}
kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
kwargs["max_tokens"] = max(max_tokens, budget + 4096)   # budget 必须 < max_tokens
kwargs["temperature"] = 1.0                              # thinking 要求 temperature=1
```

> ⚠️ **移植注意**：`{"type":"enabled","budget_tokens":N}` 只适用于老一代模型
> （本项目默认 `claude-sonnet-4-20250514`）。Opus 4.6/Sonnet 4.6 已弃用，
> Opus 4.7/4.8、Sonnet 5、Fable 5 会直接 400，要改用 `{"type":"adaptive"}`
> （Fable 5 甚至连 `disabled` 都不接受，直接不传）。新项目建议按模型分支处理。

### 4.6 temperature 的模型分支（`_supports_temperature`）

模型名包含 `fable` / `mythos` / `opus-4-8` 时**不发送 temperature**（这些模型收到
temperature 会 400）；其他模型正常透传（thinking 开启时固定 1.0）。

### 4.7 最终 kwargs 形态

```python
{
    "model": "claude-sonnet-4-20250514",     # 已剥掉 "anthropic/"、"claude-oauth/" 前缀
    "system": [{"type":"text","text":"...","cache_control":{...}}, ...],
    "messages": [...],                        # Anthropic 格式
    "max_tokens": 8192,
    "temperature": 0.7,                       # 或省略 / 1.0
    "thinking": {...},                        # 可选
    "tools": [...], "tool_choice": {...},     # 可选
    "extra_headers": {...},                   # 可选
}
```

---

## 5. 发起调用

### 5.1 非流式（`chat`）

```python
response = await self._client.messages.create(**kwargs)
return self._parse_response(response)
```

### 5.2 流式（`chat_stream`）

```python
async with self._client.messages.stream(**kwargs) as stream:
    if on_content_delta:
        async for text in stream.text_stream:     # 只订阅文本增量
            await on_content_delta(text)          # 逐段推给 UI（如飞书流式卡片）
    response = await stream.get_final_message()   # 拿完整 Message（含 tool_use/thinking/usage）
return self._parse_response(response)
```

流式和非流式**共用同一套构建和解析**：`stream.get_final_message()` 返回的对象与
`messages.create()` 完全同构，所以 tool 调用、thinking、usage 的处理不需要写两份。
`text_stream` 只产出 text delta，thinking/tool 参数增量被 SDK 自动累积。

两个方法都把异常吞掉转成 `LLMResponse(finish_reason="error")`，让上层统一处理。

---

## 6. 响应解析（`_parse_response`）

遍历 `response.content` 的 block：

- `text` → 拼进 `content`；
- `tool_use` → `ToolCallRequest(id, name, input)`；
- `thinking` → 原样存 `{"type":"thinking","thinking":...,"signature":...}`。

`stop_reason` 映射成 OpenAI 风格：`tool_use→tool_calls`、`end_turn→stop`、
`max_tokens→length`，其余（含 `refusal`）原样透传。

`usage` 转成 `prompt_tokens/completion_tokens/total_tokens`，并附带
`cache_creation_input_tokens`、`cache_read_input_tokens`（非零时）。

---

## 7. 重试层（`base.py`）

`chat_with_retry` / `chat_stream_with_retry` 是所有调用方的实际入口：

1. 未显式传参时从 `self.generation`（`GenerationSettings`）取默认
   temperature/max_tokens/reasoning_effort；
2. 调用 `_safe_chat(_stream)`，任何异常 → error 响应；
3. `finish_reason == "error"` 时看错误文本是否含 transient 标记
   （`429 / rate limit / 5xx / overloaded / timeout / connection / ...`）：
   - transient → 按 `(1, 2, 4)` 秒退避重试，用完后再打最后一次；
   - 非 transient 且消息里有图片 → **把图片替换成 `[image: path]` 占位文本重试一次**
     （处理图片过大/格式不受支持类错误）；
   - 否则直接返回错误。
4. 每次返回前 `_emit_log`：fire-and-forget 把请求参数、响应、耗时写进
   `llm_logger`（永不抛错，不影响主流程）。

注意 SDK 本身也自带 2 次 429/5xx 重试，这层是叠加在其上的应用级重试。

---

## 8. Agentic 工具循环（`runner.py`）

```python
for iteration in range(max_iterations):
    response = await provider.chat_stream_with_retry(messages=messages, tools=defs, model=model)

    if response.finish_reason == "refusal":
        # 安全拒答必须在执行工具之前处理（拒答会截断 turn，可能带着残缺 tool_use）
        # 第一次拒答：换 refusal_fallback_model 透明重试一轮；再拒 → 给用户拒答文案
        ...

    if response.has_tool_calls:
        messages.append(assistant 消息(content, tool_calls=[tc.to_openai_tool_call()...],
                                       thinking_blocks=response.thinking_blocks))  # ← thinking 回传
        results = await 并发执行工具(response.tool_calls)
        for tc, result in zip(response.tool_calls, results):
            messages.append({"role":"tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
        continue                       # 带着工具结果进入下一轮

    # 没有 tool_calls → 终局回复
    messages.append(assistant 消息(content, thinking_blocks=...))
    break
```

要点：

- 历史里 assistant 消息**必须带上 `tool_calls` 和 `thinking_blocks`**，下一轮
  `_convert_messages` 才能还原出合法的 `tool_use` / `thinking` block；
- 每个 `tool_result` 的 `tool_use_id` 必须精确对应；转换层会把同一轮的多个
  tool 结果合并进一条 user 消息（Anthropic 要求并行工具结果在同一条 user 消息里）;
- `max_iterations` 兜底防止死循环。

---

## 9. Provider 路由（`registry.py`）

按模型名关键字选 provider：模型名含 `claude-oauth`/`claude_oauth` → `ClaudeOAuthProvider`
（`is_oauth=True`，不需要 API key）；含 `anthropic`/`claude` → `AnthropicProvider`
（需要 `ANTHROPIC_API_KEY`）。两者都标 `supports_prompt_caching=True`。
模型名形如 `claude-oauth/claude-sonnet-4-20250514`，provider 内部剥前缀。

---

## 10. 移植清单（在新项目里最少要做对的事)

1. **SDK**：用官方 `anthropic` 包的 `AsyncAnthropic`，不要手写 HTTP。
2. **认证二选一**：
   - API key：`AsyncAnthropic(api_key=...)`；
   - OAuth：`AsyncAnthropic(auth_token=..., default_headers={"anthropic-beta": "oauth-2025-04-20", ...})`
     + 每次请求前检查/刷新 token + **第一个 system block 必须是 Claude Code 身份句**。
3. **thinking block 回传**：解析时保存（含 signature），写回历史时原样还原，一字不改。
4. **tool_result 归位**：`tool_use_id` 对应、同轮多个结果并进一条 user 消息、
   user/assistant 严格交替（连续同角色要合并）。
5. **空 content 处理**：Anthropic 拒绝空 text block，要么删 block 要么填占位符。
6. **caching 断点**：system 末尾 + 倒数第二条消息末尾 + tools 末尾；保持前缀字节稳定
   （system 里不要插时间戳等易变内容），用 `usage.cache_read_input_tokens` 验证。
7. **重试分层**：应用层对 transient 错误退避重试；OAuth 对 429 单独做更长退避；
   错误统一收敛成结构化响应而不是异常。
8. **新模型适配**（本项目尚未完全覆盖，移植时建议直接做对）：
   - Opus 4.7/4.8、Sonnet 5、Fable 5：`thinking` 用 `{"type":"adaptive"}`，
     不发 `budget_tokens`，不发 `temperature/top_p/top_k`；
   - Fable 5：不传 `thinking` 字段即可（显式 disabled 会 400），并处理
     `stop_reason == "refusal"`（本项目 runner 里的 fallback-model 重试就是为此准备的）。
