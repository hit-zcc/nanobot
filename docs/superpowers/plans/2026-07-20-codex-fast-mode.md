# Codex Fast Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `serviceTier: "fast"` nanobot setting that sends Codex Responses requests with `service_tier: "priority"` and deploy it without changing medium reasoning or the 640k context window.

**Architecture:** `AgentDefaults` owns the optional user-facing setting. Runtime construction passes it only into `OpenAICodexProvider`, which validates and translates it at the request boundary. Omission preserves existing request bodies; unsupported values fail locally with a fixed safe response.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest/pytest-asyncio, Ruff, Codex Responses streaming provider.

## Global Constraints

- The supported initial values are `"fast"` and omitted/null.
- `"fast"` maps exactly to Responses `service_tier: "priority"`.
- Keep `model=openai-codex/gpt-5.6-sol`, `provider=openai_codex`, `reasoningEffort=medium`, and `contextWindowTokens=655360` unchanged.
- Do not modify `~/.codex/auth.json`, sessions, memory, or unrelated dirty worktree files.
- Omitted `serviceTier` must preserve the current request shape.
- Errors must not include credentials, headers, raw transport messages, or backend bodies.

---

### Task 1: Config and provider request mapping

**Files:**
- Modify: `nanobot/config/schema.py`
- Modify: `nanobot/cli/commands.py`
- Modify: `nanobot/providers/openai_codex_provider.py`
- Test: `tests/config/test_config_migration.py`
- Test: `tests/providers/test_openai_codex_provider.py`
- Test: `tests/cli/test_commands.py`

**Interfaces:**
- Consumes: `AgentDefaults.service_tier: str | None` loaded from `serviceTier`.
- Produces: `OpenAICodexProvider(..., service_tier: str | None)` and `_resolve_service_tier(value: str | None) -> str | None`.

- [ ] **Step 1: Write failing config and provider-boundary tests**

Add tests equivalent to:

```python
def test_config_round_trips_fast_service_tier(tmp_path):
    # Load serviceTier="fast", save, and assert the attribute and JSON retain it.

@pytest.mark.asyncio
async def test_fast_service_tier_sends_priority(monkeypatch):
    provider = OpenAICodexProvider(
        credential_manager=FakeCredentials(), service_tier="fast"
    )
    # Intercept _request_codex and assert body["service_tier"] == "priority".

@pytest.mark.asyncio
async def test_omitted_service_tier_sends_no_field(monkeypatch):
    # Assert "service_tier" not in the intercepted body.

@pytest.mark.asyncio
async def test_unknown_service_tier_fails_before_transport(monkeypatch):
    # Construct with "turbo"; assert fixed safe error and transport not called.
```

Add a runtime-construction test that loads `serviceTier: "fast"` and asserts the constructed Codex provider has the configured setting.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q \
  tests/config/test_config_migration.py \
  tests/providers/test_openai_codex_provider.py \
  tests/cli/test_commands.py
```

Expected: the new tests fail because `service_tier` does not exist in config/provider construction or request bodies.

- [ ] **Step 3: Implement the minimal mapping**

In `AgentDefaults` add:

```python
service_tier: str | None = None
```

Pass `defaults.service_tier` when constructing `OpenAICodexProvider`. Store the value on the provider and resolve it with:

```python
def _resolve_service_tier(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "fast":
        return "priority"
    raise ValueError("Unsupported Codex service tier. Use 'fast' or omit serviceTier")
```

Before transport, add the resolved value only when present:

```python
service_tier = _resolve_service_tier(self.service_tier)
if service_tier:
    body["service_tier"] = service_tier
```

Keep the existing safe exception policy so local validation returns a fixed generic Codex request failure rather than reflecting arbitrary exception text.

- [ ] **Step 4: Run focused tests GREEN**

Run the Step 2 command plus:

```bash
.venv/bin/ruff check \
  nanobot/config/schema.py nanobot/cli/commands.py \
  nanobot/providers/openai_codex_provider.py \
  tests/config/test_config_migration.py \
  tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 5: Commit the implementation**

```bash
git add nanobot/config/schema.py nanobot/cli/commands.py \
  nanobot/providers/openai_codex_provider.py \
  tests/config/test_config_migration.py \
  tests/providers/test_openai_codex_provider.py tests/cli/test_commands.py
git commit -m "feat(codex): add configurable fast service tier"
```

---

### Task 2: Configure and deploy Fast mode

**Files:**
- Modify externally: `~/.nanobot/config.json` (only `agents.defaults.serviceTier`)
- Preserve: existing config backups and `~/.codex/auth.json`

**Interfaces:**
- Consumes: the reviewed Task 1 provider mapping.
- Produces: a single running Gateway whose Codex requests use the priority tier.

- [ ] **Step 1: Run the full repository gate**

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q
```

Expected: zero failures; existing unrelated logger permission diagnostics may appear without changing the exit code.

- [ ] **Step 2: Back up and change only the service-tier setting**

Create a new non-overwriting config backup, then set:

```json
"serviceTier": "fast"
```

Structurally compare before/after configs after removing `serviceTier`; expected: equal. Confirm the four existing target values are unchanged.

- [ ] **Step 3: Restart to one stable Gateway**

Use the established restart/supervisor workflow. Wait for exactly one `nanobot gateway` process, one channel initialization, and one heartbeat initialization in the final log boundary.

- [ ] **Step 4: Verify the deployed request**

Run a sanitized live Codex request through the deployed provider while observing only booleans/counts. Assert:

```text
request_ok=True
service_tier_priority=True
model_ok=True
reasoning_medium=True
context_640k=True
```

Scan the new Gateway log boundary for errors without printing user content, websocket URLs, credentials, headers, or encrypted continuation data.

- [ ] **Step 5: Record deployment evidence**

Write an ignored report under `.superpowers/sdd/` with final HEAD, tests, sanitized config values, PID/counts, live booleans, backup path, and rollback status. Do not commit external config or report files.
