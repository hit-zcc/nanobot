# Codex Fast Mode Design

## Goal

Allow nanobot's active Codex provider to use the official Codex Fast service tier while preserving the current model, reasoning effort, context window, OAuth flow, and all non-Codex providers.

## Configuration

Add `serviceTier` to `agents.defaults` as an optional string. The deployed configuration will use:

```json
{
  "agents": {
    "defaults": {
      "model": "openai-codex/gpt-5.6-sol",
      "provider": "openai_codex",
      "reasoningEffort": "medium",
      "serviceTier": "fast",
      "contextWindowTokens": 655360
    }
  }
}
```

Omitting `serviceTier` preserves current behavior. This makes Fast mode explicit and reversible rather than hard-coded.

## Request mapping

The runtime passes the optional setting only to `OpenAICodexProvider`. That provider maps the user-facing value `fast` to the Responses request field:

```json
{"service_tier": "priority"}
```

This matches the installed official Codex CLI metadata, where the `priority` tier is displayed as `Fast`. The setting is independent of reasoning effort, so `reasoningEffort: medium` remains unchanged.

Unknown non-empty service-tier values fail locally with a fixed actionable error and are never sent to Codex. The supported initial values are `fast` and an omitted/null value; no general multi-provider service-tier abstraction is introduced.

## Components

- `AgentDefaults`: declares the optional `serviceTier` config field.
- Runtime provider construction: passes the value into `OpenAICodexProvider` only.
- `OpenAICodexProvider`: validates and maps `fast` to `priority` in the request body.
- Status output: continues to show model and context; no status-format change is required for this small deployment setting.

## Verification

- Unit test that `serviceTier: fast` reaches the provider as `service_tier: priority`.
- Unit test that omission sends no service-tier field.
- Unit test that unsupported values fail safely before transport.
- Config load/save regression test for the camelCase field.
- Focused provider/config/CLI tests, Ruff, and full suite.
- After updating `~/.nanobot/config.json`, restart to one stable Gateway process and run a sanitized live request proving the priority tier is accepted.

## Operational notes

Fast mode can consume Codex credits at a higher rate. Existing config backups remain untouched. The deployment changes only `serviceTier`; model, provider, reasoning effort, context window, credentials, sessions, and memory remain unchanged.
