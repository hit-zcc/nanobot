# Codex OAuth Reuse Design

## Goal

Make nanobot's gateway use the current machine's existing Codex ChatGPT login and
subscription quota. The default model becomes `openai-codex/gpt-5.6-sol` with
`reasoningEffort` set to `medium`. Claude remains available as a configuration-only
rollback.

The integration must reuse `~/.codex/auth.json`; it must not create a second OpenAI
OAuth session or copy long-lived credentials into `~/.nanobot`.

## Current State

The repository already contains an `OpenAICodexProvider` that translates nanobot's
OpenAI-style internal messages and tools into the Codex Responses protocol. It also
supports SSE text streaming and function-call responses.

The current implementation has three problems:

1. Authentication is delegated to `oauth-cli-kit`, which may own a separate OAuth
   session instead of reusing the official Codex CLI login.
2. The documented and hard-coded default model is the obsolete
   `gpt-5.1-codex` model.
3. Credential refresh and 401 recovery are not coordinated across concurrent
   gateway work such as chat messages, cron jobs, and heartbeat checks.

## Chosen Approach

Add a Codex credential manager that treats the official Codex CLI as the owner of
authentication and refresh-token lifecycle.

The credential manager reads only the current access token and account identifier
from `~/.codex/auth.json`. When a refresh is required, it asks the installed Codex
CLI/app-server to refresh its managed ChatGPT account, then rereads the file. It
does not implement an independent OAuth browser flow and does not persist OpenAI
tokens under `~/.nanobot`.

This keeps nanobot's existing Responses and tool-call adapter while avoiding a
second OAuth authority. Using `codex exec` as the provider was rejected because it
would introduce a nested agent and tool system that conflicts with nanobot's own
tools, memory, sessions, and streaming behavior.

## Components

### Codex credential manager

The credential manager has one responsibility: return a valid ChatGPT-mode Codex
access token and account ID.

It will:

- Resolve Codex state from `$CODEX_HOME/auth.json` when `CODEX_HOME` is explicitly
  configured, otherwise from `~/.codex/auth.json`.
- Parse the JSON defensively and require `auth_mode` to be `chatgpt`.
- Require `tokens.access_token` and `tokens.account_id`.
- Inspect the access-token expiry without logging token contents.
- Serialize refresh operations with an async lock.
- Invoke the official Codex CLI/app-server refresh path only when the token is near
  expiry or after an authentication failure.
- Reread the credential file after refresh because the CLI remains the only writer.
- Never cache or write a refresh token itself.

### OpenAI Codex provider

`OpenAICodexProvider` will depend on the credential manager instead of
`oauth-cli-kit`. Before each request it obtains current request credentials. If the
Codex backend returns 401, it forces one coordinated refresh and retries the request
once with newly loaded credentials.

The existing message conversion, image input, function-tool conversion, encrypted
reasoning continuity, SSE parsing, and nanobot `LLMResponse` mapping remain in the
provider.

The provider default becomes `openai-codex/gpt-5.6-sol`. It will continue stripping
the `openai-codex/` namespace before sending the model slug to the Codex backend.

### CLI and configuration

`nanobot provider login openai-codex` will no longer start a separate Python OAuth
flow. It will:

1. Detect the `codex` executable.
2. Check whether the official CLI is logged in with ChatGPT.
3. Direct an unauthenticated user to `codex login` (or invoke the official login
   command when running interactively).
4. Report the authenticated account without exposing credentials.

The deployed configuration will set:

```json
{
  "agents": {
    "defaults": {
      "model": "openai-codex/gpt-5.6-sol",
      "provider": "openai_codex",
      "reasoningEffort": "medium"
    }
  }
}
```

No `providers.openaiCodex` secret block is required. Existing Claude configuration
and OAuth credentials remain untouched so rollback only requires changing the
three agent-default fields.

## Request and Refresh Flow

1. A channel message enters the existing `AgentLoop`.
2. Provider selection resolves `openai_codex` from the explicit provider and model
   prefix.
3. The Codex provider asks the credential manager for request credentials.
4. The manager reads `auth.json`; if the access token is still valid, it returns it.
5. The provider sends the existing Responses request with the bearer token and
   ChatGPT account header.
6. Text and tool-call SSE events are converted into nanobot's existing response
   types.
7. On 401, one caller obtains the refresh lock, asks the official Codex CLI to
   refresh, rereads `auth.json`, and retries once. Other callers reuse the refreshed
   file after acquiring the lock.
8. A second 401 is returned as an actionable authentication error rather than
   entering a retry loop.

## Error Handling

- Missing Codex CLI: instruct the user to install the official Codex CLI.
- Missing `auth.json` or non-ChatGPT auth mode: instruct the user to run
  `codex login` and choose ChatGPT login.
- Malformed or incomplete credentials: fail without printing token material.
- Refresh failure: preserve the original file, report `codex login` as the recovery
  action, and do not fall back to an unrelated OpenAI account.
- 401 after refresh: return one clear reauthentication error.
- 429: retain a quota/rate-limit-specific message; do not refresh credentials.
- Model unavailable: report the requested model and point to the Codex CLI model
  list/cache instead of silently selecting a weaker model.
- Backend or SSE protocol changes: return a bounded provider error and retain raw
  status/type metadata only when it contains no credentials.

## Security

- Never copy access or refresh tokens into nanobot configuration, sessions, or logs.
- Never log request authorization headers or the raw authentication file.
- Keep `~/.codex/auth.json` as the single credential source and let the official CLI
  remain its only writer.
- Avoid disabling TLS certificate validation. Certificate failures must be surfaced
  instead of retried with `verify=False`.
- Preserve file permissions and do not rewrite the Codex credential file.

## Testing

Unit tests will cover:

- Valid ChatGPT-mode credential parsing.
- `CODEX_HOME` and default-path resolution.
- Missing, malformed, API-key-mode, and incomplete credential errors.
- Expiry detection without token leakage.
- Concurrent refresh coalescing.
- Normal request success without refresh.
- One forced refresh and retry after 401.
- No refresh for 429 or non-authentication errors.
- Failure after a second 401.
- Current model-prefix stripping and `gpt-5.6-sol` defaults.
- CLI login/status behavior with and without the official Codex executable.
- Existing text streaming, images, and function-call round trips.

Integration verification will use a small direct agent request followed by one
tool-calling request. It will not print credentials. Gateway verification will then
confirm a Feishu response, streaming completion, and a reversible configuration
rollback to Claude.

## Rollout

1. Implement and run focused provider, CLI, and configuration tests.
2. Run the broader provider/agent test suites.
3. Back up only the deployed `~/.nanobot/config.json` before changing defaults.
4. Install the repository in editable mode using the existing deployment workflow.
5. Restart the gateway and perform a Codex smoke test.
6. If authentication or protocol compatibility fails, restore the prior agent
   defaults and restart; no credential rollback is needed because Codex credentials
   are never modified by nanobot.

## Compatibility Boundary

ChatGPT subscription access uses the backend employed by Codex rather than the
public API-key Responses endpoint. That backend is not a general public API contract
for third-party applications and can change with Codex CLI releases. The design
reduces authentication drift by delegating token ownership and refresh to the
official CLI, but the Responses adapter may still require maintenance when Codex's
request or SSE schema changes.
