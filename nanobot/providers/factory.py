"""Build an LLM provider from config, independent of any CLI surface.

The gateway builds its provider once at startup, but ``/model`` rebuilds one
mid-run from a chat channel, where there is no console to print to and no
process to exit. So the construction rules live here and report a bad config by
raising :class:`ProviderConfigError`; the CLI turns that into console output
plus ``typer.Exit``, the slash command turns it into a chat reply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider


class ProviderConfigError(Exception):
    """The config cannot produce a working provider.

    *hints* are plain-text follow-up lines (no rich markup): they are shown on
    a console and sent to chat channels alike.
    """

    def __init__(self, message: str, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints = hints or []


def build_provider(config: Config) -> LLMProvider:
    """Create the LLM provider described by *config*.

    Routing is driven by ``ProviderSpec.backend`` in the registry.
    """
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    # --- validation ---
    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ProviderConfigError(
                "Azure OpenAI requires api_key and api_base.",
                [
                    "Set them in ~/.nanobot/config.json under providers.azure_openai section",
                    "Use the model field to specify the deployment name.",
                ],
            )
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ProviderConfigError(
                "No API key configured.",
                ["Set one in ~/.nanobot/config.json under providers section"],
            )

    # --- instantiation by backend ---
    if backend == "claude_agent_sdk":
        import importlib.util
        from pathlib import Path

        if importlib.util.find_spec("claude_agent_sdk") is None:
            raise ProviderConfigError(
                "Claude Agent SDK is not installed.",
                ["Install it with: pip install 'nanobot-ai[claude-agent-sdk]'"],
            )
        from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider

        provider = ClaudeAgentSDKProvider(
            default_model=model,
            workspace=Path(config.agents.defaults.workspace).expanduser(),
        )
    elif backend == "claude_oauth":
        from nanobot.providers.claude_oauth_provider import ClaudeOAuthProvider, _load_token
        if _load_token() is None:
            raise ProviderConfigError(
                "Claude OAuth (subscription) not logged in.",
                [
                    "Run: nanobot provider login claude-oauth",
                    "Or switch to an API key: set model to 'anthropic/claude-opus-4-5' "
                    "and ANTHROPIC_API_KEY (or providers.anthropic.apiKey).",
                ],
            )
        provider = ClaudeOAuthProvider(default_model=model)
    elif backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider
        provider = OpenAICodexProvider(
            default_model=model,
            service_tier=config.agents.defaults.service_tier,
        )
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider
        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider
