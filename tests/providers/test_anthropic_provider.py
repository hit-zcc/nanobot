"""Tests for Anthropic request construction across model generations."""

import re

import pytest

from nanobot.providers.anthropic_provider import AnthropicProvider, _sanitize_tool_id

TOOL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# The composite id shape the OpenAI Codex backend persists into session history.
CODEX_ID = "call_KH8x6ohQyqHgQ8zTQ4sraN5a|fc_0dbc3fe1c13777d5016a6c5c259"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hi"},
]


@pytest.fixture
def provider(monkeypatch) -> AnthropicProvider:
    """Provider with the SDK client construction stubbed out."""
    monkeypatch.setattr(
        "anthropic.AsyncAnthropic", lambda **kw: object(), raising=False,
    )
    return AnthropicProvider(api_key="test", default_model="claude-opus-5")


def _kwargs(provider: AnthropicProvider, model: str, effort: str | None = "medium"):
    return provider._build_kwargs(
        MESSAGES, None, model, 16384, 0.1, effort, None,
    )


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7",
     "claude-fable-5", "claude-mythos-5", "anthropic/claude-opus-5"],
)
def test_adaptive_generation_detected(model):
    assert AnthropicProvider._uses_adaptive_thinking(model)


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-haiku-4-5"],
)
def test_older_generation_not_adaptive(model):
    """The 4-5/4-6 IDs must not be mistaken for the 5-class ones."""
    assert not AnthropicProvider._uses_adaptive_thinking(model)


def test_adaptive_model_uses_effort_and_omits_temperature(provider):
    """Opus 5 rejects both `temperature` and `budget_tokens`."""
    kwargs = _kwargs(provider, "claude-opus-5")

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}
    assert "temperature" not in kwargs
    # max_tokens caps thinking + text together; it must not be inflated.
    assert kwargs["max_tokens"] == 16384


def test_adaptive_model_without_effort_still_omits_temperature(provider):
    """These models think by default, so no effort still means no temperature."""
    kwargs = _kwargs(provider, "claude-opus-5", effort=None)

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "output_config" not in kwargs
    assert "temperature" not in kwargs


def test_adaptive_model_drops_unknown_effort(provider):
    kwargs = _kwargs(provider, "claude-opus-5", effort="turbo")
    assert "output_config" not in kwargs


def test_legacy_model_keeps_budget_tokens(provider):
    kwargs = _kwargs(provider, "claude-opus-4-5")

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "output_config" not in kwargs
    assert kwargs["temperature"] == 1.0


def test_legacy_model_without_effort_keeps_temperature(provider):
    kwargs = _kwargs(provider, "claude-opus-4-5", effort=None)

    assert "thinking" not in kwargs
    assert kwargs["temperature"] == 0.1


def test_conforming_tool_id_is_left_alone():
    assert _sanitize_tool_id("toolu_01ABC-xyz_9") == "toolu_01ABC-xyz_9"
    assert _sanitize_tool_id("") == ""


def test_codex_composite_id_is_made_legal():
    """A `call_x|fc_y` id from Codex history must not 400 the whole request."""
    cleaned = _sanitize_tool_id(CODEX_ID)

    assert TOOL_ID_PATTERN.match(cleaned)
    assert "|" not in cleaned


def test_sanitize_is_deterministic():
    """tool_use and its tool_result are sanitized separately — they must agree."""
    assert _sanitize_tool_id(CODEX_ID) == _sanitize_tool_id(CODEX_ID)


def test_ids_differing_only_in_illegal_chars_stay_distinct():
    """Collapsing illegal characters must not merge two different tool calls."""
    assert _sanitize_tool_id("call_a|fc_b") != _sanitize_tool_id("call_a/fc_b")
    assert _sanitize_tool_id("call_a|fc_b") != _sanitize_tool_id("call_a_fc_b")


def test_history_tool_pairs_survive_conversion(provider):
    """A replayed Codex tool call and its result must still line up."""
    other = "call_ZZ9|fc_deadbeef"
    _, converted = provider._convert_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": CODEX_ID, "function": {"name": "read_file", "arguments": "{}"}},
            {"id": other, "function": {"name": "web_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": CODEX_ID, "content": "file body"},
        {"role": "tool", "tool_call_id": other, "content": "results"},
    ])

    use_ids, result_ids = [], []
    for msg in converted:
        for block in msg["content"] if isinstance(msg["content"], list) else []:
            if block.get("type") == "tool_use":
                use_ids.append(block["id"])
            elif block.get("type") == "tool_result":
                result_ids.append(block["tool_use_id"])

    assert len(use_ids) == 2
    assert all(TOOL_ID_PATTERN.match(i) for i in use_ids + result_ids)
    # Every result still points at a tool_use in the same request.
    assert set(result_ids) == set(use_ids)


def test_missing_tool_id_still_gets_generated(provider):
    _, converted = provider._convert_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]},
    ])
    block = converted[-1]["content"][-1]
    assert block["id"].startswith("toolu_")
    assert TOOL_ID_PATTERN.match(block["id"])


def test_adaptive_model_forces_auto_tool_choice(provider):
    """Extended thinking requires tool_choice auto, even when caller says required."""
    kwargs = provider._build_kwargs(
        MESSAGES,
        [{"function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}],
        "claude-opus-5", 16384, 0.1, None, "required",
    )
    assert kwargs["tool_choice"] == {"type": "auto"}
