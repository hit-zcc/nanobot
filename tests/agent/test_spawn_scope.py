"""Tests for the spawn tool's scope and stopping-condition contract."""

from nanobot.agent.tools.spawn import SpawnTool


def test_spawn_description_requires_large_user_requested_scope() -> None:
    tool = SpawnTool(manager=None)  # description does not access the manager
    description = tool.description

    assert "genuinely large task" in description
    assert "at most 20 tool calls" in description
    assert "minimum evidence" in description
    assert "stopping condition" in description
    assert "Do not silently expand" in description
    assert "code-location question into log sampling" in description
    assert "require a user request" in description
    assert "evidence against spawning" in description


def test_spawn_description_does_not_prefer_background_for_a_handful_of_calls() -> None:
    description = SpawnTool(manager=None).description

    assert "PREFER THIS over doing the work yourself" not in description
    assert "more than a handful of tool calls" not in description
