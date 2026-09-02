from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.claude_sdk_runner import ClaudeAgentSDKRunner
from nanobot.agent.runner import AgentRunSpec
from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider


@pytest.mark.asyncio
async def test_sdk_runner_exposes_only_nanobot_mcp_tools(monkeypatch, tmp_path):
    captured = {}

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(self):
            self.usage = {"input_tokens": 9, "output_tokens": 4}
            self.result = "finished"
            self.is_error = False
            self.errors = None
            self.subtype = "success"

    class StreamEvent:
        pass

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class SdkMcpTool:
        def __init__(self, name, description, input_schema, handler):
            self.name = name
            self.description = description
            self.input_schema = input_schema
            self.handler = handler

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["options"] = self

    def create_sdk_mcp_server(name, version, tools):
        return SimpleNamespace(name=name, version=version, tools=tools)

    class ClaudeSDKClient:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def query(self, prompt):
            captured["prompt"] = prompt

        async def receive_response(self):
            server = self.options.mcp_servers["nanobot"]
            captured["tool_result"] = await server.tools[0].handler({"value": "x"})
            yield AssistantMessage([TextBlock("intermediate")])
            yield ResultMessage()

    fake_sdk = SimpleNamespace(
        AssistantMessage=AssistantMessage,
        ClaudeAgentOptions=ClaudeAgentOptions,
        ClaudeSDKClient=ClaudeSDKClient,
        ResultMessage=ResultMessage,
        SdkMcpTool=SdkMcpTool,
        StreamEvent=StreamEvent,
        TextBlock=TextBlock,
        create_sdk_mcp_server=create_sdk_mcp_server,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    tools = SimpleNamespace(
        get_definitions=lambda: [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }],
        execute=AsyncMock(return_value="echoed"),
    )
    provider = ClaudeAgentSDKProvider("claude-agent-sdk/claude-opus-5", tmp_path)
    result = await ClaudeAgentSDKRunner(provider).run(AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do it"},
        ],
        tools=tools,
        model="claude-agent-sdk/claude-opus-5",
        max_iterations=5,
    ))

    options = captured["options"]
    assert options.tools == []
    assert options.setting_sources == []
    assert options.skills == []
    assert options.strict_mcp_config is True
    assert options.permission_mode == "dontAsk"
    assert options.allowed_tools == ["mcp__nanobot__echo"]
    tools.execute.assert_awaited_once_with("echo", {"value": "x"})
    assert captured["tool_result"] == {
        "content": [{"type": "text", "text": "echoed"}],
        "is_error": False,
    }
    assert result.final_content == "finished"
    assert result.tools_used == ["echo"]
    assert result.usage == {"prompt_tokens": 9, "completion_tokens": 4}
