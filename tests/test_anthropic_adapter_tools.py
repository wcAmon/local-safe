from unittest.mock import MagicMock, patch
from pipeline.serving.base import Message
from pipeline.serving.anthropic_adapter import AnthropicAdapter
from pipeline.schemas import ToolSpec, ToolCall


def _spec_fetch_user():
    return ToolSpec(
        name="fetch_user_history",
        description="Get history",
        parameters={"type": "object", "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"]},
    )


@patch("pipeline.serving.anthropic_adapter.Anthropic")
def test_tools_kwarg_renders_anthropic_format(mock_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_text = MagicMock(); fake_text.type = "text"; fake_text.text = "ok"
    fake_resp.content = [fake_text]
    fake_resp.stop_reason = "end_turn"; fake_resp.id = "msg"
    fake_resp.usage.input_tokens = 1; fake_resp.usage.output_tokens = 1
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    fake_client.messages.create.return_value = fake_resp
    mock_cls.return_value = fake_client

    a = AnthropicAdapter(model_id="m@v1", api_model="claude-opus-4-7",
                          api_key="k", prompt_cache=False)
    a.generate(
        [Message(role="user", content="hi")],
        params={"max_tokens": 100}, request_id="r",
        tools=[_spec_fetch_user()],
    )
    called = fake_client.messages.create.call_args
    assert "tools" in called.kwargs
    t = called.kwargs["tools"]
    assert len(t) == 1
    assert t[0]["name"] == "fetch_user_history"
    assert t[0]["description"] == "Get history"
    # Anthropic uses input_schema (not parameters)
    assert t[0]["input_schema"]["properties"]["user_id"]["type"] == "string"


@patch("pipeline.serving.anthropic_adapter.Anthropic")
def test_response_tool_use_block_parsed(mock_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_use = MagicMock(); fake_use.type = "tool_use"
    fake_use.name = "fetch_user_history"; fake_use.input = {"user_id": "user_001"}
    fake_use.id = "toolu_1"
    fake_resp.content = [fake_use]
    fake_resp.stop_reason = "tool_use"; fake_resp.id = "msg"
    fake_resp.usage.input_tokens = 5; fake_resp.usage.output_tokens = 7
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    fake_client.messages.create.return_value = fake_resp
    mock_cls.return_value = fake_client

    a = AnthropicAdapter(model_id="m@v1", api_model="claude-opus-4-7",
                          api_key="k", prompt_cache=False)
    r = a.generate(
        [Message(role="user", content="hi")],
        params={"max_tokens": 100}, request_id="r",
        tools=[_spec_fetch_user()],
    )
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].tool_name == "fetch_user_history"
    assert r.tool_calls[0].arguments == {"user_id": "user_001"}
    assert r.finish_reason == "tool_use"
