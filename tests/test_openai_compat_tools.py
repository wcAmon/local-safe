from unittest.mock import MagicMock, patch
from pipeline.serving.base import Message
from pipeline.serving.openai_compat import OpenAICompatAdapter
from pipeline.schemas import ToolSpec, ToolCall


def _spec_fetch_user():
    return ToolSpec(
        name="fetch_user_history",
        description="Get history",
        parameters={"type": "object", "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"]},
    )


@patch("pipeline.serving.openai_compat.OpenAI")
def test_tools_kwarg_renders_openai_format(mock_openai_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices[0].message.content = "ok"
    fake_resp.choices[0].message.tool_calls = None
    fake_resp.choices[0].finish_reason = "stop"
    fake_resp.usage.prompt_tokens = 1
    fake_resp.usage.completion_tokens = 1
    fake_client.chat.completions.create.return_value = fake_resp
    mock_openai_cls.return_value = fake_client

    a = OpenAICompatAdapter(model_id="m@v1", api_model="m", base_url="http://x/v1")
    a.generate(
        [Message(role="user", content="hi")],
        params={"max_tokens": 100}, request_id="r1",
        tools=[_spec_fetch_user()],
    )
    called = fake_client.chat.completions.create.call_args
    assert "tools" in called.kwargs
    rendered = called.kwargs["tools"]
    assert len(rendered) == 1
    assert rendered[0]["type"] == "function"
    assert rendered[0]["function"]["name"] == "fetch_user_history"
    assert rendered[0]["function"]["description"] == "Get history"
    assert rendered[0]["function"]["parameters"]["properties"]["user_id"]["type"] == "string"


@patch("pipeline.serving.openai_compat.OpenAI")
def test_response_tool_calls_parsed(mock_openai_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices[0].message.content = ""
    fake_tc = MagicMock()
    fake_tc.id = "call_1"
    fake_tc.type = "function"
    fake_tc.function.name = "fetch_user_history"
    fake_tc.function.arguments = '{"user_id": "user_001"}'
    fake_resp.choices[0].message.tool_calls = [fake_tc]
    fake_resp.choices[0].finish_reason = "tool_calls"
    fake_resp.usage.prompt_tokens = 5
    fake_resp.usage.completion_tokens = 8
    fake_client.chat.completions.create.return_value = fake_resp
    mock_openai_cls.return_value = fake_client

    a = OpenAICompatAdapter(model_id="m@v1", api_model="m", base_url="http://x/v1")
    r = a.generate(
        [Message(role="user", content="hi")],
        params={"max_tokens": 100}, request_id="r2",
        tools=[_spec_fetch_user()],
    )
    assert len(r.tool_calls) == 1
    assert isinstance(r.tool_calls[0], ToolCall)
    assert r.tool_calls[0].tool_name == "fetch_user_history"
    assert r.tool_calls[0].arguments == {"user_id": "user_001"}
    assert r.finish_reason == "tool_calls"
