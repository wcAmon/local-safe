from unittest.mock import MagicMock, patch
from pipeline.serving.base import Message
from pipeline.serving.openai_compat import OpenAICompatAdapter


@patch("pipeline.serving.openai_compat.OpenAI")
def test_generate_returns_model_response(mock_openai_cls):
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Hello world"
    fake_response.choices[0].finish_reason = "stop"
    fake_response.usage.prompt_tokens = 5
    fake_response.usage.completion_tokens = 3
    fake_client.chat.completions.create.return_value = fake_response
    mock_openai_cls.return_value = fake_client

    adapter = OpenAICompatAdapter(
        model_id="m@v1", api_model="qwen3.6-27b-q6", base_url="http://x/v1",
    )
    resp = adapter.generate(
        [Message(role="user", content="hi")],
        params={"temperature": 0.0, "max_tokens": 100},
        request_id="req-1",
    )
    assert resp.content == "Hello world"
    assert resp.tokens_in == 5
    assert resp.tokens_out == 3
    assert resp.finish_reason == "stop"
    assert resp.cost_usd == 0.0  # local
    assert resp.latency_ms >= 0


@patch("pipeline.serving.openai_compat.OpenAI")
def test_generate_passes_correct_params(mock_openai_cls):
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "x"
    fake_response.choices[0].finish_reason = "stop"
    fake_response.usage.prompt_tokens = 1
    fake_response.usage.completion_tokens = 1
    fake_client.chat.completions.create.return_value = fake_response
    mock_openai_cls.return_value = fake_client

    adapter = OpenAICompatAdapter(model_id="m@v1", api_model="m", base_url="http://x/v1")
    adapter.generate(
        [Message(role="user", content="hello")],
        params={"temperature": 0.0, "max_tokens": 100, "seed": 42},
        request_id="req-2",
    )
    called = fake_client.chat.completions.create.call_args
    assert called.kwargs["model"] == "m"
    assert called.kwargs["temperature"] == 0.0
    assert called.kwargs["max_tokens"] == 100
    assert called.kwargs["seed"] == 42
    assert called.kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_supports_tools_returns_true():
    """Phase 3: openai_compat backends advertise tool support."""
    adapter = OpenAICompatAdapter(model_id="m@v1", api_model="m", base_url="http://x/v1")
    assert adapter.supports_tools() is True
