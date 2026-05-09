from unittest.mock import MagicMock, patch
from pipeline.serving.base import Message
from pipeline.serving.anthropic_adapter import (
    AnthropicAdapter, _estimate_cost, PRICING,
)


@patch("pipeline.serving.anthropic_adapter.Anthropic")
def test_generate_returns_model_response(mock_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_text_block = MagicMock(); fake_text_block.type = "text"; fake_text_block.text = "Hi there"
    fake_resp.content = [fake_text_block]
    fake_resp.stop_reason = "end_turn"
    fake_resp.id = "msg_abc"
    fake_resp.usage.input_tokens = 12
    fake_resp.usage.output_tokens = 5
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    fake_client.messages.create.return_value = fake_resp
    mock_cls.return_value = fake_client

    a = AnthropicAdapter(model_id="m@v1", api_model="claude-opus-4-7", api_key="k", prompt_cache=False)
    r = a.generate(
        [Message(role="user", content="hi")],
        params={"temperature": 0.0, "max_tokens": 500},
        request_id="req-1",
    )
    assert r.content == "Hi there"
    assert r.tokens_in == 12
    assert r.tokens_out == 5
    assert r.finish_reason == "end_turn"
    assert r.cost_usd > 0


@patch("pipeline.serving.anthropic_adapter.Anthropic")
def test_system_message_has_cache_control_when_enabled(mock_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(type="text", text="ok")]
    fake_resp.stop_reason = "end_turn"; fake_resp.id = "x"
    fake_resp.usage.input_tokens = 1; fake_resp.usage.output_tokens = 1
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    fake_client.messages.create.return_value = fake_resp
    mock_cls.return_value = fake_client

    a = AnthropicAdapter(model_id="m@v1", api_model="claude-opus-4-7", api_key="k", prompt_cache=True)
    a.generate(
        [Message(role="system", content="rubric"), Message(role="user", content="judge this")],
        params={"max_tokens": 100}, request_id="r",
    )
    called = fake_client.messages.create.call_args
    sys = called.kwargs["system"]
    assert isinstance(sys, list)
    assert sys[0]["text"] == "rubric"
    assert sys[0]["cache_control"] == {"type": "ephemeral"}


@patch("pipeline.serving.anthropic_adapter.Anthropic")
def test_no_cache_when_disabled(mock_cls):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(type="text", text="ok")]
    fake_resp.stop_reason = "end_turn"; fake_resp.id = "x"
    fake_resp.usage.input_tokens = 1; fake_resp.usage.output_tokens = 1
    fake_resp.usage.cache_creation_input_tokens = 0
    fake_resp.usage.cache_read_input_tokens = 0
    fake_client.messages.create.return_value = fake_resp
    mock_cls.return_value = fake_client

    a = AnthropicAdapter(model_id="m@v1", api_model="claude-opus-4-7", api_key="k", prompt_cache=False)
    a.generate(
        [Message(role="system", content="rubric"), Message(role="user", content="x")],
        params={"max_tokens": 100}, request_id="r",
    )
    sys = fake_client.messages.create.call_args.kwargs["system"]
    assert sys == "rubric"  # plain string when cache off


def test_estimate_cost_uses_pricing_table():
    # claude-opus-4-7: input 15/M, output 75/M, cache_read 1.5/M, cache_write 18.75/M
    usage = MagicMock()
    usage.input_tokens = 1_000_000
    usage.output_tokens = 1_000_000
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    cost = _estimate_cost("claude-opus-4-7", usage)
    assert abs(cost - (15.0 + 75.0)) < 1e-6


def test_estimate_cost_with_cache():
    usage = MagicMock()
    usage.input_tokens = 1000  # of which 800 cache_read, 200 cache_creation, 0 fresh
    usage.output_tokens = 100
    usage.cache_creation_input_tokens = 200
    usage.cache_read_input_tokens = 800
    cost = _estimate_cost("claude-opus-4-7", usage)
    # SDK reports input_tokens as fresh-only (not cached). 200 cache write @18.75, 800 cache read @1.5, 1000 fresh @15.
    expected = (1000 / 1_000_000) * 15.0 + (200 / 1_000_000) * 18.75 + (800 / 1_000_000) * 1.5 + (100 / 1_000_000) * 75.0
    assert abs(cost - expected) < 1e-6


def test_unknown_model_returns_zero_cost():
    usage = MagicMock(); usage.input_tokens = 1; usage.output_tokens = 1
    usage.cache_creation_input_tokens = 0; usage.cache_read_input_tokens = 0
    assert _estimate_cost("never-heard-of-it", usage) == 0.0
