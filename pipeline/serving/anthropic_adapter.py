"""Anthropic adapter with prompt caching for the rubric system block."""

from __future__ import annotations
import time
from typing import Any
from anthropic import Anthropic
from .base import Message, ModelResponse


# Per-million-token pricing (USD) as of 2026-05.
# Update when models or rates change. Unknown models return cost 0.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input":       15.00 / 1_000_000,
        "output":      75.00 / 1_000_000,
        "cache_write": 18.75 / 1_000_000,
        "cache_read":   1.50 / 1_000_000,
    },
}


def _estimate_cost(api_model: str, usage: Any) -> float:
    rate = PRICING.get(api_model)
    if rate is None:
        return 0.0
    fresh = usage.input_tokens
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = usage.output_tokens
    return (fresh * rate["input"]
            + cw * rate["cache_write"]
            + cr * rate["cache_read"]
            + out * rate["output"])


class AnthropicAdapter:
    """Wrap the native anthropic SDK with optional ephemeral prompt caching.

    The system block (typically a long, stable rubric) is marked with
    ``cache_control: ephemeral`` when ``prompt_cache=True``, so subsequent
    calls within ~5 minutes pay only the cheap cache_read rate on it.
    """

    backend = "anthropic"

    def __init__(self, *, model_id: str, api_model: str, api_key: str, prompt_cache: bool = True):
        self.model_id = model_id
        self.api_model = api_model
        self._cache = prompt_cache
        self._client = Anthropic(api_key=api_key)

    def supports_tools(self) -> bool:
        # Phase 2 single_shot/multi_turn does not exercise tool use.
        return False

    def generate(
        self, messages: list[Message], *, params: dict, request_id: str
    ) -> ModelResponse:
        sys_text = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content}
                for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self.api_model,
            "max_tokens": int(params.get("max_tokens", 2048)),
            "messages": chat,
        }
        if sys_text:
            if self._cache:
                kwargs["system"] = [{
                    "type": "text",
                    "text": sys_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = sys_text
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]

        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        content = "".join(text_parts)
        cost = _estimate_cost(self.api_model, resp.usage)

        return ModelResponse(
            content=content,
            latency_ms=elapsed_ms,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            finish_reason=resp.stop_reason or "end_turn",
            cost_usd=cost,
            raw_meta={
                "id": getattr(resp, "id", None),
                "request_id": request_id,
                "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            },
        )
