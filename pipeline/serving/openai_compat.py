"""OpenAI-compatible adapter (works with ollama-hub gateway)."""

from __future__ import annotations
import time
from openai import OpenAI
from .base import Message, ModelResponse


class OpenAICompatAdapter:
    """Wrap openai SDK against any OpenAI-compatible endpoint."""

    def __init__(self, *, model_id: str, api_model: str, base_url: str, api_key: str = "unused"):
        self.model_id = model_id
        self.api_model = api_model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def supports_tools(self) -> bool:
        # Phase 1 single_shot does not exercise tool use.
        return False

    def generate(
        self, messages: list[Message], *, params: dict, request_id: str
    ) -> ModelResponse:
        oa_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": self.api_model,
            "messages": oa_messages,
        }
        # Whitelist supported params; anything unrecognized would 400 against the gateway.
        for k in ("temperature", "max_tokens", "seed", "top_p", "stop"):
            if k in params:
                kwargs[k] = params[k]

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            latency_ms=elapsed_ms,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            finish_reason=choice.finish_reason or "stop",
            cost_usd=0.0,
            raw_meta={"id": getattr(resp, "id", None), "request_id": request_id},
        )
