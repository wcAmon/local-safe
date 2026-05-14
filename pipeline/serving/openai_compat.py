"""OpenAI / OpenAI-compatible adapter.

Works with both the local ollama-hub gateway and the hosted OpenAI API.
"""

from __future__ import annotations
import json
import time
from openai import OpenAI
from pipeline.schemas import ToolCall, ToolSpec
from .base import Message, ModelResponse


PRICING: dict[str, dict[str, float]] = {
    # USD per token. Keep this conservative and update alongside model config.
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
    # Together AI — pulled from /v1/models pricing fields (2026-05-14).
    "deepseek-ai/DeepSeek-V4-Pro": {"input": 2.10 / 1_000_000, "output": 4.40 / 1_000_000},
    "deepseek-ai/DeepSeek-V3.1": {"input": 0.60 / 1_000_000, "output": 1.70 / 1_000_000},
    "google/gemma-4-31B-it": {"input": 0.20 / 1_000_000, "output": 0.50 / 1_000_000},
    "zai-org/GLM-5.1": {"input": 1.40 / 1_000_000, "output": 4.40 / 1_000_000},
}


def _estimate_cost(api_model: str, usage) -> float:
    rate = PRICING.get(api_model)
    if rate is None or usage is None:
        return 0.0
    return (
        getattr(usage, "prompt_tokens", 0) * rate["input"]
        + getattr(usage, "completion_tokens", 0) * rate["output"]
    )


class OpenAICompatAdapter:
    """Wrap openai SDK against any OpenAI-compatible endpoint."""

    def __init__(
        self, *, model_id: str, api_model: str, base_url: str | None = None,
        api_key: str = "unused", timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.model_id = model_id
        self.api_model = api_model
        kwargs = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self._client = OpenAI(**kwargs)

    def supports_tools(self) -> bool:
        return True   # openai_compat backends advertise tool support

    def generate(
        self, messages: list[Message], *, params: dict, request_id: str,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        oa_messages = []
        for m in messages:
            entry = {"role": m.role, "content": m.content}
            if m.role == "tool" and m.tool_call_id is not None:
                entry["tool_call_id"] = m.tool_call_id
            if m.role == "assistant" and m.tool_calls is not None:
                entry["tool_calls"] = m.tool_calls
            oa_messages.append(entry)
        kwargs = {
            "model": self.api_model,
            "messages": oa_messages,
        }
        for k in ("temperature", "max_tokens", "seed", "top_p", "stop"):
            if k in params:
                kwargs[k] = params[k]
        if "extra_body" in params:
            kwargs["extra_body"] = params["extra_body"]
        if "response_format" in params:
            kwargs["response_format"] = params["response_format"]
        if tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            } for t in tools]

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        usage = resp.usage
        content = choice.message.content
        if not content:
            content = getattr(choice.message, "reasoning_content", "") or ""

        tool_calls: list[ToolCall] = []
        raw_tcs = getattr(choice.message, "tool_calls", None) or []
        for tc in raw_tcs:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            tool_calls.append(ToolCall(tool_name=tc.function.name, arguments=arguments))

        return ModelResponse(
            content=content,
            latency_ms=elapsed_ms,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            finish_reason=choice.finish_reason or "stop",
            cost_usd=_estimate_cost(self.api_model, usage),
            raw_meta={"id": getattr(resp, "id", None), "request_id": request_id,
                      "openai_tool_calls": [
                          {"id": getattr(tc, "id", None),
                           "type": "function",
                           "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                          for tc in raw_tcs
                      ]},
            tool_calls=tool_calls,
        )
