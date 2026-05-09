"""OpenAI-compatible adapter (works with ollama-hub gateway)."""

from __future__ import annotations
import json
import time
from openai import OpenAI
from pipeline.schemas import ToolCall, ToolSpec
from .base import Message, ModelResponse


class OpenAICompatAdapter:
    """Wrap openai SDK against any OpenAI-compatible endpoint."""

    def __init__(self, *, model_id: str, api_model: str, base_url: str, api_key: str = "unused"):
        self.model_id = model_id
        self.api_model = api_model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

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
            cost_usd=0.0,
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
