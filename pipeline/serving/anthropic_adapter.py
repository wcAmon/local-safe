"""Anthropic adapter with prompt caching for the rubric system block."""

from __future__ import annotations
import json
import time
from typing import Any
from anthropic import Anthropic
from .base import Message, ModelResponse
from pipeline.schemas import ToolCall, ToolSpec


# Per-million-token pricing (USD) as of 2026-05.
# Update when models or rates change. Unknown models return cost 0.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input":       15.00 / 1_000_000,
        "output":      75.00 / 1_000_000,
        "cache_write": 18.75 / 1_000_000,
        "cache_read":   1.50 / 1_000_000,
    },
    "claude-haiku-4-5-20251001": {
        "input":        1.00 / 1_000_000,
        "output":       5.00 / 1_000_000,
        "cache_write":  1.25 / 1_000_000,
        "cache_read":   0.10 / 1_000_000,
    },
    "claude-sonnet-4-6": {
        "input":        3.00 / 1_000_000,
        "output":      15.00 / 1_000_000,
        "cache_write":  3.75 / 1_000_000,
        "cache_read":   0.30 / 1_000_000,
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

    def __init__(
        self, *, model_id: str, api_model: str, api_key: str,
        prompt_cache: bool = True, timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.model_id = model_id
        self.api_model = api_model
        self._cache = prompt_cache
        kwargs = {"api_key": api_key}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self._client = Anthropic(**kwargs)

    def supports_tools(self) -> bool:
        return True

    def generate(
        self, messages: list[Message], *, params: dict, request_id: str,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        sys_text = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = _build_anthropic_chat(messages)
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
        if tools:
            kwargs["tools"] = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,   # Anthropic uses input_schema (vs OpenAI's parameters)
            } for t in tools]

        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        text_parts = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    tool_name=block.name,
                    arguments=dict(block.input),
                ))
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
                "anthropic_content": [
                    {"type": "text", "text": b.text} if getattr(b, "type", None) == "text"
                    else {"type": "tool_use", "id": getattr(b, "id", None),
                          "name": b.name, "input": dict(b.input)}
                    for b in resp.content
                ],
            },
            tool_calls=tool_calls,
        )


def _build_anthropic_chat(messages: list[Message]) -> list[dict]:
    """Convert agent_loop Messages into Anthropic chat blocks.

    The runner uses OpenAI-flavoured tool roundtrip (``Message(role="assistant",
    tool_calls=[...])`` followed by ``Message(role="tool", content=result)``).
    Anthropic requires ``tool_use`` content blocks inline in the assistant
    message and ``tool_result`` blocks in a single subsequent user message,
    paired by ``tool_use_id``. This function does that transposition.
    """
    chat: list[dict] = []
    msgs = [m for m in messages if m.role != "system"]
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == "assistant" and m.tool_calls:
            blocks, ids = _assistant_to_anthropic_blocks(m)
            chat.append({"role": "assistant", "content": blocks})
            i += 1
            results: list[dict] = []
            while i < len(msgs) and msgs[i].role == "tool":
                k = len(results)
                tid = ids[k] if k < len(ids) else f"toolu_synth_{k}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": msgs[i].content,
                })
                i += 1
            if results:
                chat.append({"role": "user", "content": results})
        elif m.role == "tool":
            # Orphan tool result — shouldn't happen in well-formed runs.
            i += 1
        else:
            chat.append({"role": m.role, "content": m.content})
            i += 1
    return chat


def _assistant_to_anthropic_blocks(m: Message) -> tuple[list[dict], list[str]]:
    """Returns (content_blocks, tool_use_ids).

    ``m.tool_calls`` can be:
    - Anthropic shape (from this adapter's prior raw_meta["anthropic_content"]):
      a list of mixed text/tool_use content blocks.
    - OpenAI-compat shape (from openai_compat's raw_meta["openai_tool_calls"]):
      a list of ``{id, type:"function", function:{name, arguments(JSON string)}}``.
    """
    tcs = list(m.tool_calls or [])
    is_anthropic_shape = bool(tcs) and tcs[0].get("type") in ("text", "tool_use")
    if is_anthropic_shape:
        ids = [b["id"] for b in tcs if b.get("type") == "tool_use" and b.get("id")]
        return tcs, ids
    blocks: list[dict] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    ids: list[str] = []
    for tc in tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tid = tc.get("id") or f"toolu_{len(ids)}"
        blocks.append({
            "type": "tool_use",
            "id": tid,
            "name": fn.get("name"),
            "input": args,
        })
        ids.append(tid)
    return blocks, ids
