"""Model adapter base types — Protocol + ModelResponse dataclass."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None       # NEW: required for tool messages
    tool_calls: list[dict] | None = None  # NEW: assistant messages with tool_use


@dataclass
class ModelResponse:
    content: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    finish_reason: str
    cost_usd: float = 0.0
    raw_meta: dict = None
    tool_calls: list = field(default_factory=list)   # NEW: list[ToolCall]


class ModelAdapter(Protocol):
    model_id: str

    def generate(self, messages: list[Message], *, params: dict, request_id: str,
                 tools: list = None) -> ModelResponse: ...

    def supports_tools(self) -> bool: ...
