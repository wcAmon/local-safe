"""Model adapter base types — Protocol + ModelResponse dataclass."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ModelResponse:
    content: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    finish_reason: str
    cost_usd: float = 0.0
    raw_meta: dict = None


class ModelAdapter(Protocol):
    model_id: str

    def generate(self, messages: list[Message], *, params: dict, request_id: str) -> ModelResponse: ...

    def supports_tools(self) -> bool: ...
