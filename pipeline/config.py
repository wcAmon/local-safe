"""YAML config loaders for models and prompts."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field

Backend = Literal["openai_compat", "openai", "anthropic", "rule"]


class ModelConfig(BaseModel):
    model_id: str
    backend: Backend
    api_model: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    prompt_cache: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ModelsConfig(BaseModel):
    under_test: list[ModelConfig]
    judges: list[ModelConfig]


class PromptConfig(BaseModel):
    prompt_id: str
    strength: int
    template: str


def load_models(path: Path) -> ModelsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(raw)


def load_prompts(path: Path) -> list[PromptConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [PromptConfig.model_validate(p) for p in raw]


def resolve_base_url(env_var: str) -> str:
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(
            f"env var {env_var!r} is not set; ensure .env is loaded "
            f"(run via `uv run` or load_dotenv() at entry)"
        )
    return val


def load_scenarios(path: Path) -> list["Scenario"]:
    from pipeline.schemas import Scenario  # late import to avoid cycle if schemas grows
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Scenario.model_validate(p) for p in raw]


def load_tools(path: Path) -> list["ToolSpec"]:
    from pipeline.schemas import ToolSpec  # late import (cycle-safety)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [ToolSpec.model_validate(t) for t in raw]
