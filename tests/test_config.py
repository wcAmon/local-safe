import os
from pathlib import Path
import pytest
from pipeline.config import (
    load_models, load_prompts, ModelConfig, PromptConfig, resolve_base_url,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_prompts_returns_4_levels():
    prompts = load_prompts(REPO_ROOT / "config" / "prompts.yaml")
    assert len(prompts) == 4
    assert {p.prompt_id for p in prompts} == {"p0_neutral", "p1_shareable", "p2_publish", "p3_explicit"}
    assert all(isinstance(p, PromptConfig) for p in prompts)


def test_load_prompts_strength_increasing():
    prompts = load_prompts(REPO_ROOT / "config" / "prompts.yaml")
    strengths = [p.strength for p in prompts]
    assert strengths == sorted(strengths)


def test_load_models_under_test_count():
    cfg = load_models(REPO_ROOT / "config" / "models.yaml")
    assert len(cfg.under_test) == 2
    assert {m.model_id for m in cfg.under_test} == {"qwen3.6-27b-q6@v1", "gemma4-26b-a4b-it@v1"}


def test_load_models_includes_rule_and_llm_judge():
    cfg = load_models(REPO_ROOT / "config" / "models.yaml")
    judge_ids = {j.model_id for j in cfg.judges}
    assert "rule_v1" in judge_ids
    assert "gpt-oss-120b@v1" in judge_ids


def test_resolve_base_url_from_env(monkeypatch):
    monkeypatch.setenv("MY_TEST_URL", "http://x:1/v1")
    assert resolve_base_url("MY_TEST_URL") == "http://x:1/v1"


def test_resolve_base_url_missing_raises(monkeypatch):
    monkeypatch.delenv("NEVER_SET_VAR_FOR_TEST", raising=False)
    with pytest.raises(RuntimeError, match="env var"):
        resolve_base_url("NEVER_SET_VAR_FOR_TEST")
