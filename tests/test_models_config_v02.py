"""v0.2.0 config: new under_test entries load and point at ollama-hub."""
from __future__ import annotations
from pathlib import Path

from pipeline.config import load_models


CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"


def test_v02_under_test_includes_gemma4_e4b():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "gemma4-e4b-it@v1" in ids


def test_v02_under_test_includes_qwen35_9b():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "qwen3.5-9b@v1" in ids


def test_gemma4_e4b_points_at_ollama_hub():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "gemma4-e4b-it@v1")
    assert m.backend == "openai_compat"
    assert m.base_url_env == "OLLAMA_HUB_BASE_URL"
    assert m.api_model == "gemma4-e4b-it"
    assert m.params.get("temperature") == 0.0
    assert m.params.get("seed") == 42
    assert m.params.get("max_tokens") == 2048


def test_qwen35_9b_points_at_ollama_hub_with_thinking_disabled():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "qwen3.5-9b@v1")
    assert m.backend == "openai_compat"
    assert m.base_url_env == "OLLAMA_HUB_BASE_URL"
    assert m.api_model == "qwen3.5-9b"
    chat_kw = m.params.get("extra_body", {}).get("chat_template_kwargs", {})
    assert chat_kw.get("enable_thinking") is False


def test_v02_keeps_v01_under_test_entries():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "qwen3.6-35b-a3b@v1" in ids
    assert "gemma4-26b-a4b-it@v1" in ids
