from pathlib import Path
import json
import pytest
from pipeline.serving.budget import BudgetGuard, BudgetExceeded


def test_check_before_call_passes_when_under_cap(tmp_path: Path):
    g = BudgetGuard(total_cap=10.0, per_judge_caps={"j1": 5.0}, cost_log_path=tmp_path / "cost.jsonl")
    assert g.check_before_call("j1") is True


def test_check_before_call_fails_when_total_exceeded(tmp_path: Path):
    g = BudgetGuard(total_cap=1.0, per_judge_caps={}, cost_log_path=tmp_path / "cost.jsonl")
    g.record(judge_id="j1", cost_usd=0.5, output_id="o1")
    g.record(judge_id="j1", cost_usd=0.6, output_id="o2")
    assert g.check_before_call("j1") is False


def test_check_before_call_fails_when_per_judge_cap_exceeded(tmp_path: Path):
    g = BudgetGuard(total_cap=100.0, per_judge_caps={"j1": 1.0}, cost_log_path=tmp_path / "cost.jsonl")
    g.record(judge_id="j1", cost_usd=1.5, output_id="o1")
    assert g.check_before_call("j1") is False
    assert g.check_before_call("j2") is True  # other judges unaffected


def test_record_persists_to_cost_log(tmp_path: Path):
    log = tmp_path / "cost.jsonl"
    g = BudgetGuard(total_cap=10.0, per_judge_caps={}, cost_log_path=log)
    g.record(judge_id="claude-opus-4-7@v1", cost_usd=0.123,
             output_id="o1", tokens_in=10, tokens_out=20,
             cache_creation_input_tokens=5, cache_read_input_tokens=15)
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["judge_id"] == "claude-opus-4-7@v1"
    assert r["cost_usd"] == 0.123
    assert r["tokens_in"] == 10
    assert r["cache_read_input_tokens"] == 15


def test_from_config_loads_yaml(tmp_path: Path):
    cfg = tmp_path / "budget.yaml"
    cfg.write_text("total_usd_cap: 5.0\nper_judge_cap:\n  x: 2.0\non_exceed: stop_and_report\n")
    g = BudgetGuard.from_config(cfg, tmp_path / "cost.jsonl")
    assert g.total_cap == 5.0
    assert g.per_judge_caps == {"x": 2.0}


def test_recovers_state_from_existing_cost_log(tmp_path: Path):
    """Re-instantiating after a partial run should pick up prior spend."""
    log = tmp_path / "cost.jsonl"
    log.write_text(
        json.dumps({"judge_id": "j1", "cost_usd": 4.0, "output_id": "old"}) + "\n"
    )
    g = BudgetGuard(total_cap=5.0, per_judge_caps={"j1": 5.0}, cost_log_path=log)
    g.record(judge_id="j1", cost_usd=2.0, output_id="new")  # exceeds total now
    assert g.check_before_call("j1") is False
