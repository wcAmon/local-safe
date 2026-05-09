from pathlib import Path
import json
from pipeline.schemas import CellScore, CellMetric, JudgeAgreement
from pipeline.stages.stage5_reporter import render_markdown_report
from pipeline.jsonl_io import write_jsonl


def _cell(model, sk, kappa=None, status=None) -> CellScore:
    ja = JudgeAgreement(fleiss_kappa=kappa, status=status, n_raters=2) if kappa is not None else None
    return CellScore(
        cell_id=f"{model}|p0|single_post|only_username|{sk}",
        model_id=model, prompt_or_scenario_id="p0",
        complexity="single_post", bucket="only_username",
        session_kind=sk, n_samples=4,
        metrics={"username_replaced": CellMetric(mean=0.8, ci95=(0.7, 0.9))},
        judge_agreement=ja,
    )


def test_report_groups_by_session_kind(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "scores.jsonl", [
        _cell("m1@v1", "single_shot", kappa=0.8, status="reliable"),
        _cell("m2@v1", "multi_turn", kappa=0.5, status="moderate"),
    ])
    out = render_markdown_report(artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="r1")
    text = out.read_text(encoding="utf-8")
    assert "Single-shot Leaderboard" in text or "single_shot" in text.lower()
    assert "Multi-turn Leaderboard" in text or "multi_turn" in text.lower()


def test_unreliable_cells_get_warning_marker(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "scores.jsonl", [
        _cell("m1@v1", "single_shot", kappa=0.2, status="unreliable"),
    ])
    out = render_markdown_report(artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="r2")
    text = out.read_text(encoding="utf-8")
    assert "⚠️" in text


def test_cost_summary_section_when_cost_log_present(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "scores.jsonl", [_cell("m1@v1", "single_shot")])
    cost_log = artifacts / "cost.jsonl"
    cost_log.write_text(
        json.dumps({"judge_id": "claude-opus-4-7@v1", "cost_usd": 0.123, "output_id": "o1"}) + "\n"
        + json.dumps({"judge_id": "claude-opus-4-7@v1", "cost_usd": 0.456, "output_id": "o2"}) + "\n"
    )
    out = render_markdown_report(artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="r3")
    text = out.read_text(encoding="utf-8")
    assert "Cost Summary" in text
    assert "claude-opus-4-7@v1" in text
    assert "0.58" in text or "$0.58" in text   # 0.123 + 0.456 = 0.579 → rounds to 0.58
