from pathlib import Path
import json
from pipeline.schemas import CellScore, CellMetric
from pipeline.stages.stage5_reporter import render_markdown_report
from pipeline.jsonl_io import write_jsonl


def test_report_contains_all_cells_and_metrics(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cells = [
        CellScore(
            cell_id="m1@v1|p0|single_post|only_username|single_shot",
            model_id="m1@v1", prompt_or_scenario_id="p0",
            complexity="single_post", bucket="only_username",
            session_kind="single_shot", n_samples=10,
            metrics={
                "pii_leak_free":       CellMetric(mean=1.0, ci95=(1.0, 1.0)),
                "username_replaced":   CellMetric(mean=0.9, ci95=(0.8, 1.0)),
                "id_format_used":      CellMetric(mean=0.7, ci95=(0.6, 0.8)),
                "governance_depth":    CellMetric(mean=0.5, ci95=(0.4, 0.6)),
                "fingerprint_leak_free": CellMetric(mean=0.8, ci95=(0.7, 0.9)),
                "fingerprint_warning": CellMetric(mean=0.1, ci95=(0.0, 0.2)),
                "task_utility":        CellMetric(mean=0.7, ci95=(0.6, 0.8)),
                "privacy_utility_balance": CellMetric(mean=0.6, ci95=(0.5, 0.7)),
            },
        ),
        CellScore(
            cell_id="m2@v1|p0|single_post|only_username|single_shot",
            model_id="m2@v1", prompt_or_scenario_id="p0",
            complexity="single_post", bucket="only_username",
            session_kind="single_shot", n_samples=10,
            metrics={
                "pii_leak_free":       CellMetric(mean=0.2, ci95=(0.1, 0.3)),
                "username_replaced":   CellMetric(mean=0.5, ci95=(0.4, 0.6)),
                "id_format_used":      CellMetric(mean=0.4, ci95=(0.3, 0.5)),
                "governance_depth":    CellMetric(mean=0.2, ci95=(0.1, 0.3)),
                "fingerprint_leak_free": CellMetric(mean=0.1, ci95=(0.0, 0.2)),
                "fingerprint_warning": CellMetric(mean=0.0, ci95=(0.0, 0.0)),
                "task_utility":        CellMetric(mean=0.8, ci95=(0.7, 0.9)),
                "privacy_utility_balance": CellMetric(mean=0.1, ci95=(0.0, 0.2)),
            },
        ),
    ]
    write_jsonl(artifacts / "scores.jsonl", cells)

    out_path = render_markdown_report(
        artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="test-run",
    )
    assert out_path.exists()
    radar_path = out_path.parent / "radar_data.json"
    assert radar_path.exists()
    radar = json.loads(radar_path.read_text(encoding="utf-8"))
    assert radar["schema"] == "datatrace-radar-v1"
    assert "single_shot" in radar["tracks"]
    text = out_path.read_text(encoding="utf-8")
    assert "m1@v1" in text
    assert "m2@v1" in text
    for sig in (
        "pii_leak_free", "username_replaced", "id_format_used",
        "governance_depth", "fingerprint_leak_free", "fingerprint_warning",
        "task_utility", "privacy_utility_balance",
    ):
        assert sig in text
    # m1 wins on username_replaced (0.9 vs 0.5)
    assert "Top: `m1@v1`" in text or "m1@v1**" in text  # tolerate formatting
