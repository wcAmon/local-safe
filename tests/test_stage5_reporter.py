from pathlib import Path
from pipeline.schemas import CellScore, CellMetric
from pipeline.stages.stage5_reporter import render_markdown_report
from pipeline.jsonl_io import write_jsonl


def test_report_contains_all_cells_and_metrics(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cells = [
        CellScore(
            cell_id="m1@v1|p0|single_post|only_username",
            model_id="m1@v1", prompt_id="p0", complexity="single_post", bucket="only_username",
            n_samples=10,
            metrics={
                "username_replaced":   CellMetric(mean=0.9, ci95=(0.8, 1.0)),
                "id_format_used":      CellMetric(mean=0.7, ci95=(0.6, 0.8)),
                "governance_depth":    CellMetric(mean=0.5, ci95=(0.4, 0.6)),
                "fingerprint_warning": CellMetric(mean=0.1, ci95=(0.0, 0.2)),
            },
        ),
        CellScore(
            cell_id="m2@v1|p0|single_post|only_username",
            model_id="m2@v1", prompt_id="p0", complexity="single_post", bucket="only_username",
            n_samples=10,
            metrics={
                "username_replaced":   CellMetric(mean=0.5, ci95=(0.4, 0.6)),
                "id_format_used":      CellMetric(mean=0.4, ci95=(0.3, 0.5)),
                "governance_depth":    CellMetric(mean=0.2, ci95=(0.1, 0.3)),
                "fingerprint_warning": CellMetric(mean=0.0, ci95=(0.0, 0.0)),
            },
        ),
    ]
    write_jsonl(artifacts / "scores.jsonl", cells)

    out_path = render_markdown_report(
        artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="test-run",
    )
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "m1@v1" in text
    assert "m2@v1" in text
    for sig in ("username_replaced", "id_format_used", "governance_depth", "fingerprint_warning"):
        assert sig in text
    # m1 wins on username_replaced (0.9 vs 0.5)
    assert "Top: `m1@v1`" in text or "m1@v1**" in text  # tolerate formatting
