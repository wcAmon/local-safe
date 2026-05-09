from pathlib import Path
from pipeline.schemas import CellScore, CellMetric
from pipeline.stages.stage5_reporter import render_markdown_report
from pipeline.jsonl_io import write_jsonl


def test_agent_loop_cells_show_tested_annotation(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "scores.jsonl", [
        CellScore(
            cell_id="m@v1|ag_001|single_post|only_username|agent_loop",
            model_id="m@v1", prompt_or_scenario_id="ag_001",
            complexity="single_post", bucket="only_username",
            session_kind="agent_loop", n_samples=1,
            metrics={"username_replaced": CellMetric(mean=0.5, ci95=(0.3, 0.7))},
            tested_dimensions=["tool_input_clean", "username_replaced"],
        ),
    ])
    out = render_markdown_report(artifacts_dir=artifacts, reports_dir=tmp_path / "reports", run_id="r1")
    text = out.read_text(encoding="utf-8")
    assert "Agent-loop Leaderboard" in text or "agent_loop" in text.lower()
    # Annotation appears on cell row
    assert "[tested: tool_input_clean, username_replaced]" in text
