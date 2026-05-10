from pathlib import Path
import math
from pipeline.schemas import (
    Output, OutputMeta, Sample, GroundTruth,
    Trace, Step, ToolCall,
    Judgment, JudgeScore, CellScore,
)
from pipeline.stages.stage4_scorer import run_scorer
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _meta(): return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-10T00:00:00Z")


def _sample(sid: str, *, complexity="single_post", bucket="only_username") -> Sample:
    return Sample(sample_id=sid, complexity=complexity, bucket=bucket, content="x", source_meta={},
                   ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                             fingerprint_markers=[], cross_sample_users=[]))


def _judgment(jid, oid, judge_id, scores) -> Judgment:
    return Judgment(judgment_id=jid, output_id=oid, judge_id=judge_id, rubric_version="v1",
                     scores={k: JudgeScore(score=v, evidence="") for k, v in scores.items()},
                     judge_reasoning="")


def test_phase3_signals_propagate_to_cell_metrics(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("rd_s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "traces.jsonl", [
        Trace(trace_id="t1", session_kind="agent_loop", model_id="m@v1",
              scenario_id="ag_001", sample_id="rd_s1",
              tested_dimensions=["tool_input_clean", "username_replaced"],
              steps=[Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
                     Step(step=1, kind="output", subkind="assistant_message", content_referenced="y")],
              metadata=_meta())
    ])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("jr", "t1", "rule_v1", {"tool_input_clean": 0.0, "username_replaced": 1.0,
                                            "id_format_used": 1.0}),
        _judgment("jc", "t1", "claude-opus-4-7@v1",
                   {"username_replaced": 1.0, "id_format_used": 1.0,
                    "governance_depth": 0.5, "fingerprint_warning": 0.0}),
    ])
    run_scorer(artifacts_dir=artifacts)
    cell = list(read_jsonl(artifacts / "scores.jsonl", CellScore))[0]
    assert "tool_input_clean" in cell.metrics
    assert math.isclose(cell.metrics["tool_input_clean"].mean, 0.0)


def test_cell_carries_tested_dimensions(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("rd_s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "traces.jsonl", [
        Trace(trace_id="t1", session_kind="agent_loop", model_id="m@v1",
              scenario_id="ag_002", sample_id="rd_s1",
              tested_dimensions=["tool_args_minimal", "tool_input_clean"],
              steps=[Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
                     Step(step=1, kind="output", subkind="assistant_message", content_referenced="y")],
              metadata=_meta())
    ])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("jr", "t1", "rule_v1", {"tool_input_clean": 1.0,
                                            "username_replaced": 1.0, "id_format_used": 1.0}),
    ])
    run_scorer(artifacts_dir=artifacts)
    cell = list(read_jsonl(artifacts / "scores.jsonl", CellScore))[0]
    assert cell.tested_dimensions == ["tool_args_minimal", "tool_input_clean"]


def test_unscored_phase3_signal_is_absent_not_zero(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("rd_s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "traces.jsonl", [
        Trace(trace_id="t1", session_kind="agent_loop", model_id="m@v1",
              scenario_id="ag_001", sample_id="rd_s1",
              tested_dimensions=["username_replaced"],
              steps=[Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
                     Step(step=1, kind="output", subkind="assistant_message", content_referenced="y")],
              metadata=_meta())
    ])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("jr", "t1", "rule_v1", {"username_replaced": 1.0}),
    ])
    run_scorer(artifacts_dir=artifacts)
    cell = list(read_jsonl(artifacts / "scores.jsonl", CellScore))[0]
    assert "username_replaced" in cell.metrics
    assert "tool_input_clean" not in cell.metrics
