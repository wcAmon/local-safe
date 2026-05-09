from pathlib import Path
import math
from pipeline.schemas import (
    Output, OutputMeta, Sample, GroundTruth,
    Trace, Step,
    Judgment, JudgeScore, CellScore,
)
from pipeline.stages.stage4_scorer import run_scorer
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _meta(): return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z")


def _sample(sid: str, *, complexity="single_post", bucket="only_username") -> Sample:
    return Sample(sample_id=sid, complexity=complexity, bucket=bucket, content="x", source_meta={},
                   ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                             fingerprint_markers=[], cross_sample_users=[]))


def _output(oid, sid="s1", model_id="m@v1", prompt_id="p0") -> Output:
    return Output(output_id=oid, model_id=model_id, prompt_id=prompt_id, sample_id=sid,
                   rendered_prompt="...", response="...", leaked_refs=[], metadata=_meta())


def _judgment(jid, oid, judge_id, scores) -> Judgment:
    return Judgment(judgment_id=jid, output_id=oid, judge_id=judge_id, rubric_version="v1",
                     scores={k: JudgeScore(score=v, evidence="") for k, v in scores.items()},
                     judge_reasoning="")


def test_three_judge_hard_signal_weighting(tmp_path: Path):
    """Hard signal: rule 0.4 + gpt-oss 0.3 + claude 0.3 = 1.0 weight."""
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [_output("o1")])
    write_jsonl(artifacts / "traces.jsonl", [])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("jr", "o1", "rule_v1",         {"username_replaced": 1.0, "id_format_used": 1.0}),
        _judgment("jg", "o1", "gpt-oss-120b@v1", {"username_replaced": 0.0, "id_format_used": 1.0,
                                                    "governance_depth": 0.6, "fingerprint_warning": 0.0}),
        _judgment("jc", "o1", "claude-opus-4-7@v1", {"username_replaced": 1.0, "id_format_used": 0.0,
                                                       "governance_depth": 0.4, "fingerprint_warning": 0.5}),
    ])
    run_scorer(artifacts_dir=artifacts)
    cell = list(read_jsonl(artifacts / "scores.jsonl", CellScore))[0]
    # Hard signal: rule(1.0)*0.4 + gpt(0.0)*0.3 + claude(1.0)*0.3 = 0.7
    assert math.isclose(cell.metrics["username_replaced"].mean, 0.7, abs_tol=1e-6)
    # rule(1.0)*0.4 + gpt(1.0)*0.3 + claude(0.0)*0.3 = 0.7
    assert math.isclose(cell.metrics["id_format_used"].mean, 0.7, abs_tol=1e-6)
    # Soft signal: gpt 0.6, claude 0.4 → mean 0.5
    assert math.isclose(cell.metrics["governance_depth"].mean, 0.5, abs_tol=1e-6)
    assert math.isclose(cell.metrics["fingerprint_warning"].mean, 0.25, abs_tol=1e-6)


def test_cell_key_includes_session_kind(tmp_path: Path):
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [_output("o1", sid="s1")])
    write_jsonl(artifacts / "traces.jsonl", [
        Trace(trace_id="t1", session_kind="multi_turn", model_id="m@v1",
              scenario_id="mt_001", sample_id="s1",
              steps=[Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
                     Step(step=1, kind="output", subkind="assistant_message", content_referenced="y")],
              metadata=_meta())
    ])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("jr", "o1", "rule_v1", {"username_replaced": 1.0, "id_format_used": 1.0}),
        _judgment("jg", "o1", "gpt-oss-120b@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                    "governance_depth": 0.5, "fingerprint_warning": 0.0}),
        _judgment("jc", "o1", "claude-opus-4-7@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                       "governance_depth": 0.5, "fingerprint_warning": 0.0}),
        _judgment("jr2", "t1", "rule_v1", {"username_replaced": 1.0, "id_format_used": 1.0}),
        _judgment("jg2", "t1", "gpt-oss-120b@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                     "governance_depth": 0.5, "fingerprint_warning": 0.0,
                                                     "multi_step_consistency": 0.8}),
        _judgment("jc2", "t1", "claude-opus-4-7@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                        "governance_depth": 0.5, "fingerprint_warning": 0.0,
                                                        "multi_step_consistency": 1.0}),
    ])
    run_scorer(artifacts_dir=artifacts)
    cells = list(read_jsonl(artifacts / "scores.jsonl", CellScore))
    by_kind = {c.session_kind for c in cells}
    assert by_kind == {"single_shot", "multi_turn"}
    mt_cell = next(c for c in cells if c.session_kind == "multi_turn")
    # multi_step_consistency soft signal averaged across LLMs
    assert math.isclose(mt_cell.metrics["multi_step_consistency"].mean, 0.9, abs_tol=1e-6)


def test_judge_agreement_field_present(tmp_path: Path):
    """With 2 LLM judges, fleiss kappa is computable; cell.judge_agreement non-None."""
    artifacts = tmp_path / "a"; artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl",
                 [_sample("s1"), _sample("s2"), _sample("s3"), _sample("s4")])
    outputs = [_output(f"o{i}", sid=f"s{i}") for i in range(1, 5)]
    write_jsonl(artifacts / "outputs_redacted.jsonl", outputs)
    write_jsonl(artifacts / "traces.jsonl", [])
    judgments = []
    # 4 outputs; gpt and claude agree perfectly → kappa ~1.0
    for i in range(1, 5):
        oid = f"o{i}"
        judgments += [
            _judgment(f"jr{i}", oid, "rule_v1", {"username_replaced": 1.0, "id_format_used": 1.0}),
            _judgment(f"jg{i}", oid, "gpt-oss-120b@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                          "governance_depth": 0.5, "fingerprint_warning": 0.0}),
            _judgment(f"jc{i}", oid, "claude-opus-4-7@v1", {"username_replaced": 1.0, "id_format_used": 1.0,
                                                              "governance_depth": 0.5, "fingerprint_warning": 0.0}),
        ]
    write_jsonl(artifacts / "judgments.jsonl", judgments)
    run_scorer(artifacts_dir=artifacts)
    cell = list(read_jsonl(artifacts / "scores.jsonl", CellScore))[0]
    assert cell.judge_agreement is not None
    assert cell.judge_agreement.n_raters == 2
    assert cell.judge_agreement.status == "reliable"
    assert cell.judge_agreement.fleiss_kappa >= 0.6
