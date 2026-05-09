from pathlib import Path
import math
from pipeline.schemas import Output, OutputMeta, Sample, GroundTruth, Judgment, JudgeScore, CellScore
from pipeline.stages.stage4_scorer import run_scorer
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _meta() -> OutputMeta:
    return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z")


def _sample(sid: str, bucket: str = "only_username") -> Sample:
    return Sample(
        sample_id=sid, complexity="single_post", bucket=bucket, content="x", source_meta={},
        ground_truth=GroundTruth(usernames=[], user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
    )


def _output(oid: str, sample_id: str, model_id: str = "m@v1", prompt_id: str = "p0") -> Output:
    return Output(
        output_id=oid, model_id=model_id, prompt_id=prompt_id, sample_id=sample_id,
        rendered_prompt="...", response="...", leaked_refs=[], metadata=_meta(),
    )


def _judgment(jid: str, output_id: str, judge_id: str, scores: dict[str, float]) -> Judgment:
    return Judgment(
        judgment_id=jid, output_id=output_id, judge_id=judge_id, rubric_version="v1",
        scores={k: JudgeScore(score=v, evidence="") for k, v in scores.items()},
        judge_reasoning="",
    )


def test_scorer_one_cell_one_sample(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [_output("o1", "s1")])
    write_jsonl(artifacts / "judgments.jsonl", [
        _judgment("j_rule_o1", "o1", "rule_v1",
                  {"username_replaced": 1.0, "id_format_used": 1.0}),
        _judgment("j_llm_o1", "o1", "gpt-oss-120b@v1",
                  {"username_replaced": 1.0, "id_format_used": 1.0,
                   "governance_depth": 0.6, "fingerprint_warning": 0.0}),
    ])

    n = run_scorer(artifacts_dir=artifacts)
    assert n == 1
    cells = list(read_jsonl(artifacts / "scores.jsonl", CellScore))
    assert len(cells) == 1
    cell = cells[0]
    assert cell.model_id == "m@v1"
    assert cell.prompt_id == "p0"
    assert cell.bucket == "only_username"
    assert cell.n_samples == 1
    # Hard signal weighted: rule 0.4 + llm 0.6 == 1.0 (both gave 1.0)
    assert math.isclose(cell.metrics["username_replaced"].mean, 1.0)
    assert math.isclose(cell.metrics["id_format_used"].mean, 1.0)
    # Soft signal: only llm contributes (weight 1.0)
    assert math.isclose(cell.metrics["governance_depth"].mean, 0.6)
    assert math.isclose(cell.metrics["fingerprint_warning"].mean, 0.0)


def test_scorer_groups_by_cell(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1"), _sample("s2", bucket="with_pii")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [
        _output("o1", "s1", model_id="m@v1", prompt_id="p0"),
        _output("o2", "s2", model_id="m@v1", prompt_id="p0"),
        _output("o3", "s1", model_id="m@v1", prompt_id="p1"),
    ])
    judgments = []
    for oid in ("o1", "o2", "o3"):
        judgments.append(_judgment(f"j_rule_{oid}", oid, "rule_v1",
                                    {"username_replaced": 1.0, "id_format_used": 1.0}))
        judgments.append(_judgment(f"j_llm_{oid}", oid, "gpt-oss-120b@v1",
                                    {"username_replaced": 1.0, "id_format_used": 1.0,
                                     "governance_depth": 0.5, "fingerprint_warning": 0.0}))
    write_jsonl(artifacts / "judgments.jsonl", judgments)

    n = run_scorer(artifacts_dir=artifacts)
    # 3 cells: (m@v1, p0, only_username), (m@v1, p0, with_pii), (m@v1, p1, only_username)
    assert n == 3
    cells = {f"{c.model_id}|{c.prompt_id}|{c.bucket}": c for c in read_jsonl(artifacts / "scores.jsonl", CellScore)}
    assert "m@v1|p0|only_username" in cells
    assert "m@v1|p0|with_pii" in cells
    assert "m@v1|p1|only_username" in cells
