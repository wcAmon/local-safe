from pathlib import Path
import json
from unittest.mock import MagicMock
from pipeline.schemas import (
    Output, OutputMeta, Sample, GroundTruth,
    Trace, Step, ExposureLedgerEntry, Judgment,
)
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelResponse
from pipeline.stages.stage3b_llm_judge import run_llm_judge
from pipeline.jsonl_io import write_jsonl, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC = REPO_ROOT / "config" / "rubric.v1.yaml"


def _meta(): return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z")


def test_llm_judge_reads_trace_and_scores_multi_step_consistency(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="only_username",
                         content="<<U-x>> said hi", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    write_jsonl(artifacts / "traces.jsonl",
                [Trace(trace_id="t1", session_kind="multi_turn",
                        model_id="m@v1", scenario_id="mt_001", sample_id="rd_s1",
                        steps=[
                            Step(step=0, kind="input", subkind="user_message", content_referenced="give me x"),
                            Step(step=1, kind="output", subkind="assistant_message", content_referenced="user_001 said hi"),
                            Step(step=2, kind="input", subkind="user_message", content_referenced="thanks"),
                            Step(step=3, kind="output", subkind="assistant_message", content_referenced="user_001 was friendly"),
                        ],
                        metadata=_meta())])

    fake = MagicMock()
    fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content=json.dumps({
            "username_replaced":      {"score": 1.0, "evidence": "no leak"},
            "id_format_used":         {"score": 1.0, "evidence": "user_001"},
            "governance_depth":       {"score": 0.5, "evidence": "ok"},
            "fingerprint_warning":    {"score": 0.0, "evidence": "no warn"},
            "multi_step_consistency": {"score": 1.0, "evidence": "stable across turns"},
        }),
        latency_ms=10, tokens_in=10, tokens_out=10, finish_reason="end_turn",
        cost_usd=0.001, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"temperature": 0.0, "max_tokens": 2048})
    n = run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC,
                      vault_dir=vault, artifacts_dir=artifacts)
    assert n == 1
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.judge_id == "claude-opus-4-7@v1"
    assert j.scores["multi_step_consistency"].score == 1.0
    assert j.output_id == "t1"


def test_llm_judge_skips_multi_step_consistency_for_outputs(tmp_path: Path):
    """For single_shot Output, multi_step_consistency is N/A — judge prompt
    omits it and parser tolerates missing key as 'missing_in_response'."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl",
                [Output(output_id="o1", model_id="m@v1", prompt_id="p0", sample_id="rd_s1",
                         rendered_prompt="...", response="user_001", leaked_refs=[], metadata=_meta())])
    write_jsonl(artifacts / "outputs_redacted.jsonl",
                [Output(output_id="o1", model_id="m@v1", prompt_id="p0", sample_id="rd_s1",
                         rendered_prompt="...", response="user_001", leaked_refs=[], metadata=_meta())])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="only_username",
                         content="x", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    write_jsonl(artifacts / "traces.jsonl", [])

    fake = MagicMock(); fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content=json.dumps({
            "username_replaced": {"score": 1.0, "evidence": ""},
            "id_format_used":    {"score": 1.0, "evidence": ""},
            "governance_depth":  {"score": 0.0, "evidence": ""},
            "fingerprint_warning": {"score": 0.0, "evidence": ""},
        }),
        latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="end_turn", cost_usd=0, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"max_tokens": 2048})
    run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC,
                   vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    # multi_step_consistency missing → 0.0 with 'missing_in_response'
    assert j.scores["multi_step_consistency"].score == 0.0
    assert "missing_in_response" in j.scores["multi_step_consistency"].evidence
