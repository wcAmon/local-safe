from pathlib import Path
from unittest.mock import MagicMock
import json
from pipeline.schemas import Output, OutputMeta, Sample, GroundTruth, Judgment
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelResponse
from pipeline.stages.stage3b_llm_judge import run_llm_judge, parse_judge_json
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _output(oid: str, sample_id: str, response: str = "user_001 was nice") -> Output:
    return Output(
        output_id=oid, model_id="m@v1", prompt_id="p0", sample_id=sample_id,
        rendered_prompt="...", response=response,
        leaked_refs=[],
        metadata=OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z"),
    )


def _sample(sid: str, content: str = "<<U-abc>> said hi") -> Sample:
    return Sample(
        sample_id=sid, complexity="single_post", bucket="only_username",
        content=content, source_meta={},
        ground_truth=GroundTruth(usernames=[], user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
    )


def test_parse_judge_json_well_formed():
    txt = json.dumps({
        "username_replaced":   {"score": 1.0, "evidence": "ok"},
        "id_format_used":      {"score": 0.5, "evidence": "partial"},
        "governance_depth":    {"score": 0.0, "evidence": "none"},
        "fingerprint_warning": {"score": 0.0, "evidence": "none"},
    })
    out = parse_judge_json(txt)
    assert out["username_replaced"]["score"] == 1.0


def test_parse_judge_json_extracts_from_code_fence():
    txt = "```json\n" + json.dumps({
        "username_replaced":   {"score": 1.0, "evidence": "x"},
        "id_format_used":      {"score": 1.0, "evidence": "y"},
        "governance_depth":    {"score": 1.0, "evidence": "z"},
        "fingerprint_warning": {"score": 1.0, "evidence": "w"},
    }) + "\n```"
    out = parse_judge_json(txt)
    assert out["governance_depth"]["score"] == 1.0


def test_run_llm_judge_writes_judgment(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [_output("o1", "s1", response="user_001 was nice")])
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1")])

    fake_adapter = MagicMock()
    fake_adapter.model_id = "gpt-oss-120b@v1"
    fake_adapter.generate.return_value = ModelResponse(
        content=json.dumps({
            "username_replaced":   {"score": 1.0, "evidence": "no leak"},
            "id_format_used":      {"score": 1.0, "evidence": "user_001"},
            "governance_depth":    {"score": 0.5, "evidence": "ok"},
            "fingerprint_warning": {"score": 0.0, "evidence": "no warn"},
        }),
        latency_ms=10, tokens_in=10, tokens_out=10, finish_reason="stop", cost_usd=0, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="gpt-oss-120b@v1", backend="openai_compat",
                             api_model="gpt-oss-120b", base_url_env="X",
                             params={"temperature": 0.0, "seed": 42, "max_tokens": 2048})

    n = run_llm_judge(
        adapter=fake_adapter, judge_cfg=judge_cfg,
        rubric_path=Path(__file__).resolve().parents[1] / "config" / "rubric.v1.yaml",
        vault_dir=vault, artifacts_dir=artifacts,
    )
    assert n == 1
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.judge_id == "gpt-oss-120b@v1"
    assert j.scores["governance_depth"].score == 0.5


def test_run_llm_judge_idempotent(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "s1")])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [_output("o1", "s1", response="user_001")])
    write_jsonl(artifacts / "samples_referenced.jsonl", [_sample("s1")])

    fake_adapter = MagicMock()
    fake_adapter.model_id = "gpt-oss-120b@v1"
    fake_adapter.generate.return_value = ModelResponse(
        content=json.dumps({
            "username_replaced":   {"score": 1.0, "evidence": ""},
            "id_format_used":      {"score": 1.0, "evidence": ""},
            "governance_depth":    {"score": 0.0, "evidence": ""},
            "fingerprint_warning": {"score": 0.0, "evidence": ""},
        }),
        latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", cost_usd=0, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="gpt-oss-120b@v1", backend="openai_compat",
                             api_model="gpt-oss-120b", base_url_env="X", params={"seed": 42})

    n1 = run_llm_judge(adapter=fake_adapter, judge_cfg=judge_cfg,
                      rubric_path=Path(__file__).resolve().parents[1] / "config" / "rubric.v1.yaml",
                      vault_dir=vault, artifacts_dir=artifacts)
    n2 = run_llm_judge(adapter=fake_adapter, judge_cfg=judge_cfg,
                      rubric_path=Path(__file__).resolve().parents[1] / "config" / "rubric.v1.yaml",
                      vault_dir=vault, artifacts_dir=artifacts)
    assert n1 == 1 and n2 == 0
    assert fake_adapter.generate.call_count == 1
