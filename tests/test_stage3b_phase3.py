from pathlib import Path
import json
from unittest.mock import MagicMock
from pipeline.schemas import (
    OutputMeta, Sample, GroundTruth,
    Trace, Step, ToolCall, Judgment,
)
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelResponse
from pipeline.stages.stage3b_llm_judge import run_llm_judge
from pipeline.jsonl_io import write_jsonl, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_V2 = REPO_ROOT / "config" / "rubric.v2.yaml"


def _meta(): return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-10T00:00:00Z")


def _agent_trace(tid: str, *, scenario_id: str, tested_dimensions: list[str], steps: list[Step]) -> Trace:
    return Trace(trace_id=tid, session_kind="agent_loop",
                  model_id="m@v1", scenario_id=scenario_id,
                  sample_id="rd_s1", steps=steps,
                  tested_dimensions=tested_dimensions, metadata=_meta())


def test_dimension_routing_only_asks_LLM_for_relevant(tmp_path: Path):
    """ag_002-style: tested_dimensions has tool_args_minimal (LLM) + tool_input_clean (rule).
    The LLM judge should be asked ONLY for tool_args_minimal."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="only_username",
                         content="<<U-x>> in <<LOC-y>>", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    steps = [
        Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
        Step(step=1, kind="output", subkind="assistant_message", content_referenced="user_001 done"),
    ]
    write_jsonl(artifacts / "traces.jsonl",
                [_agent_trace("t1", scenario_id="ag_002",
                                tested_dimensions=["tool_args_minimal", "tool_input_clean"],
                                steps=steps)])

    fake = MagicMock()
    fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content=json.dumps({"tool_args_minimal": {"score": 0.5, "evidence": "filled one optional"}}),
        latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="end_turn",
        cost_usd=0.001, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"max_tokens": 2048})
    n = run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC_V2,
                      vault_dir=vault, artifacts_dir=artifacts)
    assert n == 1
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    # Only tool_args_minimal scored; tool_input_clean is rule-only so judge wasn't asked
    assert j.scores["tool_args_minimal"].score == 0.5

    # Verify the user message asked ONLY for tool_args_minimal (not tool_input_clean)
    user_msg = fake.generate.call_args.args[0][1].content
    assert "tool_args_minimal" in user_msg
    assert "tool_input_clean" not in user_msg


def test_skip_LLM_when_no_LLM_dims(tmp_path: Path):
    """ag_001-style with only rule-scored tested_dimensions: LLM judge should NOT be called."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="only_username",
                         content="x", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    steps = [Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
             Step(step=1, kind="output", subkind="assistant_message", content_referenced="ok")]
    # Only rule-only dimensions — no LLM_SCORED_DIMENSIONS in the list
    write_jsonl(artifacts / "traces.jsonl",
                [_agent_trace("t1", scenario_id="ag_001",
                                tested_dimensions=["tool_input_clean", "id_consistency"],
                                steps=steps)])

    fake = MagicMock()
    fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content="{}", latency_ms=1, tokens_in=1, tokens_out=1,
        finish_reason="end_turn", cost_usd=0.0, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"max_tokens": 2048})
    n = run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC_V2,
                      vault_dir=vault, artifacts_dir=artifacts)
    assert n == 0   # no LLM judgment written
    assert fake.generate.call_count == 0


def test_tool_result_governance_scoring(tmp_path: Path):
    """ag_003-style: tested_dimensions includes tool_result_governance + governance_depth (both LLM)."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="only_username",
                         content="x", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    steps = [Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
             Step(step=1, kind="output", subkind="assistant_message", content_referenced="user_001 from anonymized location")]
    write_jsonl(artifacts / "traces.jsonl",
                [_agent_trace("t1", scenario_id="ag_003",
                                tested_dimensions=["tool_result_governance", "governance_depth"],
                                steps=steps)])

    fake = MagicMock()
    fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content=json.dumps({
            "tool_result_governance": {"score": 1.0, "evidence": "anonymized location"},
            "governance_depth": {"score": 0.5, "evidence": "partial"},
        }),
        latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="end_turn",
        cost_usd=0.001, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"max_tokens": 2048})
    run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC_V2,
                   vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["tool_result_governance"].score == 1.0
    assert j.scores["governance_depth"].score == 0.5


def test_fingerprint_and_utility_dimensions_are_routed_to_llm(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(artifacts / "outputs_redacted.jsonl", [])
    write_jsonl(artifacts / "samples_referenced.jsonl",
                [Sample(sample_id="rd_s1", complexity="single_post", bucket="with_pii",
                         content="<<U-x>> works at <<ORG-y>> in <<LOC-z>>", source_meta={},
                         ground_truth=GroundTruth(usernames=[], user_mentions=[],
                                                   fingerprint_markers=[], cross_sample_users=[]))])
    steps = [
        Step(step=0, kind="input", subkind="user_message", content_referenced="x"),
        Step(step=1, kind="output", subkind="assistant_message",
             content_referenced="A tech worker in a city reported the issue."),
    ]
    write_jsonl(artifacts / "traces.jsonl",
                [_agent_trace("t1", scenario_id="ag_003",
                              tested_dimensions=["fingerprint_leak_free", "task_utility"],
                              steps=steps)])

    fake = MagicMock()
    fake.model_id = "claude-opus-4-7@v1"
    fake.generate.return_value = ModelResponse(
        content=json.dumps({
            "fingerprint_leak_free": {"score": 1.0, "evidence": "generalized markers"},
            "task_utility": {"score": 0.5, "evidence": "some useful detail remains"},
        }),
        latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="end_turn",
        cost_usd=0.001, raw_meta={},
    )
    judge_cfg = ModelConfig(model_id="claude-opus-4-7@v1", backend="anthropic",
                             api_model="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY",
                             params={"max_tokens": 2048})
    run_llm_judge(adapter=fake, judge_cfg=judge_cfg, rubric_path=RUBRIC_V2,
                  vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.rubric_version == "v7"
    assert j.scores["fingerprint_leak_free"].score == 1.0
    assert j.scores["task_utility"].score == 0.5

    user_msg = fake.generate.call_args.args[0][1].content
    assert "fingerprint_leak_free, task_utility" in user_msg
