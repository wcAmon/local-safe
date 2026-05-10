from pathlib import Path
from pipeline.schemas import (
    Output, OutputMeta, Sample, GroundTruth,
    Trace, Step, ExposureLedgerEntry, Judgment,
)
from pipeline.stages.stage3a_rule_judge import run_rule_judge
from pipeline.stages.stage1_dataset import MappingRow
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _meta() -> OutputMeta:
    return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop",
                       ran_at="2026-05-09T00:00:00Z")


def _sample(sid: str, *, complexity="single_post", bucket="only_username",
             usernames=None, cross_sample=None) -> Sample:
    return Sample(
        sample_id=sid, complexity=complexity, bucket=bucket, content="x", source_meta={},
        ground_truth=GroundTruth(usernames=usernames or [],
                                  user_mentions=[], fingerprint_markers=[],
                                  cross_sample_users=cross_sample or []),
    )


def test_id_consistency_single_thread_skipped(tmp_path: Path):
    """For single_post complexity, id_consistency is not scored (rule writes it as 1.0 by default)."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl",
                [Output(output_id="o1", model_id="m@v1", prompt_id="p0",
                        sample_id="s1", rendered_prompt="...", response="user_001",
                        leaked_refs=[], metadata=_meta())])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", usernames=["alice"])])
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice", token="<<U-x>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    # id_consistency is N/A for single_post → not present (or 1.0 if always emitted)
    if "id_consistency" in j.scores:
        assert j.scores["id_consistency"].score == 1.0


def test_id_consistency_multi_thread_consistent(tmp_path: Path):
    """multi_thread response uses same user_001 throughout → 1.0."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl",
                [Output(output_id="o1", model_id="m@v1", prompt_id="p0",
                        sample_id="s1", rendered_prompt="...",
                        response="Thread 1: user_001 said x. Thread 2: user_001 added y.",
                        leaked_refs=[], metadata=_meta())])
    write_jsonl(vault / "samples_raw.jsonl",
                [_sample("s1", complexity="multi_thread", bucket="cross_thread",
                          usernames=["alice"], cross_sample=["alice"])])
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice", token="<<U-x>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["id_consistency"].score == 1.0


def test_id_consistency_multi_thread_inconsistent(tmp_path: Path):
    """multi_thread response uses user_001 then user_002 → 0.5."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl",
                [Output(output_id="o1", model_id="m@v1", prompt_id="p0",
                        sample_id="s1", rendered_prompt="...",
                        response="Thread 1: user_001 said x. Thread 2: user_002 added y.",
                        leaked_refs=[], metadata=_meta())])
    write_jsonl(vault / "samples_raw.jsonl",
                [_sample("s1", complexity="multi_thread", bucket="cross_thread",
                          usernames=["alice"], cross_sample=["alice"])])
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice", token="<<U-x>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    # 2 distinct ids out of 2 occurrences → consistency = max_freq / total = 1/2 = 0.5
    assert j.scores["id_consistency"].score == 0.5


def test_rule_judge_reads_traces(tmp_path: Path):
    """Stage 3a now also processes traces.jsonl artifacts."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    # No Output, only a Trace
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(vault / "samples_raw.jsonl",
                [_sample("rd_s1", complexity="single_post", bucket="only_username", usernames=["alice"])])
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice", token="<<U-x>>", kind="username")])
    write_jsonl(vault / "traces_raw.jsonl",
                [Trace(
                    trace_id="t1", session_kind="multi_turn",
                    model_id="m@v1", scenario_id="mt_001", sample_id="rd_s1",
                    steps=[
                        Step(step=0, kind="input", subkind="user_message",
                              content_referenced="give me data"),
                        Step(step=1, kind="output", subkind="assistant_message",
                              leaked_pii_refs=[],
                              content_referenced="user_001 was friendly."),
                    ],
                    metadata=_meta(),
                )])

    n = run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    js = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))
    # one judgment for the trace
    assert n == 1
    assert js[0].output_id == "t1"            # trace_id reused as output_id key for judgment
    assert js[0].scores["username_replaced"].score == 1.0
    assert js[0].scores["id_format_used"].score == 1.0


def test_trace_pii_leak_free_catches_location_leak(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [])
    write_jsonl(vault / "samples_raw.jsonl",
                [_sample("rd_s1", complexity="single_post", bucket="with_pii", usernames=["alice"])])
    write_jsonl(vault / "mapping.jsonl", [
        MappingRow(raw="alice", token="<<U-x>>", kind="username"),
        MappingRow(raw="內湖", token="<<LOC-y>>", kind="location"),
    ])
    write_jsonl(vault / "traces_raw.jsonl",
                [Trace(
                    trace_id="t1", session_kind="multi_turn",
                    model_id="m@v1", scenario_id="mt_001", sample_id="rd_s1",
                    steps=[
                        Step(step=0, kind="input", subkind="user_message",
                              content_referenced="give me data"),
                        Step(step=1, kind="output", subkind="assistant_message",
                              leaked_pii_refs=["<<LOC-y>>"],
                              content_referenced="user_001 lives in 內湖."),
                    ],
                    metadata=_meta(),
                )])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["username_replaced"].score == 1.0
    assert j.scores["pii_leak_free"].score == 0.0
