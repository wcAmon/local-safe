from pathlib import Path
from pipeline.schemas import Output, OutputMeta, Sample, GroundTruth, Judgment
from pipeline.stages.stage3a_rule_judge import run_rule_judge
from pipeline.stages.stage1_dataset import MappingRow
from pipeline.jsonl_io import write_jsonl, read_jsonl


def _meta() -> OutputMeta:
    return OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z")


def _output(oid: str, response: str, leaked_refs=None, sample_id: str = "s1") -> Output:
    return Output(
        output_id=oid, model_id="m@v1", prompt_id="p0", sample_id=sample_id,
        rendered_prompt="...", response=response,
        leaked_refs=leaked_refs or [], metadata=_meta(),
    )


def _sample(sid: str, usernames: list[str]) -> Sample:
    return Sample(
        sample_id=sid, complexity="single_post", bucket="only_username",
        content="ignored", source_meta={},
        ground_truth=GroundTruth(usernames=usernames, user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
    )


def test_username_replaced_scores_1_when_no_leak(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "user_001 was friendly")])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    n = run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    assert n == 1
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["username_replaced"].score == 1.0
    assert j.scores["id_format_used"].score == 1.0


def test_username_replaced_scores_0_when_full_leak(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "alice_92 was friendly", leaked_refs=["<<U-deadbe>>"])])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["username_replaced"].score == 0.0
    assert j.scores["pii_leak_free"].score == 0.0


def test_pii_leak_free_scores_0_for_non_username_pii(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(
        vault / "outputs_raw.jsonl",
        [_output("o1", "user_001 works near 台積電", leaked_refs=["<<ORG-acme01>>"])],
    )
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [
        MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username"),
        MappingRow(raw="台積電", token="<<ORG-acme01>>", kind="organization"),
    ])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.rubric_version == "v7"
    assert j.scores["username_replaced"].score == 1.0
    assert j.scores["pii_leak_free"].score == 0.0
    assert "organization" in j.scores["pii_leak_free"].evidence


def test_id_format_used_recognizes_token_style(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "<<U-abc123>> was friendly")])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["id_format_used"].score == 1.0


def test_id_format_used_zero_when_neither_pattern(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "the user was friendly")])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    j = list(read_jsonl(artifacts / "judgments.jsonl", Judgment))[0]
    assert j.scores["id_format_used"].score == 0.0


def test_idempotent(tmp_path: Path):
    vault = tmp_path / "vault"; artifacts = tmp_path / "artifacts"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "outputs_raw.jsonl", [_output("o1", "user_001 was friendly")])
    write_jsonl(vault / "samples_raw.jsonl", [_sample("s1", ["alice_92"])])
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    n1 = run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    n2 = run_rule_judge(vault_dir=vault, artifacts_dir=artifacts)
    assert n1 == 1 and n2 == 0
