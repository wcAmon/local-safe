from pathlib import Path
from unittest.mock import MagicMock
from pipeline.stages.stage2_runner import run_single_shot, output_id_for, build_pii_matcher_from_mapping
from pipeline.schemas import Sample, GroundTruth, Output
from pipeline.config import ModelConfig, PromptConfig
from pipeline.jsonl_io import write_jsonl, read_jsonl
from pipeline.serving.base import ModelResponse


def _sample(sid: str, content: str, author: str = "alice_92") -> Sample:
    return Sample(
        sample_id=sid,
        complexity="single_post",
        bucket="only_username",
        content=content,
        ground_truth=GroundTruth(usernames=[author], user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
        source_meta={"author": author},
    )


def test_output_id_is_deterministic():
    a = output_id_for("m@v1", "p0", "s1", 42)
    b = output_id_for("m@v1", "p0", "s1", 42)
    assert a == b
    c = output_id_for("m@v1", "p0", "s2", 42)
    assert a != c


def test_run_writes_both_artifacts(tmp_path: Path):
    vault = tmp_path / "vault"
    artifacts = tmp_path / "artifacts"
    vault.mkdir()
    artifacts.mkdir()

    samples = [_sample("s1", "alice_92 said hi.")]
    write_jsonl(vault / "samples_raw.jsonl", samples)
    # mapping that recognizes alice_92
    from pipeline.stages.stage1_dataset import MappingRow
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake_adapter = MagicMock()
    fake_adapter.model_id = "m@v1"
    fake_adapter.generate.return_value = ModelResponse(
        content="The user alice_92 was friendly.",  # leaks!
        latency_ms=100, tokens_in=10, tokens_out=10, finish_reason="stop", cost_usd=0.0, raw_meta={},
    )

    model_cfg = ModelConfig(model_id="m@v1", backend="openai_compat", api_model="m", base_url_env="X", params={"seed": 42})
    prompt = PromptConfig(prompt_id="p0", strength=0, template="Process: {content}")

    n = run_single_shot(
        adapter=fake_adapter,
        model_cfg=model_cfg,
        prompts=[prompt],
        samples=samples,
        vault_dir=vault,
        artifacts_dir=artifacts,
        salt="test-salt",
    )
    assert n == 1
    raws = list(read_jsonl(vault / "outputs_raw.jsonl", Output))
    redacted = list(read_jsonl(artifacts / "outputs_redacted.jsonl", Output))
    assert len(raws) == len(redacted) == 1
    assert "alice_92" in raws[0].response
    assert "alice_92" not in redacted[0].response
    assert "<<LEAKED:U-deadbe>>" in redacted[0].response
    assert redacted[0].leaked_refs == ["<<U-deadbe>>"]


def test_run_is_idempotent(tmp_path: Path):
    vault = tmp_path / "vault"
    artifacts = tmp_path / "artifacts"
    vault.mkdir()
    artifacts.mkdir()
    samples = [_sample("s1", "alice_92 said hi.")]
    write_jsonl(vault / "samples_raw.jsonl", samples)
    from pipeline.stages.stage1_dataset import MappingRow
    write_jsonl(vault / "mapping.jsonl", [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake_adapter = MagicMock()
    fake_adapter.model_id = "m@v1"
    fake_adapter.generate.return_value = ModelResponse(
        content="user_001 was friendly.",
        latency_ms=100, tokens_in=10, tokens_out=10, finish_reason="stop", cost_usd=0.0, raw_meta={},
    )
    model_cfg = ModelConfig(model_id="m@v1", backend="openai_compat", api_model="m", base_url_env="X", params={"seed": 42})
    prompt = PromptConfig(prompt_id="p0", strength=0, template="Process: {content}")

    n1 = run_single_shot(adapter=fake_adapter, model_cfg=model_cfg, prompts=[prompt], samples=samples,
                        vault_dir=vault, artifacts_dir=artifacts, salt="t")
    n2 = run_single_shot(adapter=fake_adapter, model_cfg=model_cfg, prompts=[prompt], samples=samples,
                        vault_dir=vault, artifacts_dir=artifacts, salt="t")
    assert n1 == 1 and n2 == 0
    # adapter called only once total
    assert fake_adapter.generate.call_count == 1
