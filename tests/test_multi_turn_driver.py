from pathlib import Path
from unittest.mock import MagicMock
from pipeline.schemas import Sample, GroundTruth, Scenario, UserTurn, Trace
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelResponse
from pipeline.pii.matcher import PIIMatcher
from pipeline.pii.tokens import PIIKind
from pipeline.runner.drivers.multi_turn import run_multi_turn, trace_id_for
from pipeline.jsonl_io import write_jsonl, read_jsonl
from pipeline.stages.stage1_dataset import MappingRow


def _matcher(entries=None):
    return PIIMatcher.build(entries=entries or [("alice_92", PIIKind.USERNAME)], salt="t")


def _sample(sid: str = "rd_s1", content: str = "alice_92 hi") -> Sample:
    return Sample(
        sample_id=sid, complexity="single_post", bucket="only_username",
        content=content, source_meta={"author": "alice_92"},
        ground_truth=GroundTruth(usernames=["alice_92"], user_mentions=[],
                                  fingerprint_markers=[], cross_sample_users=[]),
    )


def _model_cfg(seed: int = 42) -> ModelConfig:
    return ModelConfig(model_id="m@v1", backend="openai_compat",
                        api_model="m", base_url_env="X",
                        params={"temperature": 0.0, "seed": seed, "max_tokens": 100})


def test_trace_id_is_deterministic():
    a = trace_id_for("m@v1", "mt_001", 42)
    b = trace_id_for("m@v1", "mt_001", 42)
    c = trace_id_for("m@v1", "mt_002", 42)
    assert a == b
    assert a != c


def test_run_writes_trace_with_alternating_input_output_steps(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake = MagicMock()
    fake.model_id = "m@v1"
    fake.generate.side_effect = [
        ModelResponse(content="user_001 was friendly.", latency_ms=10,
                      tokens_in=5, tokens_out=5, finish_reason="stop", cost_usd=0.0, raw_meta={}),
        ModelResponse(content="continuing...", latency_ms=10,
                      tokens_in=8, tokens_out=3, finish_reason="stop", cost_usd=0.0, raw_meta={}),
    ]
    scenario = Scenario(
        scenario_id="mt_001_test", session_kind="multi_turn",
        sample_id="rd_s1",
        user_script=[
            UserTurn(user_turn=0, template="please anonymize: {content}"),
            UserTurn(user_turn=2, template="thanks!"),
        ],
    )

    n = run_multi_turn(
        adapter=fake, model_cfg=_model_cfg(), scenarios=[scenario],
        samples_by_id={"rd_s1": _sample()},
        vault_dir=vault, artifacts_dir=artifacts, salt="t",
    )
    assert n == 1
    traces = list(read_jsonl(artifacts / "traces.jsonl", Trace))
    assert len(traces) == 1
    t = traces[0]
    assert t.session_kind == "multi_turn"
    assert t.scenario_id == "mt_001_test"
    # 2 user turns × (1 input + 1 output step each) = 4 steps
    assert len(t.steps) == 4
    assert t.steps[0].kind == "input"
    assert t.steps[1].kind == "output"
    assert t.steps[2].kind == "input"
    assert t.steps[3].kind == "output"


def test_run_redacts_leak_in_output_steps(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake = MagicMock()
    fake.model_id = "m@v1"
    fake.generate.return_value = ModelResponse(
        content="alice_92 was friendly.", latency_ms=10, tokens_in=5, tokens_out=5,
        finish_reason="stop", cost_usd=0.0, raw_meta={},
    )
    scenario = Scenario(
        scenario_id="mt_leak_test", session_kind="multi_turn",
        sample_id="rd_s1",
        user_script=[UserTurn(user_turn=0, template="please anonymize: {content}")],
    )
    run_multi_turn(adapter=fake, model_cfg=_model_cfg(), scenarios=[scenario],
                   samples_by_id={"rd_s1": _sample()},
                   vault_dir=vault, artifacts_dir=artifacts, salt="t")

    t = list(read_jsonl(artifacts / "traces.jsonl", Trace))[0]
    assistant_step = t.steps[1]
    assert "alice_92" not in assistant_step.content_referenced
    assert "<<LEAKED:U-deadbe>>" in assistant_step.content_referenced
    assert "<<U-deadbe>>" in assistant_step.leaked_pii_refs
    # exposure ledger updated
    assert "<<U-deadbe>>" in t.exposure_ledger
    assert t.exposure_ledger["<<U-deadbe>>"].first_leaked_step == 1
    assert t.exposure_ledger["<<U-deadbe>>"].leak_count == 1


def test_run_is_idempotent(tmp_path: Path):
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake = MagicMock()
    fake.model_id = "m@v1"
    fake.generate.return_value = ModelResponse(
        content="user_001", latency_ms=10, tokens_in=5, tokens_out=2,
        finish_reason="stop", cost_usd=0.0, raw_meta={},
    )
    scenario = Scenario(
        scenario_id="mt_idem", session_kind="multi_turn", sample_id="rd_s1",
        user_script=[UserTurn(user_turn=0, template="x")],
    )
    n1 = run_multi_turn(adapter=fake, model_cfg=_model_cfg(), scenarios=[scenario],
                       samples_by_id={"rd_s1": _sample()},
                       vault_dir=vault, artifacts_dir=artifacts, salt="t")
    n2 = run_multi_turn(adapter=fake, model_cfg=_model_cfg(), scenarios=[scenario],
                       samples_by_id={"rd_s1": _sample()},
                       vault_dir=vault, artifacts_dir=artifacts, salt="t")
    assert n1 == 1 and n2 == 0
    assert fake.generate.call_count == 1


def test_visible_pii_refs_so_far_is_cumulative(tmp_path: Path):
    """User introduces alice_92 in turn 0; turn 2 user msg has no PII;
    visible_pii_refs_so_far on turn-2 input step still includes it."""
    vault = tmp_path / "v"; artifacts = tmp_path / "a"
    vault.mkdir(); artifacts.mkdir()
    write_jsonl(vault / "mapping.jsonl",
                [MappingRow(raw="alice_92", token="<<U-deadbe>>", kind="username")])

    fake = MagicMock()
    fake.model_id = "m@v1"
    fake.generate.return_value = ModelResponse(
        content="ok", latency_ms=1, tokens_in=1, tokens_out=1,
        finish_reason="stop", cost_usd=0.0, raw_meta={},
    )
    scenario = Scenario(
        scenario_id="mt_vis", session_kind="multi_turn", sample_id="rd_s1",
        user_script=[
            UserTurn(user_turn=0, template="here: {content}"),
            UserTurn(user_turn=2, template="thanks"),
        ],
    )
    run_multi_turn(adapter=fake, model_cfg=_model_cfg(), scenarios=[scenario],
                   samples_by_id={"rd_s1": _sample()},
                   vault_dir=vault, artifacts_dir=artifacts, salt="t")
    t = list(read_jsonl(artifacts / "traces.jsonl", Trace))[0]
    # step 2 is the second user input. visible_so_far should include alice's token from step 0.
    assert "<<U-deadbe>>" in t.steps[2].visible_pii_refs_so_far
