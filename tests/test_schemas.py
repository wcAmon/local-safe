import pytest
from pydantic import ValidationError
from pipeline.schemas import (
    Sample, GroundTruth, FingerprintMarker,
    Output, OutputMeta,
    Judgment, JudgeScore,
    CellScore, CellMetric,
)


def test_sample_minimum_valid():
    s = Sample(
        sample_id="rd_001_singlepost_basic",
        complexity="single_post",
        bucket="only_username",
        content="hello",
        ground_truth=GroundTruth(
            usernames=["alice_92"],
            user_mentions=[],
            fingerprint_markers=[],
            cross_sample_users=[],
        ),
        source_meta={"post_id": "abc"},
    )
    assert s.bucket == "only_username"


def test_sample_rejects_unknown_bucket():
    with pytest.raises(ValidationError):
        Sample(
            sample_id="x",
            complexity="single_post",
            bucket="not_a_bucket",  # invalid
            content="",
            ground_truth=GroundTruth(usernames=[], user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
            source_meta={},
        )


def test_fingerprint_marker_types():
    m = FingerprintMarker(type="location", text="新莊", span=(0, 2))
    assert m.span == (0, 2)


def test_output_meta_required_fields():
    om = OutputMeta(latency_ms=100, tokens_in=10, tokens_out=20, finish_reason="stop", ran_at="2026-05-09T00:00:00Z")
    assert om.tokens_in == 10


def test_judgment_score_clamped():
    js = JudgeScore(score=0.5, evidence="ok")
    assert 0.0 <= js.score <= 1.0
    with pytest.raises(ValidationError):
        JudgeScore(score=1.5, evidence="too high")


def test_cell_metric_ci_shape():
    cm = CellMetric(mean=0.8, ci95=(0.7, 0.9))
    assert cm.ci95 == (0.7, 0.9)


from pipeline.schemas import (
    Step, Trace, ExposureLedgerEntry, GovernanceAction,
    Scenario, UserTurn, JudgeAgreement,
)
from pydantic import ValidationError as _VE  # noqa: F401  re-import for clarity


def test_step_minimum_valid():
    s = Step(step=0, kind="input", subkind="user_message", content_referenced="hi")
    assert s.governance_actions == []
    assert s.position is None


def test_step_governance_action_enum():
    Step(step=1, kind="output", subkind="assistant_message",
         governance_actions=["used_id_format", "warned_about_pii"],
         content_referenced="x")
    with pytest.raises(_VE):
        Step(step=1, kind="output", subkind="assistant_message",
             governance_actions=["bogus_action"], content_referenced="x")


def test_exposure_ledger_entry_defaults():
    e = ExposureLedgerEntry(introduced_step=0)
    assert e.first_emitted_step is None
    assert e.first_leaked_step is None
    assert e.leak_count == 0
    assert e.consistency == {}


def test_trace_minimum_valid():
    t = Trace(
        trace_id="abc", session_kind="multi_turn",
        model_id="m@v1", scenario_id="mt_001",
        steps=[Step(step=0, kind="input", subkind="user_message", content_referenced="hi")],
        metadata=OutputMeta(latency_ms=1, tokens_in=1, tokens_out=1, finish_reason="stop", ran_at="2026-05-09T00:00:00Z"),
    )
    assert t.session_kind == "multi_turn"
    assert t.sample_id is None  # nullable


def test_user_turn_minimum_valid():
    u = UserTurn(user_turn=0, template="hi {content}")
    assert u.user_turn == 0


def test_scenario_minimum_valid():
    s = Scenario(
        scenario_id="mt_001", session_kind="multi_turn",
        sample_id="rd_p001_singlepost",
        user_script=[UserTurn(user_turn=0, template="hi {content}")],
    )
    assert s.user_script[0].user_turn == 0


def test_judge_agreement_status_enum():
    JudgeAgreement(fleiss_kappa=0.8, status="reliable", n_raters=2)
    with pytest.raises(_VE):
        JudgeAgreement(fleiss_kappa=0.8, status="bogus", n_raters=2)


def test_cell_score_phase2_fields():
    cs = CellScore(
        cell_id="m@v1|p0|single_post|only_username|single_shot",
        model_id="m@v1",
        prompt_or_scenario_id="p0",
        complexity="single_post",
        bucket="only_username",
        session_kind="single_shot",
        n_samples=10,
        metrics={},
    )
    assert cs.judge_agreement is None
    cs.judge_agreement = JudgeAgreement(fleiss_kappa=0.7, status="reliable", n_raters=2)
    assert cs.judge_agreement.status == "reliable"
