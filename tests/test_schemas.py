import pytest
from pydantic import ValidationError
from pipeline.schemas import (
    Sample, GroundTruth, FingerprintMarker, SamplesManifest,
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
