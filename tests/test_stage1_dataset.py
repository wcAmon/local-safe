from pathlib import Path
import json
from pipeline.stages.stage1_dataset import build_samples, classify_bucket
from pipeline.schemas import Sample
from pipeline.jsonl_io import read_jsonl


PII_MARKERS = ["新莊", "台積電", "內湖", "欸真的假的", "台北", "軟體工程師"]


def test_classify_bucket_only_username():
    assert classify_bucket("just code, no markers", PII_MARKERS) == "only_username"


def test_classify_bucket_with_pii():
    assert classify_bucket("我在新莊", PII_MARKERS) == "with_pii"


def test_build_samples_writes_three_artifacts(tmp_path: Path, tiny_reddit_path: Path):
    out = build_samples(
        reddit_path=tiny_reddit_path,
        vault_dir=tmp_path / "vault",
        artifacts_dir=tmp_path / "artifacts",
        salt="test-salt",
        pii_markers=PII_MARKERS,
    )
    assert (tmp_path / "vault" / "samples_raw.jsonl").exists()
    assert (tmp_path / "artifacts" / "samples_referenced.jsonl").exists()
    assert (tmp_path / "vault" / "mapping.jsonl").exists()
    assert out.n_samples == 5


def test_build_samples_referenced_has_no_raw_pii(tmp_path: Path, tiny_reddit_path: Path):
    build_samples(
        reddit_path=tiny_reddit_path,
        vault_dir=tmp_path / "vault",
        artifacts_dir=tmp_path / "artifacts",
        salt="test-salt",
        pii_markers=PII_MARKERS,
    )
    referenced = list(read_jsonl(tmp_path / "artifacts" / "samples_referenced.jsonl", Sample))
    blob = " ".join(s.content for s in referenced)
    # known PII strings must be absent
    for raw in ("alice_92", "bob_dev", "新莊", "台積電", "內湖"):
        assert raw not in blob, f"raw PII {raw!r} leaked into referenced sample"


def test_build_samples_buckets_assigned(tmp_path: Path, tiny_reddit_path: Path):
    build_samples(
        reddit_path=tiny_reddit_path,
        vault_dir=tmp_path / "vault",
        artifacts_dir=tmp_path / "artifacts",
        salt="test-salt",
        pii_markers=PII_MARKERS,
    )
    raws = list(read_jsonl(tmp_path / "vault" / "samples_raw.jsonl", Sample))
    by_id = {s.sample_id: s for s in raws}
    # p001, p002, p004 hit markers; p003, p005 do not
    assert by_id["rd_p001_singlepost"].bucket == "with_pii"
    assert by_id["rd_p002_singlepost"].bucket == "with_pii"
    assert by_id["rd_p003_singlepost"].bucket == "only_username"
    assert by_id["rd_p004_singlepost"].bucket == "with_pii"
    assert by_id["rd_p005_singlepost"].bucket == "only_username"


def test_build_samples_ground_truth_has_authors(tmp_path: Path, tiny_reddit_path: Path):
    build_samples(
        reddit_path=tiny_reddit_path,
        vault_dir=tmp_path / "vault",
        artifacts_dir=tmp_path / "artifacts",
        salt="test-salt",
        pii_markers=PII_MARKERS,
    )
    raws = {s.sample_id: s for s in read_jsonl(tmp_path / "vault" / "samples_raw.jsonl", Sample)}
    assert "alice_92" in raws["rd_p001_singlepost"].ground_truth.usernames
    assert "bob_dev" in raws["rd_p002_singlepost"].ground_truth.usernames
