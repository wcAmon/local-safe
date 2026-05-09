from pathlib import Path
import json
from pipeline.stages.stage1_dataset import build_samples
from pipeline.schemas import Sample
from pipeline.jsonl_io import read_jsonl


PII_MARKERS = ["新莊", "台積電", "內湖", "欸真的假的", "台北", "軟體工程師"]


def test_cross_thread_bucket_assigned_when_author_repeats(tmp_path: Path):
    reddit = tmp_path / "reddit.jsonl"
    rows = [
        {"post_id": "a1", "author": "alice_92", "subreddit": "r/x", "title": "t", "body": "hello", "scraped_at": "2026-04-01T00:00:00Z"},
        {"post_id": "a2", "author": "alice_92", "subreddit": "r/y", "title": "t", "body": "hi 新莊", "scraped_at": "2026-04-01T01:00:00Z"},
        {"post_id": "b1", "author": "bob_dev", "subreddit": "r/z", "title": "t", "body": "hello once", "scraped_at": "2026-04-01T02:00:00Z"},
    ]
    with reddit.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    build_samples(reddit_path=reddit, vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
                  salt="t", pii_markers=PII_MARKERS)

    raws = {s.sample_id: s for s in read_jsonl(tmp_path / "v" / "samples_raw.jsonl", Sample)}
    # Both alice samples → cross_thread (overrides with_pii of a2)
    assert raws["rd_a1_singlepost"].bucket == "cross_thread"
    assert raws["rd_a2_singlepost"].bucket == "cross_thread"
    # bob appears once → keeps original (only_username here)
    assert raws["rd_b1_singlepost"].bucket == "only_username"


def test_cross_thread_overrides_with_pii(tmp_path: Path):
    """Even if the body contains PII markers, recurring author wins."""
    reddit = tmp_path / "reddit.jsonl"
    rows = [
        {"post_id": "a1", "author": "alice_92", "subreddit": "r/x", "title": "t", "body": "hi 新莊", "scraped_at": "2026-04-01T00:00:00Z"},
        {"post_id": "a2", "author": "alice_92", "subreddit": "r/y", "title": "t", "body": "hi 台積電", "scraped_at": "2026-04-01T01:00:00Z"},
    ]
    with reddit.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    build_samples(reddit_path=reddit, vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
                  salt="t", pii_markers=PII_MARKERS)

    raws = {s.sample_id: s for s in read_jsonl(tmp_path / "v" / "samples_raw.jsonl", Sample)}
    assert raws["rd_a1_singlepost"].bucket == "cross_thread"
    assert raws["rd_a2_singlepost"].bucket == "cross_thread"


def test_cross_sample_users_populated_for_recurring_author(tmp_path: Path):
    reddit = tmp_path / "reddit.jsonl"
    rows = [
        {"post_id": "a1", "author": "alice_92", "subreddit": "r/x", "title": "t", "body": "hi", "scraped_at": "2026-04-01T00:00:00Z"},
        {"post_id": "a2", "author": "alice_92", "subreddit": "r/y", "title": "t", "body": "hi", "scraped_at": "2026-04-01T01:00:00Z"},
    ]
    with reddit.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    build_samples(reddit_path=reddit, vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
                  salt="t", pii_markers=PII_MARKERS)

    raws = {s.sample_id: s for s in read_jsonl(tmp_path / "v" / "samples_raw.jsonl", Sample)}
    assert raws["rd_a1_singlepost"].ground_truth.cross_sample_users == ["alice_92"]
    assert raws["rd_a2_singlepost"].ground_truth.cross_sample_users == ["alice_92"]
