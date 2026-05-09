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


def test_multi_thread_emits_grouped_samples(tmp_path: Path, tiny_reddit_v2_path: Path):
    build_samples(
        reddit_path=tiny_reddit_v2_path,
        vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
        salt="t", pii_markers=PII_MARKERS, multi_thread=True,
    )
    raws = list(read_jsonl(tmp_path / "v" / "samples_raw.jsonl", Sample))
    by_id = {s.sample_id: s for s in raws}

    # 5 single_post samples + 2 multi_thread samples (alice_92 has 2 posts, bob_dev has 2 posts; charlie_42 only 1 → no group).
    assert len(raws) == 7
    assert "rd_multi_alice_92_multipost" in by_id
    assert "rd_multi_bob_dev_multipost" in by_id

    grouped = by_id["rd_multi_alice_92_multipost"]
    assert grouped.complexity == "multi_thread"
    assert grouped.bucket == "cross_thread"
    # content contains both posts with [Thread N] markers
    assert "[Thread 1]" in grouped.content
    assert "[Thread 2]" in grouped.content
    assert "alice_92" in grouped.content   # raw form in vault sample
    assert "新莊" in grouped.content


def test_multi_thread_referenced_has_no_raw(tmp_path: Path, tiny_reddit_v2_path: Path):
    build_samples(
        reddit_path=tiny_reddit_v2_path,
        vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
        salt="t", pii_markers=PII_MARKERS, multi_thread=True,
    )
    refs = list(read_jsonl(tmp_path / "a" / "samples_referenced.jsonl", Sample))
    grouped = next(s for s in refs if s.sample_id == "rd_multi_alice_92_multipost")
    assert "alice_92" not in grouped.content
    assert "新莊" not in grouped.content


def test_multi_thread_disabled_by_default(tmp_path: Path, tiny_reddit_v2_path: Path):
    build_samples(
        reddit_path=tiny_reddit_v2_path,
        vault_dir=tmp_path / "v", artifacts_dir=tmp_path / "a",
        salt="t", pii_markers=PII_MARKERS,  # no multi_thread arg
    )
    raws = list(read_jsonl(tmp_path / "v" / "samples_raw.jsonl", Sample))
    assert len(raws) == 5  # only single-post


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
