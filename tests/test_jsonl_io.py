import json
from pathlib import Path
import pytest
from pipeline.jsonl_io import read_jsonl, write_jsonl, append_jsonl_idempotent
from pipeline.schemas import Sample, GroundTruth


def _make_sample(sid: str) -> Sample:
    return Sample(
        sample_id=sid,
        complexity="single_post",
        bucket="only_username",
        content="hi",
        ground_truth=GroundTruth(usernames=[], user_mentions=[], fingerprint_markers=[], cross_sample_users=[]),
        source_meta={},
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    p = tmp_path / "samples.jsonl"
    items = [_make_sample("a"), _make_sample("b")]
    write_jsonl(p, items)
    loaded = list(read_jsonl(p, Sample))
    assert [s.sample_id for s in loaded] == ["a", "b"]


def test_write_overwrites(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [_make_sample("a")])
    write_jsonl(p, [_make_sample("b")])
    loaded = list(read_jsonl(p, Sample))
    assert [s.sample_id for s in loaded] == ["b"]


def test_idempotent_append_skips_duplicates(tmp_path: Path):
    p = tmp_path / "y.jsonl"
    write_jsonl(p, [_make_sample("a")])
    n_added = append_jsonl_idempotent(p, [_make_sample("a"), _make_sample("b")], key="sample_id")
    assert n_added == 1
    loaded = list(read_jsonl(p, Sample))
    assert [s.sample_id for s in loaded] == ["a", "b"]


def test_idempotent_append_creates_missing_file(tmp_path: Path):
    p = tmp_path / "z.jsonl"
    n_added = append_jsonl_idempotent(p, [_make_sample("c")], key="sample_id")
    assert n_added == 1
    loaded = list(read_jsonl(p, Sample))
    assert loaded[0].sample_id == "c"


def test_read_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps(_make_sample("a").model_dump()) + "\n\n" + json.dumps(_make_sample("b").model_dump()) + "\n")
    loaded = list(read_jsonl(p, Sample))
    assert len(loaded) == 2
