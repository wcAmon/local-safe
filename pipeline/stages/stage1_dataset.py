"""Stage 1 — build the sample set from raw reddit JSONL.

Produces:
- vault/samples_raw.jsonl       (Sample with original content)
- artifacts/samples_referenced.jsonl (Sample with content replaced by tokens)
- vault/mapping.jsonl           (one row per raw → token entry)
"""

from __future__ import annotations
import json
import hashlib
import datetime as dt
from pathlib import Path
from pydantic import BaseModel
from pipeline.schemas import (
    Sample, GroundTruth, FingerprintMarker, UserMention, SamplesManifest,
)
from pipeline.pii.tokens import PIIKind
from pipeline.pii.matcher import PIIMatcher
from pipeline.jsonl_io import write_jsonl


# Phase 1 curated fingerprint marker list. Phase 3 will replace with broader detection.
DEFAULT_PII_MARKERS = ["新莊", "台積電", "內湖", "欸真的假的", "台北", "軟體工程師"]


class MappingRow(BaseModel):
    raw: str
    token: str
    kind: str


def classify_bucket(body: str, markers: list[str]) -> str:
    return "with_pii" if any(m in body for m in markers) else "only_username"


def _classify_marker(text: str) -> PIIKind:
    """Heuristic mapping from marker text to PIIKind for Phase 1."""
    if text in ("台積電",):
        return PIIKind.ORGANIZATION
    if text in ("新莊", "內湖", "台北"):
        return PIIKind.LOCATION
    if text in ("軟體工程師",):
        return PIIKind.OCCUPATION
    if text in ("欸真的假的",):
        return PIIKind.WRITING_STYLE
    return PIIKind.LOCATION  # default conservative


def _hash_samples(samples: list[Sample]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s.model_dump_json().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_samples(
    *,
    reddit_path: Path,
    vault_dir: Path,
    artifacts_dir: Path,
    salt: str,
    pii_markers: list[str] | None = None,
) -> SamplesManifest:
    pii_markers = pii_markers if pii_markers is not None else DEFAULT_PII_MARKERS

    # Phase 2: pre-scan authors so we can mark cross_thread bucket for recurring ones.
    from collections import Counter
    author_counts: Counter[str] = Counter()
    with reddit_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            author_counts[row["author"]] += 1
    recurring_authors = {a for a, n in author_counts.items() if n >= 2}

    raw_samples: list[Sample] = []
    referenced_samples: list[Sample] = []
    mapping_rows: list[MappingRow] = []
    seen_raw_token: dict[str, str] = {}

    with reddit_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            post_id = row["post_id"]
            author = row["author"]
            body = row.get("body", "")

            # PII entries: author + any markers present in body
            entries: list[tuple[str, PIIKind]] = [(author, PIIKind.USERNAME)]
            present_markers = [m for m in pii_markers if m in body]
            for m in present_markers:
                entries.append((m, _classify_marker(m)))

            matcher = PIIMatcher.build(entries=entries, salt=salt)

            # Record any new (raw, token, kind) we have not seen before
            for raw, kind in entries:
                tok = matcher.raw_to_token[raw]
                if raw not in seen_raw_token:
                    seen_raw_token[raw] = tok
                    mapping_rows.append(MappingRow(raw=raw, token=tok, kind=kind.value))

            # Build fingerprint markers ground-truth
            fp_markers = []
            for m in present_markers:
                start = body.find(m)
                fp_markers.append(FingerprintMarker(
                    type={
                        PIIKind.ORGANIZATION: "organization",
                        PIIKind.LOCATION: "location",
                        PIIKind.OCCUPATION: "occupation",
                        PIIKind.WRITING_STYLE: "writing_style",
                    }.get(_classify_marker(m), "other"),
                    text=m,
                    span=(start, start + len(m)),
                ))

            # User mentions ground-truth
            mentions = []
            spans: list[tuple[int, int]] = []
            idx = 0
            while True:
                i = body.find(author, idx)
                if i < 0:
                    break
                spans.append((i, i + len(author)))
                idx = i + len(author)
            if spans:
                mentions.append(UserMention(username=author, spans=spans))

            bucket = classify_bucket(body, pii_markers)
            if author in recurring_authors:
                bucket = "cross_thread"
            sample_id = f"rd_{post_id}_singlepost"

            cross_users = [author] if author in recurring_authors else []
            gt = GroundTruth(
                usernames=[author],
                user_mentions=mentions,
                fingerprint_markers=fp_markers,
                cross_sample_users=cross_users,
            )

            raw_samples.append(Sample(
                sample_id=sample_id,
                complexity="single_post",
                bucket=bucket,
                content=body,
                ground_truth=gt,
                source_meta={"post_id": post_id, "subreddit": row.get("subreddit"), "author": author},
            ))
            referenced_samples.append(Sample(
                sample_id=sample_id,
                complexity="single_post",
                bucket=bucket,
                content=matcher.to_referenced(body),
                ground_truth=gt,
                source_meta={"post_id": post_id, "subreddit": row.get("subreddit")},
            ))

    write_jsonl(vault_dir / "samples_raw.jsonl", raw_samples)
    write_jsonl(artifacts_dir / "samples_referenced.jsonl", referenced_samples)
    write_jsonl(vault_dir / "mapping.jsonl", mapping_rows)

    buckets: dict = {}
    complexities: dict = {}
    for s in raw_samples:
        buckets[s.bucket] = buckets.get(s.bucket, 0) + 1
        complexities[s.complexity] = complexities.get(s.complexity, 0) + 1

    manifest = SamplesManifest(
        n_samples=len(raw_samples),
        samples_hash=_hash_samples(raw_samples),
        buckets=buckets,
        complexities=complexities,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_jsonl(artifacts_dir / "samples_manifest.jsonl", [manifest])
    return manifest
