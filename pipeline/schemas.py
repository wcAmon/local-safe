"""Pydantic schemas for every JSONL artifact in the pipeline."""

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

Bucket = Literal["only_username", "with_pii", "cross_thread", "fingerprint_rich"]
Complexity = Literal["single_post", "single_thread", "multi_thread"]
SessionKind = Literal["single_shot", "multi_turn", "agent_loop", "long_context"]


class FingerprintMarker(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["location", "occupation", "writing_style", "time_pattern", "organization", "other"]
    text: str
    span: tuple[int, int]
    note: str | None = None


class UserMention(BaseModel):
    model_config = ConfigDict(frozen=True)
    username: str
    spans: list[tuple[int, int]]


class GroundTruth(BaseModel):
    usernames: list[str]
    user_mentions: list[UserMention]
    fingerprint_markers: list[FingerprintMarker]
    cross_sample_users: list[str]


class Sample(BaseModel):
    sample_id: str
    complexity: Complexity
    bucket: Bucket
    content: str
    ground_truth: GroundTruth
    source_meta: dict[str, Any]


class SamplesManifest(BaseModel):
    """Index file describing the samples set (count, hash, bucket distribution)."""
    n_samples: int
    samples_hash: str
    buckets: dict[Bucket, int]
    complexities: dict[Complexity, int]
    created_at: str


class OutputMeta(BaseModel):
    latency_ms: int
    tokens_in: int
    tokens_out: int
    finish_reason: str
    ran_at: str


class Output(BaseModel):
    output_id: str
    model_id: str
    prompt_id: str
    sample_id: str
    session_kind: SessionKind = "single_shot"
    rendered_prompt: str
    response: str
    leaked_refs: list[str] = Field(default_factory=list)
    metadata: OutputMeta


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    evidence: str


class Judgment(BaseModel):
    judgment_id: str
    output_id: str
    judge_id: str
    rubric_version: str
    scores: dict[str, JudgeScore]
    judge_reasoning: str = ""
    judge_notes: str = ""


class CellMetric(BaseModel):
    mean: float
    ci95: tuple[float, float]


class CellScore(BaseModel):
    cell_id: str
    model_id: str
    prompt_id: str
    complexity: Complexity
    bucket: Bucket
    n_samples: int
    metrics: dict[str, CellMetric]
