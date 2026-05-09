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


GovernanceAction = Literal[
    "used_id_format",
    "replaced_partial",
    "kept_raw",
    "warned_about_pii",
    "warned_about_fingerprint",
    "refused_to_proceed",
    "asked_clarification",
]


class Step(BaseModel):
    step: int
    kind: Literal["input", "output"]
    subkind: Literal[
        "user_message", "assistant_message",
        "tool_call", "tool_result",
        "context_chunk", "summary_chunk",
    ]
    position: tuple[int, int] | None = None
    introduced_pii_refs: list[str] = Field(default_factory=list)
    visible_pii_refs_so_far: list[str] = Field(default_factory=list)
    emitted_pii_refs: list[str] = Field(default_factory=list)
    leaked_pii_refs: list[str] = Field(default_factory=list)
    governance_actions: list[GovernanceAction] = Field(default_factory=list)
    content_referenced: str


class ExposureLedgerEntry(BaseModel):
    introduced_step: int
    first_emitted_step: int | None = None
    first_leaked_step: int | None = None
    leak_count: int = 0
    consistency: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    trace_id: str
    session_kind: SessionKind
    model_id: str
    sample_id: str | None = None
    scenario_id: str
    rubric_version: str = "v1"
    steps: list[Step]
    exposure_ledger: dict[str, ExposureLedgerEntry] = Field(default_factory=dict)
    metadata: OutputMeta


class UserTurn(BaseModel):
    user_turn: int
    template: str


class Scenario(BaseModel):
    scenario_id: str
    session_kind: SessionKind
    sample_id: str | None = None
    user_script: list[UserTurn]


class JudgeAgreement(BaseModel):
    fleiss_kappa: float
    status: Literal["reliable", "moderate", "unreliable"]
    n_raters: int


class CellScore(BaseModel):
    cell_id: str
    model_id: str
    prompt_or_scenario_id: str
    complexity: Complexity
    bucket: Bucket
    session_kind: SessionKind
    n_samples: int
    metrics: dict[str, CellMetric]
    judge_agreement: JudgeAgreement | None = None
