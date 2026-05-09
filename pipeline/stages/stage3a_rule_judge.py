"""Stage 3a — deterministic rule judge for hard signals.

Scores:
- username_replaced: 1.0 if zero username-kind tokens in leaked_refs, else 0.0
  (binary for Phase 1; future versions can do proportional scoring when
  multiple usernames per sample exist).
- id_format_used: 1.0 if the response contains either a synthetic ID pattern
  (user_NNN / participant_NNN / U_NNN) OR an opaque token (<<U-...>>), else 0.0.
"""

from __future__ import annotations
import re
import hashlib
from pathlib import Path
from pipeline.schemas import Output, Judgment, JudgeScore
from pipeline.stages.stage1_dataset import MappingRow
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent


JUDGE_ID = "rule_v1"
RUBRIC_VERSION = "v1"


_ID_PATTERNS = [
    re.compile(r"\buser_\d+\b", re.IGNORECASE),
    re.compile(r"\bparticipant_\d+\b", re.IGNORECASE),
    re.compile(r"\bU_\d+\b"),
    re.compile(r"<<U-[A-Fa-f0-9]+>>"),
]


def _judgment_id(output_id: str, judge_id: str, rubric_version: str) -> str:
    h = hashlib.sha256()
    h.update(f"{output_id}|{judge_id}|{rubric_version}".encode("utf-8"))
    return h.hexdigest()[:16]


def _score_username_replaced(output: Output, username_tokens: set[str]) -> JudgeScore:
    leaked_user_tokens = [r for r in output.leaked_refs if r in username_tokens]
    if not leaked_user_tokens:
        return JudgeScore(score=1.0, evidence="no username token leaked")
    return JudgeScore(score=0.0, evidence=f"leaked: {leaked_user_tokens}")


def _score_id_format_used(output: Output) -> JudgeScore:
    for pat in _ID_PATTERNS:
        if pat.search(output.response):
            return JudgeScore(score=1.0, evidence=f"matched pattern {pat.pattern!r}")
    return JudgeScore(score=0.0, evidence="no synthetic ID pattern found")


def run_rule_judge(*, vault_dir: Path, artifacts_dir: Path) -> int:
    outputs = list(read_jsonl(vault_dir / "outputs_raw.jsonl", Output))
    mapping = list(read_jsonl(vault_dir / "mapping.jsonl", MappingRow))
    username_tokens = {m.token for m in mapping if m.kind == "username"}

    judgments: list[Judgment] = []
    for o in outputs:
        scores = {
            "username_replaced": _score_username_replaced(o, username_tokens),
            "id_format_used": _score_id_format_used(o),
        }
        judgments.append(Judgment(
            judgment_id=_judgment_id(o.output_id, JUDGE_ID, RUBRIC_VERSION),
            output_id=o.output_id,
            judge_id=JUDGE_ID,
            rubric_version=RUBRIC_VERSION,
            scores=scores,
            judge_reasoning="deterministic rules; see per-score evidence",
        ))

    return append_jsonl_idempotent(
        artifacts_dir / "judgments.jsonl", judgments, key="judgment_id",
    )
