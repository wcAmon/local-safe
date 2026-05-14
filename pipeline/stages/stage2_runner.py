"""Stage 2 — single-shot runner.

Iterates (model, prompt, sample) cells, calls the adapter, redacts output
on write, and produces vault/outputs_raw.jsonl + artifacts/outputs_redacted.jsonl.

Idempotent via deterministic output_id; existing rows are skipped.
"""

from __future__ import annotations
import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from pipeline.schemas import Sample, Output, OutputMeta
from pipeline.config import ModelConfig, PromptConfig
from pipeline.serving.base import ModelAdapter, Message
from pipeline.pii.matcher import PIIMatcher
from pipeline.pii.tokens import PIIKind
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent
from pipeline.stages.stage1_dataset import MappingRow

if TYPE_CHECKING:
    from pipeline.serving.budget import BudgetGuard

logger = logging.getLogger(__name__)


def output_id_for(model_id: str, prompt_id: str, sample_id: str, seed: int) -> str:
    h = hashlib.sha256()
    h.update(f"{model_id}|{prompt_id}|{sample_id}|{seed}".encode("utf-8"))
    return h.hexdigest()[:16]


def build_pii_matcher_from_mapping(mapping_path: Path, salt: str) -> PIIMatcher:
    rows = list(read_jsonl(mapping_path, MappingRow))
    matcher = PIIMatcher(salt=salt)
    for r in rows:
        matcher.raw_to_token[r.raw] = r.token
        matcher.token_to_raw[r.token] = r.raw
        try:
            matcher.raw_to_kind[r.raw] = PIIKind(r.kind)
        except ValueError:
            matcher.raw_to_kind[r.raw] = PIIKind.LOCATION
    return matcher


def _existing_output_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        import json
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.add(json.loads(line)["output_id"])
    return out


def run_single_shot(
    *,
    adapter: ModelAdapter,
    model_cfg: ModelConfig,
    prompts: list[PromptConfig],
    samples: list[Sample],
    vault_dir: Path,
    artifacts_dir: Path,
    salt: str,
    budget_guard: "BudgetGuard | None" = None,
) -> int:
    """Run all (prompt, sample) cells for the given adapter. Returns rows added.

    If a `budget_guard` is supplied, costs are recorded against `model_cfg.model_id`
    and the per-model cap halts the loop early (stop_and_report). Halt is best-effort
    at the granularity of a single (prompt, sample) cell; partial rows are flushed.
    """
    seed = int(model_cfg.params.get("seed", 0))
    raw_path = vault_dir / "outputs_raw.jsonl"
    red_path = artifacts_dir / "outputs_redacted.jsonl"

    matcher = build_pii_matcher_from_mapping(vault_dir / "mapping.jsonl", salt=salt)

    raw_rows: list[Output] = []
    redacted_rows: list[Output] = []

    existing_raw = _existing_output_ids(raw_path)
    halted = False

    for prompt in prompts:
        if halted:
            break
        for sample in samples:
            oid = output_id_for(model_cfg.model_id, prompt.prompt_id, sample.sample_id, seed)
            if oid in existing_raw:
                continue
            if budget_guard and not budget_guard.check_before_call(model_cfg.model_id):
                logger.warning(
                    "[%s] budget cap reached; halting single-shot loop with partial outputs",
                    model_cfg.model_id,
                )
                halted = True
                break
            rendered = prompt.template.format(content=sample.content)
            resp = adapter.generate(
                [Message(role="user", content=rendered)],
                params=model_cfg.params,
                request_id=oid,
            )
            if budget_guard and resp.cost_usd:
                budget_guard.record(
                    judge_id=model_cfg.model_id, cost_usd=resp.cost_usd,
                    output_id=oid, stage="single_shot",
                    tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                )
            redacted_text, leaked = matcher.redact_output(resp.content, partial=True)

            meta = OutputMeta(
                latency_ms=resp.latency_ms,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                finish_reason=resp.finish_reason,
                ran_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            raw_rows.append(Output(
                output_id=oid, model_id=model_cfg.model_id, prompt_id=prompt.prompt_id,
                sample_id=sample.sample_id, rendered_prompt=rendered, response=resp.content,
                leaked_refs=leaked, metadata=meta,
            ))
            redacted_rows.append(Output(
                output_id=oid, model_id=model_cfg.model_id, prompt_id=prompt.prompt_id,
                sample_id=sample.sample_id, rendered_prompt=rendered, response=redacted_text,
                leaked_refs=leaked, metadata=meta,
            ))

    n_added_raw = append_jsonl_idempotent(raw_path, raw_rows, key="output_id")
    append_jsonl_idempotent(red_path, redacted_rows, key="output_id")
    return n_added_raw
