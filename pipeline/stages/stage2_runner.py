"""Stage 2 — single-shot runner.

Iterates (model, prompt, sample) cells, calls the adapter, redacts output
on write, and produces vault/outputs_raw.jsonl + artifacts/outputs_redacted.jsonl.

Idempotent via deterministic output_id; existing rows are skipped.
"""

from __future__ import annotations
import datetime as dt
import hashlib
from pathlib import Path
from pipeline.schemas import Sample, Output, OutputMeta
from pipeline.config import ModelConfig, PromptConfig
from pipeline.serving.base import ModelAdapter, Message
from pipeline.pii.matcher import PIIMatcher
from pipeline.pii.tokens import PIIKind
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent
from pipeline.stages.stage1_dataset import MappingRow


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
) -> int:
    """Run all (prompt, sample) cells for the given adapter. Returns rows added."""
    seed = int(model_cfg.params.get("seed", 0))
    raw_path = vault_dir / "outputs_raw.jsonl"
    red_path = artifacts_dir / "outputs_redacted.jsonl"

    matcher = build_pii_matcher_from_mapping(vault_dir / "mapping.jsonl", salt=salt)

    raw_rows: list[Output] = []
    redacted_rows: list[Output] = []

    existing_raw = _existing_output_ids(raw_path)

    for prompt in prompts:
        for sample in samples:
            oid = output_id_for(model_cfg.model_id, prompt.prompt_id, sample.sample_id, seed)
            if oid in existing_raw:
                continue
            rendered = prompt.template.format(content=sample.content)
            resp = adapter.generate(
                [Message(role="user", content=rendered)],
                params=model_cfg.params,
                request_id=oid,
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
