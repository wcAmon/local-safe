"""Stage 3b — LLM judge over redacted outputs + referenced samples.

The judge model never sees raw PII. It scores four dimensions and we expect
JSON-only output (per rubric system prompt). Robust parsing handles cases
where the model wraps JSON in a ```json fence.
"""

from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
import yaml
from pipeline.schemas import Output, Sample, Judgment, JudgeScore
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelAdapter, Message
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent


SCORE_KEYS = ["username_replaced", "id_format_used", "governance_depth", "fingerprint_warning"]


def _judgment_id(output_id: str, judge_id: str, rubric_version: str) -> str:
    h = hashlib.sha256()
    h.update(f"{output_id}|{judge_id}|{rubric_version}".encode("utf-8"))
    return h.hexdigest()[:16]


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_judge_json(text: str) -> dict:
    """Parse the judge's JSON response, tolerating code fences and leading prose."""
    stripped = text.strip()
    # Try direct first.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Try fenced code block.
    m = _FENCE_RE.search(stripped)
    if m:
        return json.loads(m.group(1))
    # Last-ditch: find the outermost { ... } block.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError(f"could not extract JSON from judge response: {text[:200]!r}")


def _existing_judgment_keys(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (output_id, judge_id, rubric_version) already judged."""
    if not path.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.add((row["output_id"], row["judge_id"], row["rubric_version"]))
    return out


def run_llm_judge(
    *,
    adapter: ModelAdapter,
    judge_cfg: ModelConfig,
    rubric_path: Path,
    vault_dir: Path,
    artifacts_dir: Path,
) -> int:
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    rubric_version = rubric["version"]
    sys_prompt = rubric["system_prompt"]
    user_template = rubric["user_template"]

    # Use redacted outputs (judge never sees raw PII)
    outputs = {o.output_id: o for o in read_jsonl(artifacts_dir / "outputs_redacted.jsonl", Output)}
    samples = {s.sample_id: s for s in read_jsonl(artifacts_dir / "samples_referenced.jsonl", Sample)}

    judgments_path = artifacts_dir / "judgments.jsonl"
    existing = _existing_judgment_keys(judgments_path)

    new: list[Judgment] = []
    for output_id, output in outputs.items():
        if (output_id, judge_cfg.model_id, rubric_version) in existing:
            continue
        sample = samples[output.sample_id]
        user_msg = user_template.format(
            referenced_input=sample.content,
            redacted_output=output.response,
        )
        resp = adapter.generate(
            [Message(role="system", content=sys_prompt), Message(role="user", content=user_msg)],
            params=judge_cfg.params, request_id=f"judge-{output_id}",
        )
        try:
            parsed = parse_judge_json(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            new.append(Judgment(
                judgment_id=_judgment_id(output_id, judge_cfg.model_id, rubric_version),
                output_id=output_id, judge_id=judge_cfg.model_id, rubric_version=rubric_version,
                scores={k: JudgeScore(score=0.0, evidence="parse_error") for k in SCORE_KEYS},
                judge_reasoning=resp.content[:500],
                judge_notes=f"parse_error: {e!s}",
            ))
            continue

        scores = {}
        for k in SCORE_KEYS:
            entry = parsed.get(k, {"score": 0.0, "evidence": "missing_in_response"})
            scores[k] = JudgeScore(score=float(entry.get("score", 0.0)),
                                    evidence=str(entry.get("evidence", "")))
        new.append(Judgment(
            judgment_id=_judgment_id(output_id, judge_cfg.model_id, rubric_version),
            output_id=output_id, judge_id=judge_cfg.model_id, rubric_version=rubric_version,
            scores=scores, judge_reasoning=resp.content,
        ))

    return append_jsonl_idempotent(judgments_path, new, key="judgment_id")
