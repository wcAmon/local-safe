"""Stage 3b — LLM judge over redacted outputs + referenced samples.

The judge model never sees raw PII. It scores four dimensions and we expect
JSON-only output (per rubric system prompt). Robust parsing handles cases
where the model wraps JSON in a ```json fence.

Robust to two failure modes that the gpt-oss-120b judge has shown in practice:
- The model returns a 500 with harmony-format tokens leaking through llama.cpp's
  parser (`<|channel|>...<|message|>...`). We classify these as `api_error` and
  emit a placeholder judgment instead of crashing the whole judge run.
- The model returns valid text but malformed JSON. Pre-existing `parse_error`
  path handles this.
"""

from __future__ import annotations
import json
import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
import yaml
from pipeline.schemas import Output, Sample, Judgment, JudgeScore
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelAdapter, Message
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent

if TYPE_CHECKING:
    from pipeline.serving.budget import BudgetGuard

logger = logging.getLogger(__name__)

# String hints that identify a llama.cpp / gpt-oss harmony-format parse failure.
# When the API raises an exception containing any of these, we record an
# api_error placeholder judgment and continue, rather than crashing the run.
_HARMONY_ERROR_HINTS = ("<|channel|>", "<|message|>", "<|constrain|>", "harmony")


def _is_harmony_api_error(exc: Exception) -> bool:
    """True iff the exception message looks like a llama.cpp harmony parse 500."""
    msg = str(exc)
    return any(hint in msg for hint in _HARMONY_ERROR_HINTS)


def _api_error_judgment(
    *, target_id: str, judge_id: str, rubric_version: str, exc: Exception,
) -> Judgment:
    """Build a placeholder judgment for a harmony-style API failure."""
    return Judgment(
        judgment_id=_judgment_id(target_id, judge_id, rubric_version),
        output_id=target_id,
        judge_id=judge_id,
        rubric_version=rubric_version,
        scores={k: JudgeScore(score=0.0, evidence="api_error") for k in SCORE_KEYS},
        judge_reasoning="",
        judge_notes=f"api_error: {str(exc)[:300]}",
    )


SCORE_KEYS_PHASE2 = ["username_replaced", "id_format_used",
                     "governance_depth", "fingerprint_leak_free",
                     "fingerprint_warning", "task_utility",
                     "multi_step_consistency"]
SCORE_KEYS_PHASE3 = SCORE_KEYS_PHASE2 + [
    "tool_args_minimal", "tool_result_governance", "prompt_injection_resisted",
    "cloud_search_clean", "reverse_leak_resisted",
]
SCORE_KEYS = SCORE_KEYS_PHASE3                     # superset

LLM_SCORED_DIMENSIONS = frozenset({
    "governance_depth", "fingerprint_leak_free", "fingerprint_warning",
    "task_utility", "multi_step_consistency",
    "tool_args_minimal", "tool_result_governance", "prompt_injection_resisted",
})

DEFAULT_DIMENSIONS_BY_KIND = {
    "single_shot": ["username_replaced", "id_format_used",
                     "governance_depth", "fingerprint_leak_free",
                     "fingerprint_warning", "task_utility"],
    "multi_turn":  ["username_replaced", "id_format_used",
                     "governance_depth", "fingerprint_leak_free",
                     "fingerprint_warning", "task_utility",
                     "multi_step_consistency"],
}


def _judgment_id(output_id: str, judge_id: str, rubric_version: str) -> str:
    h = hashlib.sha256()
    h.update(f"{output_id}|{judge_id}|{rubric_version}".encode("utf-8"))
    return h.hexdigest()[:16]


def _generate_with_retries(
    *,
    adapter: ModelAdapter,
    messages: list[Message],
    params: dict,
    request_id: str,
    attempts: int = 2,
):
    """Bounded retry wrapper for cloud judges.

    Timeouts are configured on the adapter client. This wrapper prevents one
    transient API error from aborting the whole judge batch.
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return adapter.generate(messages, params=params, request_id=request_id)
        except Exception as exc:  # provider SDKs use different exception trees
            last_exc = exc
            if attempt + 1 >= max(1, attempts):
                break
            logger.warning("judge call failed once for %s; retrying: %s", request_id, exc)
    assert last_exc is not None
    raise last_exc


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
    budget_guard: "BudgetGuard | None" = None,
    limit: int | None = None,
    flush_every: int = 10,
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
    total_added = 0

    def _queue(judgment: Judgment) -> bool:
        nonlocal total_added, new
        new.append(judgment)
        if len(new) >= max(1, flush_every):
            total_added += append_jsonl_idempotent(judgments_path, new, key="judgment_id")
            new = []
        return limit is not None and (total_added + len(new)) >= limit

    for output_id, output in outputs.items():
        if limit is not None and (total_added + len(new)) >= limit:
            break
        if (output_id, judge_cfg.model_id, rubric_version) in existing:
            continue
        if budget_guard and not budget_guard.check_before_call(judge_cfg.model_id):
            break        # stop_and_report; partial judgments still get appended below
        sample = samples[output.sample_id]
        if "{transcript}" in user_template:        # rubric.v2 path
            user_msg = user_template.format(
                referenced_input=sample.content,
                transcript=output.response,
                dimensions_csv=", ".join(DEFAULT_DIMENSIONS_BY_KIND["single_shot"]),
            )
        else:                                        # rubric.v1 path
            user_msg = user_template.format(
                referenced_input=sample.content,
                redacted_output=output.response,
            )
        try:
            resp = _generate_with_retries(
                adapter=adapter,
                messages=[
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=user_msg),
                ],
                params=judge_cfg.params, request_id=f"judge-{output_id}",
                attempts=int(judge_cfg.params.get("judge_attempts", 2)),
            )
        except Exception as e:
            kind = "harmony API error" if _is_harmony_api_error(e) else "API error"
            logger.warning("[%s] %s on output %s; emitting api_error placeholder",
                           judge_cfg.model_id, kind, output_id)
            if _queue(_api_error_judgment(
                target_id=output_id, judge_id=judge_cfg.model_id,
                rubric_version=rubric_version, exc=e,
            )):
                break
            continue
        if budget_guard and resp.cost_usd:
            budget_guard.record(
                judge_id=judge_cfg.model_id, cost_usd=resp.cost_usd,
                output_id=output_id,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cache_creation_input_tokens=(resp.raw_meta or {}).get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=(resp.raw_meta or {}).get("cache_read_input_tokens", 0),
            )
        try:
            parsed = parse_judge_json(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            if _queue(Judgment(
                judgment_id=_judgment_id(output_id, judge_cfg.model_id, rubric_version),
                output_id=output_id, judge_id=judge_cfg.model_id, rubric_version=rubric_version,
                scores={k: JudgeScore(score=0.0, evidence="parse_error") for k in SCORE_KEYS},
                judge_reasoning=resp.content[:500],
                judge_notes=f"parse_error: {e!s}",
            )):
                break
            continue

        scores = {}
        for k in SCORE_KEYS:
            entry = parsed.get(k, {"score": 0.0, "evidence": "missing_in_response"})
            if not isinstance(entry, dict):
                entry = {"score": float(entry) if isinstance(entry, (int, float)) else 0.0,
                         "evidence": "flat_format"}
            scores[k] = JudgeScore(score=float(entry.get("score", 0.0)),
                                    evidence=str(entry.get("evidence", "")))
        if _queue(Judgment(
            judgment_id=_judgment_id(output_id, judge_cfg.model_id, rubric_version),
            output_id=output_id, judge_id=judge_cfg.model_id, rubric_version=rubric_version,
            scores=scores, judge_reasoning=resp.content,
        )):
            break

    # Trace path (Phase 2 multi-turn)
    from pipeline.schemas import Trace
    traces = list(read_jsonl(artifacts_dir / "traces.jsonl", Trace))
    samples_for_traces = {s.sample_id: s for s in read_jsonl(artifacts_dir / "samples_referenced.jsonl", Sample)}
    for trace in traces:
        if limit is not None and (total_added + len(new)) >= limit:
            break
        if (trace.trace_id, judge_cfg.model_id, rubric_version) in existing:
            continue
        if budget_guard and not budget_guard.check_before_call(judge_cfg.model_id):
            break        # stop_and_report; partial judgments still get appended below
        # Compose a "transcript" string from the trace's assistant steps for the judge prompt.
        transcript_parts = []
        for s in trace.steps:
            role = "user" if s.kind == "input" else "assistant"
            transcript_parts.append(f"[{role}, step {s.step}] {s.content_referenced}")
        transcript = "\n\n".join(transcript_parts)

        sample_text = ""
        if trace.sample_id and trace.sample_id in samples_for_traces:
            sample_text = samples_for_traces[trace.sample_id].content

        # Determine which dimensions to ask LLM about
        trace_dims = trace.tested_dimensions or DEFAULT_DIMENSIONS_BY_KIND.get(trace.session_kind, [])
        llm_dims = [d for d in trace_dims if d in LLM_SCORED_DIMENSIONS]
        if not llm_dims:
            # Nothing for this LLM judge to score on this trace; skip the call entirely.
            continue

        if "{transcript}" in user_template:        # rubric.v2 path
            user_msg = user_template.format(
                referenced_input=sample_text or "(no shared sample)",
                transcript=transcript,
                dimensions_csv=", ".join(llm_dims),
            )
        else:                                        # rubric.v1 path (Phase 2 backward compat)
            user_msg = user_template.format(
                referenced_input=sample_text or "(no shared sample)",
                redacted_output=transcript,
            )
        try:
            resp = _generate_with_retries(
                adapter=adapter,
                messages=[
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=user_msg),
                ],
                params=judge_cfg.params, request_id=f"judge-{trace.trace_id}",
                attempts=int(judge_cfg.params.get("judge_attempts", 2)),
            )
        except Exception as e:
            kind = "harmony API error" if _is_harmony_api_error(e) else "API error"
            logger.warning("[%s] %s on trace %s; emitting api_error placeholder",
                           judge_cfg.model_id, kind, trace.trace_id)
            if _queue(_api_error_judgment(
                target_id=trace.trace_id, judge_id=judge_cfg.model_id,
                rubric_version=rubric_version, exc=e,
            )):
                break
            continue
        if budget_guard and resp.cost_usd:
            budget_guard.record(
                judge_id=judge_cfg.model_id, cost_usd=resp.cost_usd,
                output_id=trace.trace_id,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cache_creation_input_tokens=(resp.raw_meta or {}).get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=(resp.raw_meta or {}).get("cache_read_input_tokens", 0),
            )
        try:
            parsed = parse_judge_json(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            if _queue(Judgment(
                judgment_id=_judgment_id(trace.trace_id, judge_cfg.model_id, rubric_version),
                output_id=trace.trace_id, judge_id=judge_cfg.model_id,
                rubric_version=rubric_version,
                scores={k: JudgeScore(score=0.0, evidence="parse_error") for k in SCORE_KEYS},
                judge_reasoning=resp.content[:500],
                judge_notes=f"parse_error: {e!s}",
            )):
                break
            continue

        scores = {}
        for k in SCORE_KEYS:
            entry = parsed.get(k, {"score": 0.0, "evidence": "missing_in_response"})
            if not isinstance(entry, dict):
                entry = {"score": float(entry) if isinstance(entry, (int, float)) else 0.0,
                         "evidence": "flat_format"}
            scores[k] = JudgeScore(score=float(entry.get("score", 0.0)),
                                    evidence=str(entry.get("evidence", "")))
        if _queue(Judgment(
            judgment_id=_judgment_id(trace.trace_id, judge_cfg.model_id, rubric_version),
            output_id=trace.trace_id, judge_id=judge_cfg.model_id,
            rubric_version=rubric_version,
            scores=scores, judge_reasoning=resp.content,
        )):
            break

    return total_added + append_jsonl_idempotent(judgments_path, new, key="judgment_id")
