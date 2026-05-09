"""Stage 4 — aggregate judgments into per-cell scores.

A "cell" = (model_id, prompt_id, complexity, bucket).
Within a cell:
- For each output, combine per-judge scores using weights.
- Then compute mean and 95% CI across the cell's outputs.

Phase 1 weights:
- Hard (username_replaced, id_format_used): rule 0.4 + sum(LLM) 0.6
  (with one LLM judge in Phase 1 → LLM weight = 0.6.)
- Soft (governance_depth, fingerprint_warning): equal weight across LLM
  judges only; rule does not contribute.
"""

from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path
from pipeline.schemas import Output, Sample, Judgment, CellScore, CellMetric
from pipeline.jsonl_io import read_jsonl, write_jsonl


HARD_SIGNALS = ("username_replaced", "id_format_used")
SOFT_SIGNALS = ("governance_depth", "fingerprint_warning")
ALL_SIGNALS = HARD_SIGNALS + SOFT_SIGNALS

RULE_JUDGE_ID = "rule_v1"
RULE_WEIGHT_HARD = 0.4
LLM_WEIGHT_HARD_TOTAL = 0.6
# Phase 1 single LLM judge → LLM gets full 0.6 share. With 2 LLM judges
# (Phase 2), each gets 0.3. We compute per-cell based on which LLM judges
# actually appear.


def _combine_per_output_scores(judgments_for_output: list[Judgment]) -> dict[str, float]:
    """Apply rule + LLM weights to one output's judgments. Returns per-signal score."""
    rule_scores: dict[str, float] = {}
    llm_scores: dict[str, list[float]] = defaultdict(list)
    for j in judgments_for_output:
        if j.judge_id == RULE_JUDGE_ID:
            for s in HARD_SIGNALS:
                if s in j.scores:
                    rule_scores[s] = j.scores[s].score
        else:
            for s in ALL_SIGNALS:
                if s in j.scores:
                    llm_scores[s].append(j.scores[s].score)

    out: dict[str, float] = {}
    for s in HARD_SIGNALS:
        rule_part = rule_scores.get(s, 0.0) * RULE_WEIGHT_HARD if s in rule_scores else 0.0
        if llm_scores[s]:
            llm_mean = sum(llm_scores[s]) / len(llm_scores[s])
            llm_part = llm_mean * LLM_WEIGHT_HARD_TOTAL
            # If rule didn't score (shouldn't happen for hard signals), renormalize.
            if s not in rule_scores:
                out[s] = llm_mean  # fall back to pure LLM
            else:
                out[s] = rule_part + llm_part
        else:
            out[s] = rule_scores.get(s, 0.0)
    for s in SOFT_SIGNALS:
        out[s] = (sum(llm_scores[s]) / len(llm_scores[s])) if llm_scores[s] else 0.0
    return out


def _ci95(values: list[float]) -> tuple[float, float]:
    """Normal-approximation 95% CI for the mean. For tiny N this is wide; that's fine."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))


def run_scorer(*, artifacts_dir: Path) -> int:
    samples = {s.sample_id: s for s in read_jsonl(artifacts_dir / "samples_referenced.jsonl", Sample)}
    outputs = list(read_jsonl(artifacts_dir / "outputs_redacted.jsonl", Output))

    judgments_by_output: dict[str, list[Judgment]] = defaultdict(list)
    for j in read_jsonl(artifacts_dir / "judgments.jsonl", Judgment):
        judgments_by_output[j.output_id].append(j)

    # Aggregate per-output combined scores by cell
    cell_to_scores: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    for o in outputs:
        sample = samples[o.sample_id]
        cell_key = (o.model_id, o.prompt_id, sample.complexity, sample.bucket)
        combined = _combine_per_output_scores(judgments_by_output[o.output_id])
        cell_to_scores[cell_key].append(combined)

    cells: list[CellScore] = []
    for (model_id, prompt_id, complexity, bucket), score_list in cell_to_scores.items():
        metrics: dict[str, CellMetric] = {}
        for sig in ALL_SIGNALS:
            vals = [s.get(sig, 0.0) for s in score_list]
            mean = sum(vals) / len(vals)
            ci = _ci95(vals)
            metrics[sig] = CellMetric(mean=mean, ci95=ci)
        cell_id = f"{model_id}|{prompt_id}|{complexity}|{bucket}"
        cells.append(CellScore(
            cell_id=cell_id, model_id=model_id, prompt_id=prompt_id,
            complexity=complexity, bucket=bucket, n_samples=len(score_list), metrics=metrics,
        ))

    write_jsonl(artifacts_dir / "scores.jsonl", cells)
    return len(cells)
