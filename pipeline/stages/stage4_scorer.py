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


def _discretize(score: float) -> int:
    """Map [0,1] continuous score to {0, 1, 2} bin via round(score * 2)."""
    clamped = max(0.0, min(1.0, score))
    return min(2, int((clamped + 0.25) * 2))


def fleiss_kappa(ratings_per_item: list[list[float]]) -> float:
    """Fleiss kappa over discretized ratings (3 categories: low/mid/high).

    `ratings_per_item[i][r]` = score in [0,1] from rater r on item i.
    Each item must have the same number of raters n >= 2. Returns NaN if
    fewer than 2 raters or no items.
    """
    if not ratings_per_item:
        return math.nan
    n = len(ratings_per_item[0])
    if n < 2:
        return math.nan
    K = 3  # number of categories (0,1,2)
    N = len(ratings_per_item)
    # Build N×K count matrix of rater→category.
    counts = [[0] * K for _ in range(N)]
    for i, row in enumerate(ratings_per_item):
        if len(row) != n:
            return math.nan
        for s in row:
            counts[i][_discretize(s)] += 1

    # P_i: agreement on item i.
    P_items = []
    for i in range(N):
        s = sum(c * c for c in counts[i])
        P_items.append((s - n) / (n * (n - 1)))
    P_bar = sum(P_items) / N

    # P_e: chance agreement.
    p_j = [sum(counts[i][j] for i in range(N)) / (N * n) for j in range(K)]
    P_e = sum(p * p for p in p_j)

    if P_e >= 1.0:
        # All raters always picked one category → agreement is by definition perfect/undefined.
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)
