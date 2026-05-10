"""Stage 5 — render markdown leaderboard from scores.jsonl."""

from __future__ import annotations
import datetime as dt
import json as _json
from collections import defaultdict
from pathlib import Path
from pipeline.schemas import CellScore
from pipeline.jsonl_io import read_jsonl


SIGNALS = (
    "pii_leak_free", "username_replaced", "id_format_used",
    "governance_depth", "fingerprint_leak_free", "fingerprint_warning",
    "task_utility", "cloud_search_clean", "reverse_leak_resisted",
    "third_party_exfiltration_resisted",
)

TRACK_LABELS = {
    "single_shot": "single_shot",
    "multi_turn": "multi_shot",
    "agent_loop": "agentic_workflow",
    "long_context": "long_context",
}

RADAR_AXES = {
    "direct_privacy": ("pii_leak_free",),
    "identity_substitution": ("replaced_AND_substituted", "username_replaced"),
    "fingerprint_safety": ("fingerprint_leak_free", "fingerprint_warning"),
    "cloud_tool_safety": (
        "third_party_exfiltration_resisted",
        "cloud_search_clean",
        "tool_input_clean",
    ),
    "task_utility": ("task_utility", "workflow_completed", "multi_step_consistency"),
    "reverse_resistance": ("reverse_leak_resisted", "prompt_injection_resisted"),
}


def _format_cell(c: CellScore, sig: str) -> str:
    m = c.metrics[sig]
    half = (m.ci95[1] - m.ci95[0]) / 2
    return f"{m.mean:.2f} ± {half:.2f}"


def _mean_for_axis(cells: list[CellScore], signals: tuple[str, ...]) -> float | None:
    values: list[tuple[float, int]] = []
    for c in cells:
        for sig in signals:
            if sig in c.metrics:
                values.append((c.metrics[sig].mean, c.n_samples))
                break
    if not values:
        return None
    total_n = sum(n for _, n in values)
    return sum(v * n for v, n in values) / total_n


def _write_radar_data(out_dir: Path, cells: list[CellScore]) -> None:
    by_track_model: dict[tuple[str, str], list[CellScore]] = defaultdict(list)
    for c in cells:
        by_track_model[(TRACK_LABELS.get(c.session_kind, c.session_kind), c.model_id)].append(c)

    rows = []
    for (track, model_id), model_cells in sorted(by_track_model.items()):
        axes = {}
        for axis, signals in RADAR_AXES.items():
            value = _mean_for_axis(model_cells, signals)
            if value is not None:
                axes[axis] = round(value, 4)
        rows.append({
            "track": track,
            "model_id": model_id,
            "axes": axes,
        })

    payload = {
        "schema": "datatrace-radar-v1",
        "axes": list(RADAR_AXES.keys()),
        "tracks": sorted({r["track"] for r in rows}),
        "models": sorted({r["model_id"] for r in rows}),
        "rows": rows,
    }
    (out_dir / "radar_data.json").write_text(
        _json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_markdown_report(*, artifacts_dir: Path, reports_dir: Path, run_id: str) -> Path:
    cells = list(read_jsonl(artifacts_dir / "scores.jsonl", CellScore))
    out_dir = reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leaderboard.md"
    _write_radar_data(out_dir, cells)

    lines: list[str] = []
    lines.append(f"# DataTrace Privacy Benchmark — `{run_id}`")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"Cells: {len(cells)}")
    lines.append("Tracks: `single_shot`, `multi_shot`, `agentic_workflow`")
    lines.append(f"Radar data: `radar_data.json`")
    lines.append("")

    # Group cells by session_kind
    cells_by_kind: dict[str, list[CellScore]] = defaultdict(list)
    for c in cells:
        cells_by_kind[c.session_kind].append(c)

    # Top performers (per signal, across all cells)
    by_signal: dict[str, list[CellScore]] = defaultdict(list)
    for c in cells:
        for sig in tuple(SIGNALS) + (
            "multi_step_consistency", "id_consistency",
            "workflow_completed", "replaced_AND_substituted",
            "privacy_utility_balance",
        ):
            if sig in c.metrics:
                by_signal[sig].append(c)

    lines.append("## Top performers")
    lines.append("")
    for sig in list(SIGNALS) + [
        "multi_step_consistency", "id_consistency",
        "workflow_completed", "replaced_AND_substituted",
        "privacy_utility_balance",
    ]:
        if not by_signal[sig]:
            continue
        top = max(by_signal[sig], key=lambda c: c.metrics[sig].mean)
        lines.append(f"- **{sig}**: Top: `{top.model_id}` "
                     f"({top.prompt_or_scenario_id} / {top.bucket} / {top.session_kind}) — {_format_cell(top, sig)}")
    lines.append("")

    # Per-session_kind leaderboards
    for kind in ("single_shot", "multi_turn", "agent_loop", "long_context"):
        if kind not in cells_by_kind:
            continue
        kind_label = {
            "single_shot": "Single-shot",
            "multi_turn": "Multi-shot",
            "agent_loop": "Agentic workflow",
            "long_context": "Long-context",
        }[kind]
        lines.append(f"# {kind_label} Leaderboard")
        lines.append("")
        kind_cells = cells_by_kind[kind]
        kind_signals = [sig for sig in (
            list(SIGNALS) + [
                "multi_step_consistency", "id_consistency",
                "workflow_completed", "replaced_AND_substituted",
                "privacy_utility_balance",
            ]
        )
                          if any(sig in c.metrics for c in kind_cells)]
        for sig in kind_signals:
            lines.append(f"## `{sig}`")
            lines.append("")
            col_keys = sorted({(c.prompt_or_scenario_id, c.complexity, c.bucket) for c in kind_cells})
            models = sorted({c.model_id for c in kind_cells})
            header = "| Model | " + " | ".join(f"{p}/{co}/{b}" for (p, co, b) in col_keys) + " |"
            sep = "|" + "|".join(["---"] * (1 + len(col_keys))) + "|"
            lines.append(header); lines.append(sep)
            lookup = {(c.model_id, c.prompt_or_scenario_id, c.complexity, c.bucket): c for c in kind_cells}
            for m in models:
                row = [f"`{m}`"]
                for p, co, b in col_keys:
                    c = lookup.get((m, p, co, b))
                    if c is None or sig not in c.metrics:
                        row.append("—")
                    else:
                        cell_str = _format_cell(c, sig)
                        if c.judge_agreement and c.judge_agreement.status == "unreliable":
                            cell_str += " ⚠️"
                        # Phase 3: append [tested:...] for cells with tested_dimensions
                        if c.tested_dimensions:
                            cell_str += f" [tested: {', '.join(c.tested_dimensions)}]"
                        row.append(cell_str)
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Cost summary
    cost_log = artifacts_dir / "cost.jsonl"
    if cost_log.exists():
        totals: dict[str, float] = defaultdict(float)
        with cost_log.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                totals[row["judge_id"]] += row["cost_usd"]
        lines.append("# Cost Summary")
        lines.append("")
        lines.append("| Judge | Total cost (USD) |")
        lines.append("|---|---|")
        grand = 0.0
        for jid, total in sorted(totals.items()):
            lines.append(f"| `{jid}` | ${total:.2f} |")
            grand += total
        lines.append(f"| **Total** | **${grand:.2f}** |")
        lines.append("")

    # Footnote on agreement
    if any((c.judge_agreement and c.judge_agreement.status == "unreliable") for c in cells):
        lines.append("---")
        lines.append("⚠️ = Fleiss kappa < 0.4 across LLM judges. Treat as preliminary.")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
