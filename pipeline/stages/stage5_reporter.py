"""Stage 5 — render markdown leaderboard from scores.jsonl."""

from __future__ import annotations
import datetime as dt
from collections import defaultdict
from pathlib import Path
from pipeline.schemas import CellScore
from pipeline.jsonl_io import read_jsonl


SIGNALS = ("username_replaced", "id_format_used", "governance_depth", "fingerprint_warning")


def _format_cell(c: CellScore, sig: str) -> str:
    m = c.metrics[sig]
    half = (m.ci95[1] - m.ci95[0]) / 2
    return f"{m.mean:.2f} ± {half:.2f}"


def render_markdown_report(*, artifacts_dir: Path, reports_dir: Path, run_id: str) -> Path:
    cells = list(read_jsonl(artifacts_dir / "scores.jsonl", CellScore))
    out_dir = reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leaderboard.md"

    lines: list[str] = []
    lines.append(f"# PII Governance Benchmark — `{run_id}`")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"Cells: {len(cells)}")
    lines.append("")

    # Group by (prompt_id, complexity, bucket) → rows = model
    by_signal: dict[str, list[CellScore]] = defaultdict(list)
    for c in cells:
        for sig in SIGNALS:
            if sig in c.metrics:
                by_signal[sig].append(c)

    # Top performer per signal
    lines.append("## Top performers")
    lines.append("")
    for sig in SIGNALS:
        if not by_signal[sig]:
            continue
        top = max(by_signal[sig], key=lambda c: c.metrics[sig].mean)
        lines.append(f"- **{sig}**: Top: `{top.model_id}` "
                     f"({top.prompt_or_scenario_id} / {top.bucket}) — {_format_cell(top, sig)}")
    lines.append("")

    # Per-signal table grouped by (prompt, bucket); rows = model
    for sig in SIGNALS:
        lines.append(f"## `{sig}`")
        lines.append("")
        # Pivot: rows = model_id, cols = (prompt_or_scenario_id|complexity|bucket)
        col_keys: list[tuple[str, str, str]] = sorted({
            (c.prompt_or_scenario_id, c.complexity, c.bucket) for c in cells
        })
        models = sorted({c.model_id for c in cells})
        header = "| Model | " + " | ".join(f"{p}/{b}" for (p, _co, b) in col_keys) + " |"
        sep = "|" + "|".join(["---"] * (1 + len(col_keys))) + "|"
        lines.append(header)
        lines.append(sep)
        cell_lookup = {(c.model_id, c.prompt_or_scenario_id, c.complexity, c.bucket): c for c in cells}
        for m in models:
            row = [f"`{m}`"]
            for p, co, b in col_keys:
                c = cell_lookup.get((m, p, co, b))
                row.append(_format_cell(c, sig) if c else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
