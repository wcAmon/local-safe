#!/usr/bin/env python3
"""Regenerate the Current Leaderboard tables in README from the latest report.

Reads ``reports/<run-id>/radar_data.json`` (latest by name, or ``--run-id``)
and rewrites the block between the ``<!-- LEADERBOARD:AUTO-START -->`` and
``<!-- LEADERBOARD:AUTO-END -->`` markers in ``README.md``. Leaves the
``### Analysis`` prose below the markers untouched.

Usage:
    uv run python scripts/update_leaderboard.py
    uv run python scripts/update_leaderboard.py --run-id 20260512-195227
    uv run python scripts/update_leaderboard.py --check  # CI: exit 1 if stale

When you add a new model, append a row to ``MODEL_META`` so the
"Family / shape" column has something more useful than the model_id.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_README = REPO / "README.md"
DEFAULT_REPORTS_DIR = REPO / "reports"
START_MARK = "<!-- LEADERBOARD:AUTO-START -->"
END_MARK = "<!-- LEADERBOARD:AUTO-END -->"

# Hand-maintained model descriptions for the "Family / shape" column.
# Add an entry when adding a new model. Missing entries render as "—".
MODEL_META: dict[str, str] = {
    "gpt-oss-safeguard-120b@v1": "gpt-oss safety-tuned, MoE 117B/5.1B",
    "gemma4-e4b-it@v1": "gemma, ~5B dense",
    "gemma4-26b-a4b-it@v1": "gemma, MoE 26B/4B",
    "gpt-oss-safeguard-20b@v1": "gpt-oss safety-tuned, 20B dense",
    "qwen3.5-9b@v1": "qwen, 9B dense (prev gen)",
    "qwen3.6-35b-a3b@v1": "qwen, MoE 35B/3B",
}

AXIS_DISPLAY = {
    "direct_privacy": "direct_privacy",
    "identity_substitution": "identity_subst",
    "fingerprint_safety": "fingerprint",
    "cloud_tool_safety": "cloud_tool",
    "task_utility": "task_utility",
    "reverse_resistance": "reverse_resist",
}
TRACK_ORDER = ["single_shot", "multi_shot", "agentic_workflow"]
TRACK_DISPLAY = {
    "single_shot": "single_shot (autonomy-sensitive)",
    "multi_shot": "multi_shot",
    "agentic_workflow": "agentic_workflow",
}


def latest_run(reports_dir: Path) -> str:
    """Most recently modified report directory containing radar_data.json.

    mtime, not name — older reports use legacy ``datatrace-*`` prefixes that
    sort after the new ``YYYYMMDD-HHMMSS`` format.
    """
    candidates = [
        p for p in reports_dir.iterdir()
        if p.is_dir() and (p / "radar_data.json").exists()
    ]
    if not candidates:
        sys.exit(f"no runs with radar_data.json found in {reports_dir}")
    return max(candidates, key=lambda p: (p / "radar_data.json").stat().st_mtime).name


def humanize_run_id(run_id: str) -> str:
    """Convert ``YYYYMMDD-HHMMSS`` → ``YYYY-MM-DD HH:MM UTC`` if it parses."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$", run_id)
    if not m:
        return run_id
    y, mo, d, hh, mm, _ = m.groups()
    return f"{y}-{mo}-{d} {hh}:{mm} UTC"


def fmt2(v: float) -> str:
    return f"{v:.2f}"


def fmt3(v: float) -> str:
    return f"{v:.3f}"


def bold_if_top(v: float, top: float, *, tied: bool, fmt) -> str:
    if abs(v - top) < 5e-4:  # match the rounding precision
        return f"**{'=' if tied else ''}{fmt(v)}**"
    return fmt(v)


def build_block(radar: dict, run_id: str) -> str:
    axes: list[str] = radar["axes"]
    models: list[str] = radar["models"]
    rows: list[dict] = radar["rows"]

    per_model_axis: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_model_track: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_model_all: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        m = r["model_id"]
        t = r["track"]
        missing = [a for a in axes if a not in r["axes"]]
        if missing:
            sys.exit(
                f"row {m}/{t} is missing axes {missing} — looks like an older "
                "report format. Re-run `make report` against the current "
                "rubric before regenerating the leaderboard."
            )
        for a in axes:
            v = r["axes"][a]
            per_model_axis[(m, a)].append(v)
            per_model_track[(m, t)].append(v)
            per_model_all[m].append(v)

    composite = {m: sum(per_model_all[m]) / len(per_model_all[m]) for m in models}
    # Rank by composite desc, then model_id asc for stable tie-breaking.
    ranked = sorted(models, key=lambda m: (-composite[m], m))

    lines: list[str] = []
    n_tracks = len(set(r["track"] for r in rows))
    lines.append("")
    lines.append(
        f"> **Snapshot**: auto-generated from "
        f"`reports/{run_id}/radar_data.json` "
        f"({humanize_run_id(run_id)}). "
        f"{len(models)} models × {len(axes)} axes × {n_tracks} tracks. "
        f"Regenerate with `make leaderboard` after `make report`. "
        f"All scores in [0,1]; higher is better."
    )
    lines.append("")

    # ── overall composite ──
    lines.append(f"### Overall composite (mean across {len(axes)} axes × {n_tracks} tracks)")
    lines.append("")
    lines.append("| Rank | Model | Family / shape | Composite |")
    lines.append("|---|---|---|---|")
    for i, m in enumerate(ranked, 1):
        meta = MODEL_META.get(m, "—")
        score = fmt3(composite[m])
        score_cell = f"**{score}**" if i == 1 else score
        lines.append(f"| #{i} | `{m.replace('@v1', '')}` | {meta} | {score_cell} |")
    lines.append("")

    # ── per-track composite ──
    lines.append(f"### Per-track composite (mean across {len(axes)} axes within each track)")
    lines.append("")
    lines.append("| Model | " + " | ".join(TRACK_DISPLAY[t] for t in TRACK_ORDER) + " |")
    lines.append("|---|" + "|".join("---" for _ in TRACK_ORDER) + "|")
    track_vals = {
        (m, t): sum(per_model_track[(m, t)]) / len(per_model_track[(m, t)])
        for m in models for t in TRACK_ORDER
    }
    track_top = {t: max(track_vals[(m, t)] for m in models) for t in TRACK_ORDER}
    track_top_count = {
        t: sum(1 for m in models if abs(track_vals[(m, t)] - track_top[t]) < 5e-4)
        for t in TRACK_ORDER
    }
    for m in ranked:
        cells = [
            bold_if_top(
                track_vals[(m, t)], track_top[t],
                tied=track_top_count[t] > 1, fmt=fmt2,
            )
            for t in TRACK_ORDER
        ]
        lines.append(f"| `{m.replace('@v1', '')}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ── per-axis composite ──
    lines.append(f"### Per-axis composite (mean across {n_tracks} tracks)")
    lines.append("")
    lines.append("Bold = top-of-column. Tie marks (=) indicate ties.")
    lines.append("")
    lines.append("| Model | " + " | ".join(AXIS_DISPLAY.get(a, a) for a in axes) + " |")
    lines.append("|---|" + "|".join("---" for _ in axes) + "|")
    axis_vals = {
        (m, a): sum(per_model_axis[(m, a)]) / len(per_model_axis[(m, a)])
        for m in models for a in axes
    }
    axis_top = {a: max(axis_vals[(m, a)] for m in models) for a in axes}
    axis_top_count = {
        a: sum(1 for m in models if abs(axis_vals[(m, a)] - axis_top[a]) < 5e-4)
        for a in axes
    }
    for m in ranked:
        cells = [
            bold_if_top(
                axis_vals[(m, a)], axis_top[a],
                tied=axis_top_count[a] > 1, fmt=fmt2,
            )
            for a in axes
        ]
        lines.append(f"| `{m.replace('@v1', '')}` | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def replace_block(readme_text: str, new_block: str) -> str:
    pattern = re.compile(
        re.escape(START_MARK) + r".*?" + re.escape(END_MARK),
        flags=re.DOTALL,
    )
    if not pattern.search(readme_text):
        sys.exit(
            f"could not find {START_MARK} ... {END_MARK} in README — add the "
            "markers around the auto-generated tables first."
        )
    replacement = f"{START_MARK}\n{new_block}\n{END_MARK}"
    return pattern.sub(replacement, readme_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="reports/<run-id>; defaults to latest by name")
    parser.add_argument("--readme", default=str(DEFAULT_README), help="path to README.md")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if README would change; useful in CI",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    run_id = args.run_id or latest_run(reports_dir)
    radar_path = reports_dir / run_id / "radar_data.json"
    if not radar_path.exists():
        sys.exit(f"radar_data.json missing at {radar_path}")
    radar = json.loads(radar_path.read_text())

    new_block = build_block(radar, run_id)
    readme_path = Path(args.readme)
    text = readme_path.read_text()
    updated = replace_block(text, new_block)

    if updated == text:
        print(f"README leaderboard already current (run {run_id})")
        return 0

    if args.check:
        print(f"README leaderboard stale vs run {run_id}", file=sys.stderr)
        return 1

    readme_path.write_text(updated)
    print(f"updated README leaderboard from run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
