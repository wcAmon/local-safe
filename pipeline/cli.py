"""Command-line entry for the local-safe pipeline.

Subcommands:
  build-samples         Stage 1
  run                   Stage 2 (single-shot, all under_test models)
  judge-rule            Stage 3a
  judge-llm             Stage 3b for a specific judge model_id
  score                 Stage 4
  report                Stage 5
"""

from __future__ import annotations
import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pipeline.config import load_models, load_prompts, resolve_base_url
from pipeline.serving.openai_compat import OpenAICompatAdapter
from pipeline.stages.stage1_dataset import build_samples
from pipeline.stages.stage2_runner import run_single_shot
from pipeline.stages.stage3a_rule_judge import run_rule_judge
from pipeline.stages.stage3b_llm_judge import run_llm_judge
from pipeline.stages.stage4_scorer import run_scorer
from pipeline.stages.stage5_reporter import render_markdown_report
from pipeline.schemas import Sample
from pipeline.jsonl_io import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = REPO_ROOT / "vault"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CONFIG = REPO_ROOT / "config"
DEFAULT_REPORTS = REPO_ROOT / "reports"


def _adapter_for(model_cfg) -> OpenAICompatAdapter:
    if model_cfg.backend != "openai_compat":
        raise NotImplementedError(f"Phase 1 only supports openai_compat, got {model_cfg.backend!r}")
    base_url = resolve_base_url(model_cfg.base_url_env)
    return OpenAICompatAdapter(
        model_id=model_cfg.model_id, api_model=model_cfg.api_model, base_url=base_url,
    )


def cmd_build_samples(args: argparse.Namespace) -> None:
    salt = os.environ.get("LOCAL_SAFE_VAULT_KEY", "phase1-default-salt")
    manifest = build_samples(
        reddit_path=Path(args.reddit),
        vault_dir=DEFAULT_VAULT,
        artifacts_dir=DEFAULT_ARTIFACTS,
        salt=salt,
    )
    print(f"Built {manifest.n_samples} samples; buckets={manifest.buckets}")


def cmd_run(args: argparse.Namespace) -> None:
    models_cfg = load_models(DEFAULT_CONFIG / "models.yaml")
    prompts = load_prompts(DEFAULT_CONFIG / "prompts.yaml")
    samples = list(read_jsonl(DEFAULT_VAULT / "samples_raw.jsonl", Sample))
    salt = os.environ.get("LOCAL_SAFE_VAULT_KEY", "phase1-default-salt")
    total_added = 0
    # Model-major loop (per spec §10.5: single-resident swap)
    for model_cfg in models_cfg.under_test:
        adapter = _adapter_for(model_cfg)
        n = run_single_shot(
            adapter=adapter, model_cfg=model_cfg, prompts=prompts,
            samples=samples, vault_dir=DEFAULT_VAULT, artifacts_dir=DEFAULT_ARTIFACTS, salt=salt,
        )
        print(f"[{model_cfg.model_id}] added {n} outputs")
        total_added += n
    print(f"Total new outputs: {total_added}")


def cmd_judge_rule(_: argparse.Namespace) -> None:
    n = run_rule_judge(vault_dir=DEFAULT_VAULT, artifacts_dir=DEFAULT_ARTIFACTS)
    print(f"Rule judge added {n} judgments")


def cmd_judge_llm(args: argparse.Namespace) -> None:
    models_cfg = load_models(DEFAULT_CONFIG / "models.yaml")
    judge_cfg = next((j for j in models_cfg.judges if j.model_id == args.judge), None)
    if judge_cfg is None:
        sys.exit(f"unknown judge model_id: {args.judge!r}")
    adapter = _adapter_for(judge_cfg)
    n = run_llm_judge(
        adapter=adapter, judge_cfg=judge_cfg,
        rubric_path=DEFAULT_CONFIG / "rubric.v1.yaml",
        vault_dir=DEFAULT_VAULT, artifacts_dir=DEFAULT_ARTIFACTS,
    )
    print(f"[{judge_cfg.model_id}] added {n} judgments")


def cmd_score(_: argparse.Namespace) -> None:
    n = run_scorer(artifacts_dir=DEFAULT_ARTIFACTS)
    print(f"Wrote {n} cell scores")


def cmd_report(args: argparse.Namespace) -> None:
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = render_markdown_report(
        artifacts_dir=DEFAULT_ARTIFACTS, reports_dir=DEFAULT_REPORTS, run_id=run_id,
    )
    print(f"Report: {p}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="local-safe", description="PII governance benchmark CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bs = sub.add_parser("build-samples", help="Stage 1: build samples from reddit jsonl")
    p_bs.add_argument("--reddit", required=True, help="path to reddit JSONL")
    p_bs.set_defaults(func=cmd_build_samples)

    p_run = sub.add_parser("run", help="Stage 2: run single-shot inference")
    p_run.set_defaults(func=cmd_run)

    p_jr = sub.add_parser("judge-rule", help="Stage 3a: rule-based judging")
    p_jr.set_defaults(func=cmd_judge_rule)

    p_jl = sub.add_parser("judge-llm", help="Stage 3b: LLM judge")
    p_jl.add_argument("--judge", required=True, help="judge model_id from models.yaml")
    p_jl.set_defaults(func=cmd_judge_llm)

    p_sc = sub.add_parser("score", help="Stage 4: aggregate scores")
    p_sc.set_defaults(func=cmd_score)

    p_rp = sub.add_parser("report", help="Stage 5: render markdown report")
    p_rp.add_argument("--run-id", default=None)
    p_rp.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
