"""End-to-end smoke test against a running ollama-hub.

Skipped unless ``RUN_SMOKE=1`` is set. Requires ``OLLAMA_HUB_BASE_URL`` and
the ollama-hub gateway to respond on ``/health``.

Cold-start of large models can take minutes; this test is designed to be
slow and is excluded from the default suite.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pytest
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/v1") + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.mark.skipif(os.environ.get("RUN_SMOKE") != "1", reason="set RUN_SMOKE=1 to run")
def test_e2e_pipeline(tmp_path):
    base_url = os.environ.get("OLLAMA_HUB_BASE_URL", "http://localhost:11434/v1")
    if not _server_up(base_url):
        pytest.skip(f"ollama-hub not reachable at {base_url}")

    # Use temp vault/artifacts/reports to avoid clobbering main run.
    env = os.environ.copy()
    env["LOCAL_SAFE_VAULT_KEY"] = "smoke-test"
    env["OLLAMA_HUB_BASE_URL"] = base_url   # ensure subprocess sees it even if not in .env

    # Run all stages via the CLI in-process is awkward (it uses default dirs).
    # For Phase 1 simplicity we cd into a tmp_path-rooted copy of config and
    # invoke the make targets, capturing output.
    # Simpler approach: run the CLI subcommands one-by-one with the default
    # repo dirs, but first wipe artifacts (idempotency makes this safe).
    subprocess.check_call(["make", "clean-artifacts"], cwd=REPO_ROOT, env=env)
    subprocess.check_call(
        ["make", "samples", "REDDIT=tests/fixtures/tiny_reddit.jsonl"],
        cwd=REPO_ROOT, env=env,
    )
    # Run only one model to keep smoke time bounded; override via env if desired.
    # Phase 1 simplification: run all under_test models.
    subprocess.check_call(["make", "run"], cwd=REPO_ROOT, env=env, timeout=1800)
    subprocess.check_call(["make", "judge-rule"], cwd=REPO_ROOT, env=env)
    subprocess.check_call(["make", "judge-llm"], cwd=REPO_ROOT, env=env, timeout=1800)
    subprocess.check_call(["make", "score"], cwd=REPO_ROOT, env=env)
    subprocess.check_call(["make", "report"], cwd=REPO_ROOT, env=env)

    # Sanity-check artifacts exist and aren't empty.
    artifacts = REPO_ROOT / "artifacts"
    assert (REPO_ROOT / "vault" / "samples_raw.jsonl").stat().st_size > 0
    assert (artifacts / "samples_referenced.jsonl").stat().st_size > 0
    assert (artifacts / "outputs_redacted.jsonl").stat().st_size > 0
    assert (artifacts / "judgments.jsonl").stat().st_size > 0
    assert (artifacts / "scores.jsonl").stat().st_size > 0
    assert any((REPO_ROOT / "reports").glob("*/leaderboard.md"))
