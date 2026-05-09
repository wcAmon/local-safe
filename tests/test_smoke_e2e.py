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

    env = os.environ.copy()
    env["LOCAL_SAFE_VAULT_KEY"] = "smoke-test"
    env["OLLAMA_HUB_BASE_URL"] = base_url

    subprocess.check_call(["make", "clean-artifacts"], cwd=REPO_ROOT, env=env)
    # Phase 2: build with --multi-thread
    subprocess.check_call(
        ["make", "samples-multi", "REDDIT=tests/fixtures/tiny_reddit_v2.jsonl"],
        cwd=REPO_ROOT, env=env,
    )
    subprocess.check_call(["make", "run"], cwd=REPO_ROOT, env=env, timeout=1800)
    subprocess.check_call(["make", "run-multi-turn"], cwd=REPO_ROOT, env=env, timeout=1800)
    subprocess.check_call(["make", "judge-rule"], cwd=REPO_ROOT, env=env)
    subprocess.check_call(["make", "judge-llm-all"], cwd=REPO_ROOT, env=env, timeout=1800)
    subprocess.check_call(["make", "score"], cwd=REPO_ROOT, env=env)
    subprocess.check_call(["make", "report"], cwd=REPO_ROOT, env=env)

    artifacts = REPO_ROOT / "artifacts"
    assert (REPO_ROOT / "vault" / "samples_raw.jsonl").stat().st_size > 0
    assert (REPO_ROOT / "vault" / "traces_raw.jsonl").stat().st_size > 0
    assert (artifacts / "outputs_redacted.jsonl").stat().st_size > 0
    assert (artifacts / "traces.jsonl").stat().st_size > 0
    assert (artifacts / "judgments.jsonl").stat().st_size > 0
    assert (artifacts / "scores.jsonl").stat().st_size > 0
    assert (artifacts / "cost.jsonl").exists()  # may be empty if no anthropic call
    assert any((REPO_ROOT / "reports").glob("*/leaderboard.md"))
