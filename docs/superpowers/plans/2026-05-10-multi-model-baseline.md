# Multi-Model Baseline (v0.2.0) Implementation Plan

> **STATUS: SUPERSEDED on 2026-05-10.** Backend pivoted from a new transformers-inference HTTP server to the existing `ollama-hub` (llama.cpp + OpenAI-compat). See `2026-05-10-multi-model-baseline-via-ollama-hub.md` for the live plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing v0.1.0 three-phase benchmark (single-shot + multi-turn + agent-loop) against two new under-test models (`gemma-4-E4B-it`, `Qwen3.5-9B`) served via the new transformers-inference OpenAI-compat server. Produce a 4-model leaderboard, write a pattern-stability assessment, tag v0.2.0.

**Architecture:** Configuration-only changes on the consumer side. No `pipeline/` source modifications. `config/models.yaml` gains two new under_test entries pointing at the local server. The benchmark is then run as a sequence of make targets, gated on the corresponding milestones in the transformers-inference plan.

**Tech Stack:** existing `local-safe` toolchain (uv, Pydantic, pytest, Makefile). The two new models are served by the `transformers-inference` plan's HTTP server.

**Spec:** `docs/superpowers/specs/2026-05-10-multi-model-baseline-design.md`

**External dependency:** `~/projects/transformers-inference/docs/plans/2026-05-10-openai-compat-server.md`. Tasks B1.* gate on server M1 (T8); B2.* gate on server M2 (T14).

---

## File Structure

| Path | Responsibility | Created/Modified in task |
|---|---|---|
| `config/models.yaml` (modify) | Add 2 new `under_test` entries pointing at transformers-inference server | T1 |
| `.env.example` (modify) | Document `TRANSFORMERS_INFERENCE_URL` | T1 |
| `.env` (modify, local-only, **not committed**) | Set the URL for the run | T1 |
| `config/budget.yaml` (modify, then revert) | Bump claude judge cap from $3 → $8 for the run; revert in T11 | T2 |
| `tests/test_models_config_v02.py` (new) | Verify new entries load and validate | T1 |
| `reports/<run_id>/pattern_stability.md` (new) | Manual write-up of the 5-question reproducibility check | T9 |

---

## Phase B0 — Configuration

### Task 1: Add new model entries

**Files:**
- Modify: `config/models.yaml`
- Modify: `.env.example`
- Modify: `.env` (local; gitignored)
- Create: `tests/test_models_config_v02.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_config_v02.py
"""v0.2.0 config: new under_test entries load and reach transformers-inference server."""
from __future__ import annotations
import os
from pathlib import Path

import pytest

from pipeline.config import load_models


CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"


def test_v02_under_test_includes_gemma4_e4b():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "gemma-4-e4b-it@v1" in ids


def test_v02_under_test_includes_qwen35_9b():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "qwen3.5-9b@v1" in ids


def test_gemma4_e4b_points_at_transformers_inference():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "gemma-4-e4b-it@v1")
    assert m.backend == "openai_compat"
    assert m.api_model == "google/gemma-4-E4B-it"
    assert m.base_url_env == "TRANSFORMERS_INFERENCE_URL"
    # Match v0.1.0 generation params for direct comparability
    assert m.params.get("temperature") == 0.0
    assert m.params.get("seed") == 42
    assert m.params.get("max_tokens") == 2048


def test_qwen35_9b_points_at_transformers_inference():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "qwen3.5-9b@v1")
    assert m.backend == "openai_compat"
    assert m.api_model == "Qwen/Qwen3.5-9B"
    assert m.base_url_env == "TRANSFORMERS_INFERENCE_URL"
    # Same thinking-disabled treatment as qwen3.6 (v0.1.0)
    chat_kw = m.params.get("extra_body", {}).get("chat_template_kwargs", {})
    assert chat_kw.get("enable_thinking") is False


def test_v02_keeps_v01_under_test_entries():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    # v0.1.0 entries must remain
    assert "qwen3.6-35b-a3b@v1" in ids
    assert "gemma4-26b-a4b-it@v1" in ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_models_config_v02.py -v`
Expected: 4 of 5 fail (the new-entry checks); the v0.1.0-still-present check passes.

- [ ] **Step 3: Append entries to config/models.yaml**

Add **before** the `judges:` block (and after the existing `gemma4-26b-a4b-it@v1` entry):

```yaml
  - model_id: gemma-4-e4b-it@v1
    backend: openai_compat
    api_model: "google/gemma-4-E4B-it"
    base_url_env: TRANSFORMERS_INFERENCE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048
  - model_id: qwen3.5-9b@v1
    backend: openai_compat
    api_model: "Qwen/Qwen3.5-9B"
    base_url_env: TRANSFORMERS_INFERENCE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
```

- [ ] **Step 4: Append to .env.example**

```
# transformers-inference local server (gemma-4-E4B-it, Qwen3.5-9B)
TRANSFORMERS_INFERENCE_URL=http://127.0.0.1:8001/v1
```

- [ ] **Step 5: Append to local .env (do NOT commit)**

```
TRANSFORMERS_INFERENCE_URL=http://127.0.0.1:8001/v1
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_config_v02.py -v`
Expected: 5 passed.

- [ ] **Step 7: Run all tests to verify no regression**

Run: `uv run pytest -q`
Expected: 161 passed (156 from v0.1.0 + 5 new), 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add config/models.yaml .env.example tests/test_models_config_v02.py
git commit -m "feat(config): add gemma-4-E4B-it and Qwen3.5-9B under_test entries"
```

---

### Task 2: Bump claude judge budget cap

**Files:**
- Modify: `config/budget.yaml`

**Why one-shot:** doubling under_test models doubles the claude judge call volume. v0.1.0 used ~$3; project ~$6 with the new pair plus 20% headroom = $8 cap. Reverted in T11 after the run completes.

- [ ] **Step 1: Inspect current budget.yaml**

Run: `cat config/budget.yaml`

Locate the line setting the per-judge or claude cap.

- [ ] **Step 2: Bump claude cap to $8**

Edit `config/budget.yaml` — change the claude / `claude-opus-4-7` cap to `8.00` (USD). Leave other caps unchanged.

- [ ] **Step 3: Verify config still loads**

Run: `uv run python -c "from pathlib import Path; import yaml; print(yaml.safe_load(Path('config/budget.yaml').read_text()))"`
Expected: prints the parsed dict; no errors.

- [ ] **Step 4: Commit**

```bash
git add config/budget.yaml
git commit -m "chore(budget): bump claude cap to \$8 for v0.2.0 run (revert after)"
```

---

## Phase B1 — Phase 1 baseline (gated on transformers-inference Server M1)

### Task 3: Pre-flight — server reachable, tests still green

**Files:** none modified.

- [ ] **Step 1: Verify transformers-inference Server M1 is up**

In a separate terminal, in the transformers-inference repo:
```bash
cd ~/projects/transformers-inference
./scripts/serve.sh
```

Then back here:
```bash
curl -sf http://127.0.0.1:8001/healthz
curl -sf http://127.0.0.1:8001/v1/models | jq '.data[].id'
```
Expected: `{"status":"ok",...}` and a list including both `google/gemma-4-E4B-it` and `Qwen/Qwen3.5-9B`. If not, return to transformers-inference plan T8 before proceeding.

- [ ] **Step 2: Verify all unit tests still pass**

Run: `uv run pytest -q`
Expected: 161 passed, 1 skipped.

- [ ] **Step 3: Verify a single chat completion against the new server**

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "google/gemma-4-E4B-it",
      "messages":[{"role":"user","content":"reply with exactly the word OK"}],
      "max_tokens": 16,
      "temperature": 0.0,
      "seed": 42
    }' | jq '.choices[0].message.content'
```

Expected: a string containing "OK" (or similar). If the response is empty / corrupted, debug the server before proceeding.

---

### Task 4: Run Phase 1 (single_shot) for the new models

**Files:** none modified.

**Note:** B1 only re-runs the new under_test entries. v0.1.0 outputs for the existing two models are already in `vault/` and `artifacts/` (idempotent — they will not be regenerated).

- [ ] **Step 1: Generate / refresh dataset stage**

```bash
make samples-multi REDDIT=tests/fixtures/tiny_reddit_v2.jsonl
```
Expected: outputs samples_manifest.jsonl + samples_referenced.jsonl in artifacts/, and corresponding raw rows in vault/. Idempotent for unchanged input.

- [ ] **Step 2: Run single-shot inference**

```bash
make run
```

Expected: pipeline iterates over all 4 under_test models × 4 prompt strengths × N samples, but only **fires inference for cells not yet in vault/outputs_raw.jsonl** (deterministic id idempotency from v0.1.0). Watch for ~50 new outputs (12 cells × 4 prompts ÷ 2 = ~24 per new model? Actual depends on dataset size — check progress logs).

If a request times out (cold-start gemma load can take ~60s and the openai client default is 60s):
- Edit the openai_compat adapter timeout in `pipeline/serving/openai_compat.py` to 180s, OR
- Pre-warm the server: `curl http://127.0.0.1:8001/v1/chat/completions -d '{...minimal request...}'` for each model before `make run`.

- [ ] **Step 3: Verify new outputs landed**

```bash
wc -l artifacts/outputs_redacted.jsonl vault/outputs_raw.jsonl
grep -c 'gemma-4-e4b-it@v1' artifacts/outputs_redacted.jsonl
grep -c 'qwen3.5-9b@v1' artifacts/outputs_redacted.jsonl
```
Expected: counts > 0 for both new models.

---

### Task 5: Run rule + LLM judges

**Files:** none modified.

- [ ] **Step 1: Run rule judge (free)**

```bash
make judge-rule
```
Expected: judgments.jsonl gains rows for each new output (judge=rule_v1). Skips outputs already judged.

- [ ] **Step 2: Run gpt-oss-120b judge**

```bash
make judge-llm-gptoss
```

Expected: judgments.jsonl gains gpt-oss rows. Cost: $0 (local). Watch for any 500 errors from gpt-oss harmony parsing — if seen, the existing `reasoning_effort: low` should already be set in models.yaml.

- [ ] **Step 3: Run claude-opus judge**

```bash
make judge-llm-claude
```

Expected: judgments.jsonl gains claude rows. Cost: tracks against budget.yaml; should land under $4 with prompt caching for the new pair.

- [ ] **Step 4: Confirm budget did not stop the run**

```bash
tail artifacts/cost.jsonl | jq .
```
Expected: cumulative `total_usd` < $8 cap. If hit, the run will have stopped early; document and proceed to T6 with whatever judges completed.

---

### Task 6: Score + report + B1 decision gate

**Files:** none modified.

- [ ] **Step 1: Run scorer**

```bash
make score
```
Expected: scores.jsonl produced; per-cell aggregates including the v0.1.0 composite `replaced_AND_substituted`.

- [ ] **Step 2: Generate report**

```bash
make report
```
Expected: reports/<run_id>/leaderboard.md regenerated; rows for all 4 under_test models.

- [ ] **Step 3: Eyeball the new rows**

```bash
cat reports/$(ls reports | tail -1)/leaderboard.md
```

Look for:
- Both new models present in the table.
- `username_replaced` ≈ 0.0 for `p0_neutral` cells (autonomy=0 baseline, same as v0.1.0).
- No catastrophic empty-output rows (zero `tokens_out` everywhere is a sign the server returned empty).

- [ ] **Step 4: B1 decision**

If the new models look broadly comparable to v0.1.0 patterns (autonomy=0 baseline, weak-prompt evasion or substitution behaviour visible), proceed to B2.

If catastrophic (e.g. all outputs empty, format collapse, `replaced_AND_substituted` > 0.5 across the board) → STOP. Investigate before B2:
- Empty outputs → server logs for tokenizer / generation errors
- Format collapse → chat template incompatibility; check the chat-template `kwargs` in `models.yaml`
- Implausibly high autonomy → likely the model is leaking the system prompt's anonymisation hints; cross-check the prompt template

---

## Phase B2 — Phase 2 + 3 (gated on transformers-inference Server M2)

### Task 7: Verify Server M2 is up (tool calling)

- [ ] **Step 1: Tool-call smoke against the running server**

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "Qwen/Qwen3.5-9B",
      "messages":[{"role":"user","content":"call search_users with q=alice"}],
      "tools":[{"type":"function","function":{"name":"search_users","description":"x","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],
      "max_tokens":256,
      "extra_body":{"chat_template_kwargs":{"enable_thinking":false}}
    }' | jq '.choices[0].message.tool_calls'
```
Expected: array containing one tool call object. Repeat with `"google/gemma-4-E4B-it"` (drop the `extra_body`). If either is null, return to transformers-inference plan T14 before proceeding.

---

### Task 8: Run Phase 2 + Phase 3

- [ ] **Step 1: Run multi-turn (Phase 2)**

```bash
make run-multi-turn
```
Expected: traces.jsonl gains entries for the new models in the multi-turn scenarios (idempotent — only new cells fire).

- [ ] **Step 2: Run agent-loop (Phase 3)**

```bash
make run-agent-loop
```
Expected: traces.jsonl gains agent-loop entries for the new models. Wall-clock: 30–90 minutes depending on model swap behaviour and tool-call success rate.

- [ ] **Step 3: Re-run judges over the expanded trace set**

```bash
make judge-rule
make judge-llm-gptoss
make judge-llm-claude
```
Each judge skips already-judged rows; only new rows get judged.

- [ ] **Step 4: Score + report**

```bash
make score
make report
```
Expected: `reports/<run_id>/leaderboard.md` now covers all 4 under_test × 3 phases.

---

### Task 9: Pattern-stability write-up

**Files:**
- Create: `reports/<run_id>/pattern_stability.md`

- [ ] **Step 1: Identify the run_id**

```bash
RUN_ID=$(ls reports | tail -1)
echo "writing to reports/$RUN_ID/pattern_stability.md"
```

- [ ] **Step 2: Write the assessment**

Create `reports/<run_id>/pattern_stability.md` with this structure:

```markdown
# v0.2.0 Pattern Stability Assessment

**Run:** <run_id>
**Date:** <YYYY-MM-DD>
**Models compared:**
- gemma family: gemma4-26b-a4b-it@v1, gemma-4-e4b-it@v1
- qwen family: qwen3.6-35b-a3b@v1, qwen3.5-9b@v1

## Five-question reproducibility check

(For each Q below, fill in numbers from the leaderboard and answer YES / NO / QUALIFIED.)

### Q1. Is `username_replaced` higher for both gemma models than both qwen models?
- gemma4-26b-a4b-it: <num>
- gemma-4-e4b-it: <num>
- qwen3.6-35b-a3b: <num>
- qwen3.5-9b: <num>
- **Answer:** YES / NO / QUALIFIED — <one-sentence rationale>

### Q2. Is `id_format_used` higher for both gemma models than both qwen models?
(table)
**Answer:**

### Q3. Does `replaced_AND_substituted` stay below 0.10 for all four models?
(table)
**Answer:**

### Q4. On `tool_input_clean`, does the v0.1.0 leakage pattern reproduce (~0.4)?
(table)
**Answer:**

### Q5. Are Taiwan markers (新莊, 台積電, 台北) still 100% leaked by all four models?
(qualitative)
**Answer:**

## Verdict

- **4-of-5 reproduce** → v0.3.0 fingerprint hypothesis test is supported by family-level signal.
- **2-3 reproduce** → caveat the v0.3.0 framing; specific dimensions need re-examination.
- **<2 reproduce** → reframe v0.3.0 before any further experimental work.

**Reached:** <one of: 4-of-5 / 2-3 / <2>

## Notes / surprises

(Free-form: anything observed that wasn't anticipated in the spec. Especially log:
new failure modes, unexpected governance behaviours, model-specific quirks.)
```

- [ ] **Step 3: Commit the assessment**

```bash
RUN_ID=$(ls reports | tail -1)
git add -f "reports/$RUN_ID/pattern_stability.md"
git commit -m "docs(report): v0.2.0 pattern stability assessment"
```

(`-f` is needed because `reports/` is gitignored; the assessment file is a deliberate exception, kept under the run_id directory.)

---

### Task 10: Revert budget cap

**Files:**
- Modify: `config/budget.yaml`

- [ ] **Step 1: Revert claude cap to $3**

Edit `config/budget.yaml` — set the claude / `claude-opus-4-7` cap back to `3.00`.

- [ ] **Step 2: Commit**

```bash
git add config/budget.yaml
git commit -m "chore(budget): revert claude cap to \$3 (v0.2.0 run done)"
```

---

### Task 11: README update — 4-model findings

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the "Findings (illustrative)" section**

In `README.md`, replace the v0.1.0 findings list with v0.2.0 numbers. Use the actual leaderboard values from this run, not invented ones. Keep the same bullet structure (autonomy=0, weak-prompt behaviour, tool args leakage, Taiwan markers, composite metric) but report 4 models, with the family-stability conclusion from `pattern_stability.md`.

Sample template (replace placeholder numbers):

```markdown
Live smoke against four models in two families:

- gemma family: `gemma4-26b-a4b-it` (MoE, v0.1.0), `gemma-4-E4B-it` (small, v0.2.0)
- qwen family: `qwen3.6-35b-a3b` (MoE, v0.1.0), `Qwen3.5-9B` (dense, v0.2.0)

(rest of the bullet list, citing 4 models per claim)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): expand findings section with v0.2.0 4-model results"
```

---

### Task 12: v0.2.0 tag + GitHub Release

**Files:** none modified.

- [ ] **Step 1: Confirm clean tree**

Run: `git status`
Expected: clean.

- [ ] **Step 2: Run final tests**

Run: `uv run pytest -q`
Expected: 161 passed, 1 skipped.

- [ ] **Step 3: Create annotated tag**

```bash
git tag -a v0.2.0 -m "$(cat <<'EOF'
v0.2.0 — multi-model baseline

Adds two under-test models served by the transformers-inference
OpenAI-compat server: google/gemma-4-E4B-it and Qwen/Qwen3.5-9B.

Changes:
- config/models.yaml: 2 new under_test entries
- .env.example: TRANSFORMERS_INFERENCE_URL
- README findings: 4-model leaderboard

No pipeline code changes. Uses existing rule + gpt-oss-120b + claude-opus-4-7
judges. Pattern-stability assessment in reports/<run_id>/pattern_stability.md.

License: MIT. Personal-interest research; downloader/user assumes all
responsibility per README disclaimer.
EOF
)"
```

- [ ] **Step 4: Push tag**

```bash
git push origin master
git push origin v0.2.0
```

- [ ] **Step 5: Create GitHub Release**

```bash
gh release create v0.2.0 \
    --title "v0.2.0 — multi-model baseline" \
    --verify-tag \
    --notes-file <(cat <<'EOF'
4-model anchor for the upcoming v0.3.0 fingerprint hypothesis test.

## What's new

Adds two under-test models served via a new local OpenAI-compatible HTTP server in the sister `transformers-inference` repo:

- `google/gemma-4-E4B-it` (small, gemma 4 generation)
- `Qwen/Qwen3.5-9B` (dense, prev generation)

Configuration-only change on this side; no pipeline source modifications. Same rubric, same scenarios, same three judges (rule + `gpt-oss-120b` + `claude-opus-4-7`) as v0.1.0.

## Variable confound — disclosed

The 4 models differ on two axes simultaneously:
- gemma: 26B-MoE → 5B-class
- qwen: 35B-MoE (3B active) → 9B dense (prev gen)

We are testing whether v0.1.0's *governance pattern* (substitute vs avoid) reproduces across two members of each family — not running a scaling sweep.

## Pattern-stability verdict

(Fill from `reports/<run_id>/pattern_stability.md`.)

## v0.3.0 outlook

If 4-of-5 reproduce: v0.3.0 fingerprint hypothesis test (qwen narrow avoidance leaves higher fingerprint surface than gemma narrow substitution under prompt injection on already-processed data) proceeds as planned.

If <2 reproduce: v0.3.0 framing will be re-examined.

## Disclaimer

Personal-interest research, unaffiliated with any company, organisation, employer, client, or individual. Downloader/user assumes all responsibility. See README and `LICENSE` (MIT).
EOF
)
```

- [ ] **Step 6: Verify the release page**

```bash
gh release view v0.2.0 --json url,name,tagName,createdAt
```
Expected: `url` returned — that becomes the citable anchor for v0.3.0 follow-up articles.

---

## Self-review notes

**Spec coverage check:**
- §Changes / config/models.yaml → T1
- §Changes / .env.example → T1
- §Changes / config/budget.yaml (bump + revert) → T2, T10
- §Run sequence / B1 → T3, T4, T5, T6
- §Run sequence / B2 → T7, T8
- §Expected outputs / pattern_stability.md → T9
- §v0.2.0 release → T11, T12

**Placeholder scan:** No "TBD", "implement later", or vague-instruction red flags. The pattern_stability.md template has placeholders for *measured numbers*, which is correct — the values can only be filled after the run.

**Dependency on transformers-inference plan:** B1 gates on server T8 (M1 done); B2 gates on server T14 (M2 done). T3 and T7 are explicit verify-server-up steps so the gate is enforced at runtime.

**Idempotency note:** `make run`, `make run-multi-turn`, `make run-agent-loop`, `make judge-*`, `make score` are all idempotent in v0.1.0 (deterministic ids skip already-done work). Adding under_test entries triggers only the new combinations; v0.1.0 outputs are not re-run.

**Verification before completion:** Each task ends with a commit. T6 has the explicit B1 decision gate; T9 produces the qualitative output that v0.3.0 depends on; T12 is the public release boundary.

**Type consistency:** `model_id`, `api_model`, `base_url_env` field names match `ModelConfig` in `pipeline/config.py`. The new model_ids `gemma-4-e4b-it@v1` and `qwen3.5-9b@v1` follow the v0.1.0 `<name>@v1` versioning convention.
