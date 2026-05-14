# Multi-Model Baseline (v0.2.0) — via ollama-hub — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing v0.1.0 three-phase benchmark (single-shot + multi-turn + agent-loop) against two new under-test models (`gemma-4-e4b-it`, `qwen3.5-9b`) served by the existing `ollama-hub` gateway. Produce a 4-model leaderboard, write a pattern-stability assessment, tag v0.2.0.

**Architecture:** Configuration-only changes. Two new `under_test` entries in `config/models.yaml` reuse the same `OLLAMA_HUB_BASE_URL` already wired for v0.1.0. No `pipeline/` source modifications. The benchmark runs as the same Makefile sequence v0.1.0 used; idempotent stages skip already-done work.

**Tech Stack:** existing `local-safe` toolchain (uv, Pydantic, pytest, Makefile). Models served by `ollama-hub` (llama.cpp + OpenAI-compat gateway).

**Spec:** `docs/superpowers/specs/2026-05-10-multi-model-baseline-design.md`

**Phase 3 risk acknowledged:** Tool-calling on llama.cpp depends on `--jinja` and per-model chat-template completeness. Plan attempts Phase 3 and documents the result; partial data is acceptable.

---

## File Structure

| Path | Responsibility | Created/Modified in task |
|---|---|---|
| `config/models.yaml` (modify) | Add 2 new `under_test` entries pointing at ollama-hub | T2 |
| `config/budget.yaml` (modify, then revert) | Bump claude judge cap from $3 → $8 for the run; revert in T11 | T3, T11 |
| `tests/test_models_config_v02.py` (new) | Verify new entries load and validate | T2 |
| `reports/<run_id>/pattern_stability.md` (new) | Manual write-up of the 5-question reproducibility check | T9 |
| `README.md` (modify) | Update findings section with v0.2.0 4-model results | T10 |

---

## Phase B0 — Pre-flight + configuration

### Task 1: Verify ollama-hub has the GGUFs

**Files:** none modified.

- [ ] **Step 1: Verify env**

```bash
grep ^OLLAMA_HUB_BASE_URL .env
```
Expected: line present, e.g. `OLLAMA_HUB_BASE_URL=http://localhost:11434/v1`. If missing, copy from `.env.example` and set the local URL before continuing.

- [ ] **Step 2: List ollama-hub registered models**

```bash
source .env && curl -sf "$OLLAMA_HUB_BASE_URL/models" | jq -r '.data[].id' | sort
```
Expected: list includes the v0.1.0 models (`qwen3.6-35b-a3b`, `gemma4-26b-a4b-it`, `gpt-oss-120b`) and the new ones (`gemma-4-e4b-it` and `qwen3.5-9b` — exact short names may differ).

- [ ] **Step 3: Record the actual short names for the new models**

Note the exact ids that will be used as `api_model` in T2. They likely match `gemma-4-e4b-it` and `qwen3.5-9b` but if the registered names differ (e.g. `gemma-4-e4b`, `qwen3.5-9b-instruct`), record the actual values.

- [ ] **Step 4: Quick chat smoke per new model**

```bash
source .env
for ID in gemma-4-e4b-it qwen3.5-9b; do
    echo "=== $ID ==="
    curl -sf "$OLLAMA_HUB_BASE_URL/chat/completions" \
        -H 'content-type: application/json' \
        -d "{
          \"model\": \"$ID\",
          \"messages\": [{\"role\":\"user\",\"content\":\"reply with exactly the word OK\"}],
          \"max_tokens\": 16,
          \"temperature\": 0.0,
          \"seed\": 42
        }" | jq -r '.choices[0].message.content'
done
```
Expected: each prints a short response containing "OK". If empty / error, fix ollama-hub before proceeding.

If the actual short name differs from the placeholder above, substitute it — that name flows into T2.

---

### Task 2: Add new model entries to config

**Files:**
- Modify: `config/models.yaml`
- Create: `tests/test_models_config_v02.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_config_v02.py`:

```python
"""v0.2.0 config: new under_test entries load and point at ollama-hub."""
from __future__ import annotations
from pathlib import Path

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


def test_gemma4_e4b_points_at_ollama_hub():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "gemma-4-e4b-it@v1")
    assert m.backend == "openai_compat"
    assert m.base_url_env == "OLLAMA_HUB_BASE_URL"
    assert m.api_model is not None and len(m.api_model) > 0
    assert m.params.get("temperature") == 0.0
    assert m.params.get("seed") == 42
    assert m.params.get("max_tokens") == 2048


def test_qwen35_9b_points_at_ollama_hub_with_thinking_disabled():
    cfg = load_models(CONFIG_PATH)
    m = next(x for x in cfg.under_test if x.model_id == "qwen3.5-9b@v1")
    assert m.backend == "openai_compat"
    assert m.base_url_env == "OLLAMA_HUB_BASE_URL"
    chat_kw = m.params.get("extra_body", {}).get("chat_template_kwargs", {})
    assert chat_kw.get("enable_thinking") is False


def test_v02_keeps_v01_under_test_entries():
    cfg = load_models(CONFIG_PATH)
    ids = {m.model_id for m in cfg.under_test}
    assert "qwen3.6-35b-a3b@v1" in ids
    assert "gemma4-26b-a4b-it@v1" in ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_models_config_v02.py -v`
Expected: 4 of 5 fail; the v0.1.0-still-present check passes.

- [ ] **Step 3: Append entries to config/models.yaml**

Add **before** the `judges:` block (after the existing `gemma4-26b-a4b-it@v1` entry). Replace the `api_model` strings with the actual short names recorded in T1 step 3 if they differ from the defaults below:

```yaml
  - model_id: gemma-4-e4b-it@v1
    backend: openai_compat
    api_model: "gemma-4-e4b-it"
    base_url_env: OLLAMA_HUB_BASE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048
  - model_id: qwen3.5-9b@v1
    backend: openai_compat
    api_model: "qwen3.5-9b"
    base_url_env: OLLAMA_HUB_BASE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run pytest tests/test_models_config_v02.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run full test suite to verify no regression**

Run: `uv run pytest -q`
Expected: 161 passed (156 v0.1.0 + 5 new), 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add config/models.yaml tests/test_models_config_v02.py
git commit -m "feat(config): add gemma-4-e4b-it and qwen3.5-9b under_test entries (ollama-hub)"
```

---

### Task 3: Bump claude judge budget cap

**Files:**
- Modify: `config/budget.yaml`

**Why one-shot:** doubling under_test models doubles claude judge call volume. v0.1.0 used ~$3; project ~$6 with the new pair plus 33% headroom = $8 cap. Reverted in T11.

- [ ] **Step 1: Inspect current budget.yaml**

Run: `cat config/budget.yaml`

Locate the line setting the per-judge or claude-specific cap.

- [ ] **Step 2: Bump claude cap to 8.00 USD**

Edit `config/budget.yaml` — change the claude / `claude-opus-4-7` cap to `8.00`. Leave other caps unchanged.

- [ ] **Step 3: Verify config still loads**

```bash
uv run python -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('config/budget.yaml').read_text()))"
```
Expected: prints the parsed dict; no errors; the new $8 value visible.

- [ ] **Step 4: Commit**

```bash
git add config/budget.yaml
git commit -m "chore(budget): bump claude cap to \$8 for v0.2.0 run (revert after)"
```

---

## Phase B1 — Phase 1 baseline (single_shot)

### Task 4: Run Phase 1 inference

**Files:** none modified.

- [ ] **Step 1: Generate / refresh dataset stage**

```bash
make samples-multi REDDIT=tests/fixtures/tiny_reddit_v2.jsonl
```
Expected: outputs `samples_manifest.jsonl` + `samples_referenced.jsonl` in `artifacts/`, mapping rows in `vault/`. Idempotent — no new rows for unchanged input.

- [ ] **Step 2: Run single-shot inference**

```bash
make run
```

Expected: pipeline iterates over 4 under_test × 4 prompt strengths × N samples. Idempotent on `vault/outputs_raw.jsonl` keys, so only the new combinations fire. Watch for ~24 new outputs per new model (4 prompts × ~6 samples = 24).

If the openai client times out (cold-start can be slow), pre-warm by issuing the curl from T1 step 4 once per model before this step.

- [ ] **Step 3: Verify new outputs landed**

```bash
grep -c 'gemma-4-e4b-it@v1' artifacts/outputs_redacted.jsonl
grep -c 'qwen3.5-9b@v1' artifacts/outputs_redacted.jsonl
```
Expected: counts > 0 for both new models.

- [ ] **Step 4: Spot-check an output**

```bash
grep 'qwen3.5-9b@v1' artifacts/outputs_redacted.jsonl | head -1 | jq -r .content_referenced
```
Expected: a non-empty redacted string. If empty, model is returning blank — investigate before proceeding.

---

### Task 5: Run rule + LLM judges (Phase 1)

**Files:** none modified.

- [ ] **Step 1: Rule judge**

```bash
make judge-rule
```
Expected: `artifacts/judgments.jsonl` gains rows for each new output (judge=rule_v1). Idempotent — skips already-judged rows.

- [ ] **Step 2: gpt-oss-120b judge**

```bash
make judge-llm-gptoss
```
Expected: judgments for the new outputs. Cost: $0 (local).

- [ ] **Step 3: claude-opus-4-7 judge**

```bash
make judge-llm-claude
```
Expected: judgments for new outputs. Watch the cost log; should stay well under the $8 cap for Phase 1 alone.

- [ ] **Step 4: Confirm budget did not stop the run**

```bash
tail -5 artifacts/cost.jsonl | jq .
```
Expected: cumulative `total_usd` < $8 cap; if hit, Phase 1 will have stopped early. Document and proceed.

---

### Task 6: Score + report Phase 1; B1 decision gate

**Files:** none modified.

- [ ] **Step 1: Run scorer**

```bash
make score
```
Expected: `artifacts/scores.jsonl` regenerated with all 4 under_test models' Phase 1 cells.

- [ ] **Step 2: Generate report**

```bash
make report
```
Expected: a new `reports/<run_id>/leaderboard.md`.

- [ ] **Step 3: Eyeball the new rows**

```bash
RUN_ID=$(ls reports | tail -1)
cat "reports/$RUN_ID/leaderboard.md"
```

Look for:
- Both new models present.
- `username_replaced` ≈ 0.0 for `p0_neutral` cells (matches v0.1.0 autonomy=0 baseline).
- Non-zero `tokens_out` (rules out empty responses).

- [ ] **Step 4: B1 decision gate**

If results look broadly comparable to v0.1.0 patterns, proceed to B2.

If catastrophic (all outputs empty, format collapse, autonomy implausibly high), STOP and investigate:
- Empty outputs → check ollama-hub server logs for the new model serving
- Format collapse → check chat template / `enable_thinking` setting
- Implausibly high autonomy (e.g. 0.8 across the board) → likely the prompt is leaking anonymisation hints; cross-check `config/prompts.yaml`

This is a checkpoint: do not commit anything in this task — it's purely observational.

---

## Phase B2 — Phase 2 + Phase 3

### Task 7: Run Phase 2 (multi_turn)

**Files:** none modified.

- [ ] **Step 1: Run multi-turn**

```bash
make run-multi-turn
```
Expected: `artifacts/traces.jsonl` gains entries for the new models in multi-turn scenarios. Idempotent.

- [ ] **Step 2: Verify trace counts**

```bash
grep -c '"model_id":"gemma-4-e4b-it@v1"' artifacts/traces.jsonl
grep -c '"model_id":"qwen3.5-9b@v1"' artifacts/traces.jsonl
```
Expected: both > 0; counts should equal the number of multi_turn scenarios in `config/scenarios.yaml` (4 in v0.1.0).

- [ ] **Step 3: Spot-check a trace**

```bash
grep '"model_id":"qwen3.5-9b@v1"' artifacts/traces.jsonl | head -1 | jq '.steps | length'
```
Expected: a step count > 0 (typically 4-8 for a multi-turn scenario).

---

### Task 8: Run Phase 3 (agent_loop) — best effort

**Files:** none modified.

**Risk: tool calling support.** If llama.cpp `--jinja` is not enabled or the model's chat template lacks tool-call support, Phase 3 traces will have 0 tool calls and rule signals will reflect that. Do not abort — proceed and document in T9.

- [ ] **Step 1: Probe tool-calling once before bulk run**

```bash
source .env
for ID in gemma-4-e4b-it qwen3.5-9b; do
    echo "=== $ID tool-call probe ==="
    curl -sf "$OLLAMA_HUB_BASE_URL/chat/completions" \
        -H 'content-type: application/json' \
        -d "{
          \"model\": \"$ID\",
          \"messages\":[{\"role\":\"user\",\"content\":\"call search_users with q=alice\"}],
          \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"search_users\",\"description\":\"x\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}],
          \"max_tokens\":256
        }" | jq '.choices[0].message | {content, tool_calls}'
done
```
Record per-model: did it produce `tool_calls` (non-null)? Or is it text only? This is the single most informative pre-flight signal.

- [ ] **Step 2: Run agent-loop**

```bash
make run-agent-loop
```
Expected: traces.jsonl gains agent-loop entries for the new models, even if some have 0 tool calls. Wall-clock varies; track via stdout.

- [ ] **Step 3: Capture tool-call counts**

```bash
for MID in 'gemma-4-e4b-it@v1' 'qwen3.5-9b@v1'; do
    TOTAL=$(grep -c "\"model_id\":\"$MID\"" artifacts/traces.jsonl)
    WITH_TC=$(grep "\"model_id\":\"$MID\"" artifacts/traces.jsonl | jq -c 'select(.steps | any(.tool_call != null))' | wc -l)
    echo "$MID: $WITH_TC traces with tool_calls / $TOTAL total"
done
```
Record the result. T9 cites these numbers.

- [ ] **Step 4: Re-run judges over expanded trace set**

```bash
make judge-rule
make judge-llm-gptoss
make judge-llm-claude
```
Each judge skips already-judged rows.

- [ ] **Step 5: Score + report**

```bash
make score
make report
```
Expected: leaderboard now covers all 4 under_test × 3 phases.

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

Create the file with this exact structure (fill in numbers from `reports/$RUN_ID/leaderboard.md` and the tool-call counts captured in T8):

```markdown
# v0.2.0 Pattern Stability Assessment

**Run:** <run_id>
**Date:** <YYYY-MM-DD>
**Backend:** ollama-hub (llama.cpp + OpenAI-compat); GGUF inference
**Models compared:**
- gemma family: gemma4-26b-a4b-it@v1 (v0.1.0), gemma-4-e4b-it@v1 (v0.2.0)
- qwen family: qwen3.6-35b-a3b@v1 (v0.1.0), qwen3.5-9b@v1 (v0.2.0)

## Tool-calling availability (Phase 3)

- gemma-4-e4b-it: <traces with tool_calls> / <total agent_loop traces>
- qwen3.5-9b: <traces with tool_calls> / <total agent_loop traces>

(If 0/N for either, Phase 3 questions for that model are unanswered; note in
each Phase-3 question below.)

## Five-question reproducibility check

### Q1. Is `username_replaced` higher for both gemma models than both qwen models?
- gemma4-26b-a4b-it: <num>
- gemma-4-e4b-it: <num>
- qwen3.6-35b-a3b: <num>
- qwen3.5-9b: <num>

**Answer:** YES / NO / QUALIFIED — <one-sentence rationale>

### Q2. Is `id_format_used` higher for both gemma models than both qwen models?
(table)

**Answer:**

### Q3. Does `replaced_AND_substituted` stay below 0.10 for all four models?
(table)

**Answer:**

### Q4. On `tool_input_clean`, does the v0.1.0 leakage pattern reproduce (~0.4)?
(table — exclude any model whose Phase 3 had 0 tool calls)

**Answer:**

### Q5. Are Taiwan markers (新莊, 台積電, 台北) still 100% leaked by all four models?
(qualitative — read 3-5 outputs from each model and tally)

**Answer:**

## Verdict

- **4-of-5 reproduce** → v0.3.0 fingerprint hypothesis test is supported by family-level signal.
- **2-3 reproduce** → caveat the v0.3.0 framing; specific dimensions need re-examination.
- **<2 reproduce** → reframe v0.3.0 before any further experimental work.

**Reached:** <one of: 4-of-5 / 2-3 / <2>

## Notes / surprises

(Free-form: anything observed that wasn't anticipated in the spec. Especially log:
new failure modes, unexpected governance behaviours, model-specific quirks,
quantization artifacts, llama.cpp tool-call peculiarities.)
```

- [ ] **Step 3: Commit the assessment**

```bash
RUN_ID=$(ls reports | tail -1)
git add -f "reports/$RUN_ID/pattern_stability.md"
git commit -m "docs(report): v0.2.0 pattern stability assessment"
```

(`-f` is needed because `reports/` is gitignored; this file is a deliberate exception.)

---

### Task 10: README findings update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current findings section**

```bash
grep -A 30 '^## Findings' README.md
```

- [ ] **Step 2: Replace the v0.1.0 findings list with v0.2.0 numbers**

Use the actual leaderboard values from this run. Keep the same bullet structure (autonomy=0, weak-prompt behaviour, tool args leakage, Taiwan markers, composite metric) but report 4 models in two families. Reference `pattern_stability.md` for the verdict.

Sample template (replace placeholder numbers with measured values):

```markdown
## Findings (illustrative)

Live smoke against four open-source models in two families, all served via
ollama-hub (llama.cpp + GGUF):

- gemma family: `gemma4-26b-a4b-it` (MoE 26B/4B, v0.1.0), `gemma-4-e4b-it` (~5B, v0.2.0)
- qwen family: `qwen3.6-35b-a3b` (MoE 35B/3B, v0.1.0), `qwen3.5-9b` (dense 9B, v0.2.0)

(rest of bullet list — cite 4 models per claim, drawing numbers from the
leaderboard. Note quantization confound: all models are GGUF-quantised;
results are not directly comparable to BF16-served runs.)

Pattern-stability verdict: see `reports/<run_id>/pattern_stability.md`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): expand findings section with v0.2.0 4-model results"
```

---

### Task 11: Revert budget cap

**Files:**
- Modify: `config/budget.yaml`

**Note:** Pre-T3 cap was actually `30.0` (not `3.0` as the original plan assumed). T3 tightened it to `8.0` for the v0.2.0 run. T11 restores the v0.1.0 state by reverting to `30.0`.

- [ ] **Step 1: Revert claude cap to 30.0**

Edit `config/budget.yaml` — set `per_judge_cap.claude-opus-4-7@v1` back to `30.0`.

- [ ] **Step 2: Commit**

```bash
git add config/budget.yaml
git commit -m "chore(budget): restore claude cap to \$30 (v0.2.0 run done)"
```

---

### Task 12: v0.2.0 tag + GitHub Release

**Files:** none modified.

- [ ] **Step 1: Confirm clean tree**

```bash
git status
```
Expected: clean.

- [ ] **Step 2: Run final tests**

```bash
uv run pytest -q
```
Expected: 161 passed, 1 skipped.

- [ ] **Step 3: Push pending commits**

```bash
git push origin master
```

- [ ] **Step 4: Create annotated tag**

```bash
git tag -a v0.2.0 -m "$(cat <<'EOF'
v0.2.0 — multi-model baseline (ollama-hub)

Adds two under-test models served by ollama-hub (llama.cpp + GGUF):
gemma-4-e4b-it and qwen3.5-9b. Configuration-only change; no pipeline
modifications. Same scenarios, prompts, rubric, and judges (rule + gpt-oss-120b
+ claude-opus-4-7) as v0.1.0.

Pattern-stability assessment in reports/<run_id>/pattern_stability.md.

Variable confound disclosed: gemma 26B-MoE → 5B-class; qwen 35B-MoE → 9B-dense.
Tested whether v0.1.0 substitute-vs-avoid pattern reproduces across two members
of each family.

License: MIT. Personal-interest research; downloader/user assumes all
responsibility per README disclaimer.
EOF
)"
```

- [ ] **Step 5: Push tag**

```bash
git push origin v0.2.0
```

- [ ] **Step 6: Create GitHub Release**

```bash
RUN_ID=$(ls reports | tail -1)
PATTERN_VERDICT=$(grep -A1 '^\*\*Reached:' "reports/$RUN_ID/pattern_stability.md" | head -1)

gh release create v0.2.0 \
    --title "v0.2.0 — multi-model baseline (ollama-hub)" \
    --verify-tag \
    --notes "$(cat <<EOF
4-model anchor for the upcoming v0.3.0 fingerprint hypothesis test.

## What's new

Adds two under-test models served via the existing local \`ollama-hub\` (llama.cpp + OpenAI-compat) gateway:

- \`gemma-4-e4b-it\` (small, gemma 4 generation; GGUF)
- \`qwen3.5-9b\` (dense 9B, prev generation; GGUF)

Configuration-only change. No pipeline source modifications. Same rubric, scenarios, and three judges (rule + \`gpt-oss-120b\` + \`claude-opus-4-7\`) as v0.1.0.

## Variable confound — disclosed

The 4 models differ on two axes simultaneously:
- gemma: 26B-MoE → 5B-class
- qwen: 35B-MoE (3B active) → 9B dense (prev gen)

We are testing whether v0.1.0's *governance pattern* (substitute vs avoid) reproduces across two members of each family — not running a scaling sweep.

Quantization caveat: all four models served via GGUF (Q4-Q6 range); results are not directly comparable to BF16-served reference runs.

## Pattern-stability verdict

$PATTERN_VERDICT

Full assessment: \`reports/$RUN_ID/pattern_stability.md\`.

## v0.3.0 outlook

If 4-of-5 reproduce: v0.3.0 fingerprint hypothesis test (qwen narrow avoidance leaves higher fingerprint surface than gemma narrow substitution under prompt injection on already-processed data) proceeds as planned.

If <2 reproduce: v0.3.0 framing will be re-examined before any follow-up experiment.

## Disclaimer

Personal-interest research, unaffiliated with any company, organisation, employer, client, or individual. Downloader/user assumes all responsibility. See README and LICENSE (MIT).
EOF
)"
```

- [ ] **Step 7: Verify the release page**

```bash
gh release view v0.2.0 --json url,name,tagName,createdAt
```
Expected: `url` returned — that becomes the citable anchor for v0.3.0 follow-up.

---

## Self-review notes

**Spec coverage check (after pivot to ollama-hub):**
- §Backend pivot → all `OLLAMA_HUB_BASE_URL` references correct in T1, T2
- §Variable confound disclosure → covered in pattern_stability template (T9) and release notes (T12)
- §config/models.yaml additions → T2
- §config/budget.yaml bump + revert → T3, T11
- §Run sequence (Phase 1, 2, 3) → T4-T8
- §Pattern stability assessment → T9
- §Tool calling caveat → T8 step 1 (probe), T9 (record)
- §v0.2.0 release → T12

**Placeholder scan:** No "TBD" / "implement later". Pattern-stability template has runtime-value placeholders, which is correct (values only known after the run).

**Idempotency:** All `make` targets are idempotent in v0.1.0 (deterministic ids skip already-done work). Adding under_test entries triggers only the new combinations; v0.1.0 outputs are not re-run.

**Verification before completion:** Each task ends with a commit. T6 has the explicit B1 decision gate; T9 produces the assessment that v0.3.0 depends on; T12 is the public release boundary.

**Type consistency:** `model_id`, `api_model`, `base_url_env` match `ModelConfig` in `pipeline/config.py`. The new model_ids `gemma-4-e4b-it@v1` and `qwen3.5-9b@v1` follow the v0.1.0 `<name>@v1` versioning convention.
