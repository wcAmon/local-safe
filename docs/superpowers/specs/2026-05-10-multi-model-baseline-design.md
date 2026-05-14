# Multi-Model Baseline (v0.2.0) — Design Spec

**Date:** 2026-05-10
**Status:** Updated 2026-05-10 — backend pivoted from a new transformers-inference HTTP server to the existing `ollama-hub` (llama.cpp + OpenAI-compat). All section references to "transformers-inference server" and `TRANSFORMERS_INFERENCE_URL` should be read as `ollama-hub` and `OLLAMA_HUB_BASE_URL`. The two new models are served as GGUF via llama.cpp.
**Scope:** Extend the `local-safe` v0.1.0 benchmark to two additional under-test models (`gemma-4-E4B-it`, `Qwen3.5-9B`) served via `ollama-hub` (the same gateway the v0.1.0 models already use). Produce a 4-model leaderboard that anchors the upcoming v0.3.0 fingerprint hypothesis.

## Why

`v0.1.0` measured `qwen3.6-35b-a3b` and `gemma4-26b-a4b-it` (both MoE, ~3-4B active). The findings flagged two narrow PII tactics:

- gemma: substitute (replace `alice_92` with `<<U-001>>`)
- qwen: avoid (omit `alice_92` entirely, leave surrounding context intact)

The next research step (v0.3.0) tests a sharper hypothesis: **qwen's narrow avoidance leaves a higher fingerprint surface than gemma's narrow substitution**, especially under prompt injection on already-processed data. To make that comparison defensible we first need baseline data on a second pair of models from the same families:

- `gemma-4-E4B-it` (small dense / mixed, gemma 4 generation)
- `Qwen3.5-9B` (dense, previous generation)

If the gemma-substitute / qwen-avoid pattern is **family-stable** (visible in 4 models, not 2), the v0.3.0 fingerprint experiment has solid footing. If it's **idiosyncratic** to the v0.1.0 model pair, v0.3.0 needs to refocus before any article goes out.

## Goal

Run all three existing benchmark phases (`single_shot`, `multi_turn`, `agent_loop`) against the two new models, using identical scenarios, prompts, judges, and rubric as v0.1.0. Publish a 4-model leaderboard. Tag as `v0.2.0`.

## Non-goals

These are explicitly out of scope:

- Any pipeline code changes (`pipeline/` modules). v0.1.0 architecture is the contract.
- New scoring dimensions, scenarios, prompts, or rubric edits. Adding any of these would confound the family-stability test.
- New judges. Re-use rule + `gpt-oss-120b` + `claude-opus-4-7`. (Per user direction: "既有judges".)
- Server work — that's covered by the parallel spec at `~/projects/transformers-inference/docs/specs/2026-05-10-openai-compat-server-design.md`.
- Fingerprint hypothesis testing — that's v0.3.0; this spec sets it up but does not run it.
- Comparing `gemma-4-E4B-it` to `gemma4-26b-a4b-it` as a "scaling study". Different parameter counts and architectures; we treat them as four data points within a common family-pattern test, not a scaling sweep.

## Variable confound — disclosed up front

The four models differ on **two axes simultaneously**:

| Family | v0.1.0 model | v0.2.0 model |
|---|---|---|
| gemma | `gemma4-26b-a4b-it` (MoE, 26B/4B) | `gemma-4-E4B-it` (small, 5B class) |
| qwen | `qwen3.6-35b-a3b` (MoE, 35B/3B) | `Qwen3.5-9B` (dense, prev gen) |

Findings will be reported with both axes called out. We are not claiming "qwen vs gemma at fixed size"; we are testing whether the *governance pattern* (substitute vs avoid) holds across two members of each family. If yes → family-level signal; if no → caution for v0.3.0.

## Changes to repository

This spec touches only configuration and budget. No `pipeline/` source files are modified.

### `config/models.yaml` — add 2 under_test entries

Both new models point at the transformers-inference server (default port `8001`). Existing entries unchanged.

```yaml
under_test:
  # ... existing entries (qwen3.6-35b-a3b@v1, gemma4-26b-a4b-it@v1) ...

  - model_id: gemma-4-e4b-it@v1
    backend: openai_compat
    api_model: "gemma-4-e4b-it"     # short name as registered in ollama-hub; verify via /v1/models
    base_url_env: OLLAMA_HUB_BASE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048

  - model_id: qwen3.5-9b@v1
    backend: openai_compat
    api_model: "qwen3.5-9b"          # short name as registered in ollama-hub; verify via /v1/models
    base_url_env: OLLAMA_HUB_BASE_URL
    params:
      temperature: 0.0
      seed: 42
      max_tokens: 2048
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
```

Field names match the existing `ModelConfig` schema (`pipeline/config.py`): `model_id` (local identifier), `api_model` (wire-format id sent in request body), `base_url_env` (env var name). `temperature: 0.0`, `seed: 42`, `max_tokens: 2048` match v0.1.0's existing entries for direct comparability. `api_model` short names follow the v0.1.0 ollama-hub convention; the implementation plan starts with a `/v1/models` GET to confirm the actual names registered on the running gateway.

The `enable_thinking: false` mirrors the v0.1.0 fix for `qwen3.6` — Qwen ChatML chat templates default to thinking-on for some configs and we want comparable behaviour.

### `.env.example` — no new var needed

`OLLAMA_HUB_BASE_URL` is already documented from v0.1.0. No additions.

### Tool calling caveat (Phase 3 risk)

Phase 3 (`agent_loop`) requires the model to emit tool calls in OpenAI format. With `ollama-hub` (llama.cpp), this depends on llama.cpp being run with `--jinja` and the model's chat template supporting tool calling. For the new pair:

- **Qwen3.5-9B**: Qwen3 family tool-call support in llama.cpp is mature; expected to work.
- **gemma-4-E4B-it**: Gemma 4 tool-call support in llama.cpp is newer; format compatibility may vary.

The implementation plan attempts Phase 3 and records observed behaviour. If tool calls are malformed or absent for either model, the plan documents the gap in `pattern_stability.md` and proceeds with whatever Phase 3 data can be obtained — partial Phase 3 is more informative than no Phase 3 attempt.

### `config/budget.yaml` — bump claude judge cap

v0.1.0 used ~$3 of `claude-opus-4-7` budget per phase 3 run. Adding two under_test models doubles judge work: bump cap to **$8** for the v0.2.0 run, then revert to $3 for steady-state. The cap is a stop-and-report safety, not a target.

### `tests/fixtures/` — no changes

Same `tiny_reddit.jsonl` / `tiny_reddit_v2.jsonl`. Adding fixture entries would change the *content* of the benchmark and break v0.1.0 comparability.

## Run sequence

Two milestones, gated on the corresponding milestones in the transformers-inference spec.

### Milestone B1 — phase 1 baseline (gated on server M1)

Pre-conditions:
- Server M1 done; `curl http://127.0.0.1:8001/v1/models` returns both ids.
- `.env` has `TRANSFORMERS_INFERENCE_URL`.

Sequence:
```bash
uv run pytest                          # 156 tests passing
make run                               # phase 1: single_shot, all 4 under_test × 4 prompt strengths × N samples
make judge-rule                        # rule judge (free)
make judge-llm openai_compat:gpt-oss   # gpt-oss judge
make judge-llm anthropic:claude-opus   # claude judge
make score && make report
git diff --stat reports/               # eyeball: did the leaderboard differ from v0.1.0?
```

**Decision point:** if the new models' phase 1 results look catastrophically broken (e.g., empty outputs, format collapse), stop here and debug the server before continuing. Expected pattern: similar autonomy=0.0 baseline as v0.1.0; if it's wildly off (say, 0.8) something is wrong with the chat template or generation params, not with the model's governance behaviour.

### Milestone B2 — phase 2 + phase 3 (gated on server M2)

Pre-conditions:
- B1 results reviewed and accepted.
- Server M2 done; tool-calling parsers verified for both models.

Sequence:
```bash
make run-multi-turn                    # phase 2
make run-agent-loop                    # phase 3 (tool calling)
make judge-rule
make judge-llm openai_compat:gpt-oss
make judge-llm anthropic:claude-opus
make score && make report
```

Expected wall-clock: 1-3 hours for full multi-turn + agent-loop on DGX Spark, plus ~30 minutes of judging.

## Expected outputs

### Leaderboard

`reports/<run_id>/leaderboard.md` — same schema as v0.1.0 but with 4 rows. Columns include `username_replaced`, `id_format_used`, `id_consistency`, `replaced_AND_substituted` (composite), and the phase 3 hard signals (`tool_input_clean`, `prompt_injection_resisted`, `error_path_clean`).

### Pattern-stability assessment

A short `reports/<run_id>/pattern_stability.md` (a manual write-up after results land), answering:

1. Is `username_replaced` higher for both gemma models than both qwen models?
2. Is `id_format_used` higher for both gemma models than both qwen models?
3. Does `replaced_AND_substituted` stay below 0.10 for all four models (autonomous-governance failure pattern from v0.1.0)?
4. On `tool_input_clean`, does the leakage pattern reproduce (~0.4 across the board)?
5. Are Taiwan markers (`新莊`, `台積電`, etc.) still 100% leaked?

If 4-of-5 reproduce → v0.3.0 fingerprint hypothesis test proceeds as planned.
If <2 reproduce → reframe v0.3.0 before continuing.

## v0.2.0 release

Tag and GitHub Release after B2 leaderboard is generated.

```
git tag -a v0.2.0 -m "..."
git push origin v0.2.0
gh release create v0.2.0 --title "..." --notes-file <prepared>
```

Release notes summarise: 4-model leaderboard, pattern-stability finding (yes/no/qualified), updated `replaced_AND_substituted` numbers, and a forward-pointer to v0.3.0 fingerprint hypothesis. Public-research disclaimer carried forward from v0.1.0 README.

## Decisions

**Why not also extend `vault/` retention or judge-call budget cap permanently?** The bumped cap is a one-shot for the larger run; permanent change would lower the safety value of the cap. Revert in the same PR that adds the v0.2.0 tag.

**Why not add `transformers_inference` as its own backend in `config/models.yaml`?** The transformers-inference server *is* OpenAI-compatible. Adding a distinct backend label would force a new adapter in `pipeline/serving/` for no functional difference. Use `openai_compat` with a different `base_url_env`.

**Why run all three phases instead of phase 1 only?** Per user direction. Even if phase 3 is the most informative for the v0.3.0 hypothesis (tool-call leakage signals fingerprint surface most directly), phase 1 + 2 give the autonomy-baseline comparability anchor. Skipping them would leave the v0.2.0 leaderboard non-comparable to v0.1.0.

**Why not parameterise `seed` across multiple values for variance?** Seed-sweep variance is a real concern but would add 4× cost. Held over for a future v0.4.0 reliability spec; v0.2.0 keeps `seed=0` to match v0.1.0 directly.

## Risks

- **Server M2 tool-call parser is the critical path.** If gemma-4 or qwen3.5 tool calls aren't reliably parseable, phase 3 results collapse. Mitigation: B1 happens first, gives confidence the basic plumbing works before B2 commits.
- **Wall-clock blow-up.** DGX Spark inference for 9B models at FP16 is much slower than the API-served v0.1.0 models. Each phase could take 10× longer. Mitigation: run B1 phase 1 first to measure, project total time, decide whether to drop temperature variance / shorten max_tokens.
- **Judge cost overrun.** $8 cap is conservative but if phase 3 tool-trace context is much longer than v0.1.0, claude-opus tokens balloon. Mitigation: stop-and-report cap engages at $8; if hit, finalise with rule + gpt-oss judges only and note the kappa caveat.
- **Reproducibility risk.** Findings could be sensitive to chat-template versions, transformers library version, dtype. Mitigation: pin transformers version in transformers-inference, record git SHAs of both repos in the v0.2.0 release notes.

## Open questions to resolve before B1

1. Should `gemma-4-E4B-it` use `temperature=0.7` (matches v0.1.0) or its model-card-recommended sampling (if different)? Decision: stay with 0.7 unless model card explicitly forbids.
2. Should `Qwen3.5-9B` need `repetition_penalty` or other generation params for stability? Defer until first phase 1 run; tune only if the output is visibly degraded.
3. Are 156 unit tests still passing after adding the two model entries (config-only change shouldn't break anything but always verify)? Run `uv run pytest` as the first step of B1.
