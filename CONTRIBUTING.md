# Contributing

Thanks for your interest. This is a research / benchmark project; contributions
that broaden coverage of governance dimensions, models, or session shapes are
especially welcome.

## Development setup

```bash
uv sync
cp .env.example .env  # then edit OLLAMA_HUB_BASE_URL and (optional) ANTHROPIC_API_KEY
uv run pytest          # 156 tests should pass without any model server
```

The unit test suite mocks all model calls and runs offline. The opt-in
`RUN_SMOKE=1 uv run pytest tests/test_smoke_e2e.py` runs the full pipeline
against a live OpenAI-compatible endpoint and an Anthropic API key.

## Code style

- Python 3.12+; Pydantic v2 for all schemas
- Type hints on public functions; `from __future__ import annotations` at the
  top of new modules
- Tests-first: each new feature lands with a failing test in the same commit
  or PR, not retrofitted later

## Adding a scoring dimension

1. Decide whether it's deterministic (rule-judged) or behavioural (LLM-judged)
2. For rule-judged: add a `_score_<dim>` function in
   `pipeline/stages/stage3a_rule_judge.py` and wire it into the dispatch dict
3. For LLM-judged: add an entry to `config/rubric.v2.yaml` and add the key to
   `SCORE_KEYS_PHASE3` in `pipeline/stages/stage3b_llm_judge.py`
4. Decide whether the dimension applies to all session kinds or only some, and
   update `DEFAULT_DIMENSIONS_BY_KIND` if it's a subset
5. Mark it in `HARD_SIGNALS` or `SOFT_SIGNALS` in
   `pipeline/stages/stage4_scorer.py` so the scorer knows how to aggregate it
6. Add unit tests under `tests/stages/` covering at least one positive and one
   negative case
7. If it changes the leaderboard schema, update
   `pipeline/stages/stage5_reporter.py` and a fixture under `tests/fixtures/`

## Adding a scenario

Edit `config/scenarios.yaml`. Required fields per scenario:

- `scenario_id` — unique string
- `session_kind` — `single_shot`, `multi_turn`, or `agent_loop`
- `sample_id` — fixture sample to feed in (or `null` for synthetic-only)
- `tested_dimensions` — list of dimension names this scenario actually
  exercises (judges skip dimensions not in this list)
- For multi-turn: `user_script` (list of turns with `{template}`)
- For agent-loop: `initial_prompt`, `tools_used`, `mock_returns`, `max_steps`

## Commit messages

Conventional-commits style:
- `feat(<scope>): ...` — new behaviour
- `fix(<scope>): ...` — bug fix with regression test
- `test(<scope>): ...` — test-only changes
- `docs: ...` — README / doc changes
- `chore: ...` — tooling / config

## What not to commit

The repository has aggressive `.gitignore` rules. Never bypass them:

- `vault/*` — raw model outputs and PII mapping. Local only.
- `artifacts/*` (except `.gitkeep`) — pipeline output JSONL. Regenerated per
  run; local only.
- `reports/*` — leaderboards / preview HTMLs. Regenerated per run.
- `.env` — API keys. The example template is `.env.example`.

If you find yourself wanting to commit a fixture that contains real user data,
stop and synthesise a plausible analogue instead. The benchmark deliberately
avoids real PII even in test fixtures.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see `LICENSE`).
