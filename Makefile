PYTHON := uv run python
CLI    := $(PYTHON) -m pipeline.cli
REDDIT ?= tests/fixtures/tiny_reddit.jsonl

.PHONY: help test samples samples-multi run run-multi-turn run-agent-loop agent judge-rule judge-llm judge-llm-all score report leaderboard all clean-artifacts

help:
	@echo "Targets:"
	@echo "  test              Run full test suite"
	@echo "  samples           Stage 1 (REDDIT=path/to/reddit.jsonl, default fixture)"
	@echo "  samples-multi     Stage 1 with --multi-thread flag"
	@echo "  run               Stage 2 — single-shot inference (under_test models)"
	@echo "  run-multi-turn    Stage 2b — multi-turn scenarios (under_test models)"
	@echo "  run-agent-loop    Stage 2c — agent_loop scenarios (under_test models)"
	@echo "  agent             Alias for run-agent-loop"
	@echo "  judge-rule        Stage 3a — deterministic rule judge"
	@echo "  judge-llm         Stage 3b — default OpenAI cloud LLM judge"
	@echo "  judge-llm-all     Stage 3b — all non-rule judges from models.yaml"
	@echo "  score             Stage 4 — aggregate"
	@echo "  report            Stage 5 — markdown leaderboard"
	@echo "  leaderboard       Refresh README's Current Leaderboard from latest report"
	@echo "  all               samples-multi + run + run-multi-turn + run-agent-loop + judge-rule + judge-llm-all + score + report + leaderboard"
	@echo "  clean-artifacts   remove artifacts/ and vault/ contents (KEEP .gitkeep)"

test:
	uv run pytest

samples:
	$(CLI) build-samples --reddit $(REDDIT)

samples-multi:
	$(CLI) build-samples --reddit $(REDDIT) --multi-thread

run:
	$(CLI) run

run-multi-turn:
	$(CLI) run-multi-turn

run-agent-loop:
	$(CLI) run-agent-loop

# Convenience alias
agent: run-agent-loop

judge-rule:
	$(CLI) judge-rule

judge-llm:
	$(CLI) judge-llm --judge openai-gpt-4.1-mini@v1

judge-llm-all:
	$(CLI) judge-llm-all

score:
	$(CLI) score

report:
	$(CLI) report

leaderboard:
	$(PYTHON) scripts/update_leaderboard.py

all: samples-multi run run-multi-turn run-agent-loop judge-rule judge-llm-all score report leaderboard

clean-artifacts:
	find vault     -mindepth 1 ! -name .gitkeep -delete
	find artifacts -mindepth 1 ! -name .gitkeep -delete
