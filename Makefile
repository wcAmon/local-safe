PYTHON := uv run python
CLI    := $(PYTHON) -m pipeline.cli
REDDIT ?= tests/fixtures/tiny_reddit.jsonl

.PHONY: help test samples run judge-rule judge-llm score report all clean-artifacts

help:
	@echo "Targets:"
	@echo "  test          Run full test suite"
	@echo "  samples       Stage 1 (REDDIT=path/to/reddit.jsonl, default fixture)"
	@echo "  run           Stage 2 — single-shot inference (under_test models)"
	@echo "  judge-rule    Stage 3a — deterministic rule judge"
	@echo "  judge-llm     Stage 3b — LLM judge gpt-oss-120b@v1"
	@echo "  score         Stage 4 — aggregate"
	@echo "  report        Stage 5 — markdown leaderboard"
	@echo "  all           samples + run + judge-rule + judge-llm + score + report"
	@echo "  clean-artifacts  remove artifacts/ and vault/ contents (KEEP .gitkeep)"

test:
	uv run pytest

samples:
	$(CLI) build-samples --reddit $(REDDIT)

run:
	$(CLI) run

judge-rule:
	$(CLI) judge-rule

judge-llm:
	$(CLI) judge-llm --judge gpt-oss-120b@v1

score:
	$(CLI) score

report:
	$(CLI) report

all: samples run judge-rule judge-llm score report

clean-artifacts:
	find vault     -mindepth 1 ! -name .gitkeep -delete
	find artifacts -mindepth 1 ! -name .gitkeep -delete
