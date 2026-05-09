"""Shared pytest fixtures."""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_reddit_path() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "tiny_reddit.jsonl"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tiny_reddit_v2_path() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "tiny_reddit_v2.jsonl"
