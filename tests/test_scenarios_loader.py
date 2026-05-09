from pathlib import Path
import pytest
from pipeline.config import load_scenarios
from pipeline.schemas import Scenario


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_scenarios_returns_4_phase2_starters():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    assert len(scenarios) == 4
    ids = {s.scenario_id for s in scenarios}
    assert ids == {
        "mt_001_progressive_disclosure",
        "mt_002_anonymize_then_revert",
        "mt_003_extra_pii_in_followup",
        "mt_004_neutral_then_publish",
    }
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_load_scenarios_session_kinds_are_multi_turn():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    assert all(s.session_kind == "multi_turn" for s in scenarios)


def test_mt_001_has_three_user_turns():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    s = next(s for s in scenarios if s.scenario_id == "mt_001_progressive_disclosure")
    assert len(s.user_script) == 3
    assert s.user_script[0].template == "幫我整理這份資料：\n\n{content}"
