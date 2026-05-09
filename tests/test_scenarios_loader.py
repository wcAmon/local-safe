from pathlib import Path
import pytest
from pipeline.config import load_scenarios, load_tools
from pipeline.schemas import Scenario


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_scenarios_returns_4_phase2_starters():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    assert len(scenarios) == 9  # 4 Phase 2 + 5 Phase 3
    mt_ids = {s.scenario_id for s in scenarios if s.session_kind == "multi_turn"}
    assert mt_ids == {
        "mt_001_progressive_disclosure",
        "mt_002_anonymize_then_revert",
        "mt_003_extra_pii_in_followup",
        "mt_004_neutral_then_publish",
    }
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_load_scenarios_session_kinds_are_multi_turn_or_agent_loop():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    assert all(s.session_kind in ("multi_turn", "agent_loop") for s in scenarios)


def test_mt_001_has_three_user_turns():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    s = next(s for s in scenarios if s.scenario_id == "mt_001_progressive_disclosure")
    assert len(s.user_script) == 3
    assert s.user_script[0].template == "幫我整理這份資料：\n\n{content}"


def test_load_tools_returns_4_specs():
    tools = load_tools(REPO_ROOT / "config" / "tools.yaml")
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"fetch_user_history", "fetch_user_profile", "search_users", "send_notification"}


def test_load_tools_parameters_structure():
    tools = load_tools(REPO_ROOT / "config" / "tools.yaml")
    fetch_history = next(t for t in tools if t.name == "fetch_user_history")
    assert fetch_history.parameters["type"] == "object"
    assert "user_id" in fetch_history.parameters["properties"]
    assert fetch_history.parameters["required"] == ["user_id"]


def test_load_tools_legacy_optional_fields_visible():
    tools = load_tools(REPO_ROOT / "config" / "tools.yaml")
    profile = next(t for t in tools if t.name == "fetch_user_profile")
    for legacy_field in ("full_name", "email", "address"):
        assert legacy_field in profile.parameters["properties"]
        assert "Legacy" in profile.parameters["properties"][legacy_field].get("description", "")


def test_load_scenarios_includes_5_agent_loop():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    agent_ids = {s.scenario_id for s in scenarios if s.session_kind == "agent_loop"}
    assert agent_ids == {
        "ag_001_input_leak",
        "ag_002_args_minimal",
        "ag_003_result_governance",
        "ag_004_prompt_injection",
        "ag_005_error_path",
    }


def test_agent_scenarios_have_required_phase3_fields():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    for s in scenarios:
        if s.session_kind != "agent_loop":
            continue
        assert s.initial_prompt is not None and "{content}" in s.initial_prompt
        assert s.tools_used               # at least one tool
        assert s.max_steps                # configured
        assert s.tested_dimensions        # at least one dimension


def test_ag_001_mock_returns_has_three_entries():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    ag_001 = next(s for s in scenarios if s.scenario_id == "ag_001_input_leak")
    fh_mocks = ag_001.mock_returns["fetch_user_history"]
    assert len(fh_mocks) == 3
    assert fh_mocks[0].args == {"user_id": "user_001"}
    assert fh_mocks[1].is_error is True
    assert fh_mocks[2].args == {}                 # default fallback last
