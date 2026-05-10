from pathlib import Path
import pytest
from pipeline.config import load_scenarios, load_tools
from pipeline.schemas import Scenario


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_scenarios_returns_4_phase2_starters():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    assert len(scenarios) == 17  # 6 multi-shot + 11 agentic workflow
    mt_ids = {s.scenario_id for s in scenarios if s.session_kind == "multi_turn"}
    assert mt_ids == {
        "mt_001_progressive_disclosure",
        "mt_002_anonymize_then_revert",
        "mt_003_extra_pii_in_followup",
        "mt_004_neutral_then_publish",
        "mt_005_cloud_search_safe",
        "mt_006_reverse_leak_pressure",
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


def test_load_tools_returns_9_specs():
    tools = load_tools(REPO_ROOT / "config" / "tools.yaml")
    assert len(tools) == 9
    names = {t.name for t in tools}
    assert names == {
        "fetch_user_history", "fetch_user_profile", "search_users",
        "send_notification", "write_public_report", "delegate_to_large_model",
        "cloud_search_web", "compromised_cloud_search_web",
        "send_query_to_third_party",
    }


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


def test_load_scenarios_includes_11_agent_loop():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    agent_ids = {s.scenario_id for s in scenarios if s.session_kind == "agent_loop"}
    assert agent_ids == {
        "ag_001_input_leak",
        "ag_002_args_minimal",
        "ag_003_result_governance",
        "ag_004_prompt_injection",
        "ag_005_error_path",
        "ag_006_long_chain_late_leak",
        "ag_007_fallback_model_handoff",
        "ag_008_cloud_search_tool_leak",
        "ag_009_reverse_leak_pressure",
        "ag_010_search_result_third_party_injection",
        "ag_011_tool_description_third_party_injection",
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


def test_long_chain_and_fallback_scenarios_cover_late_leak_surfaces():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    ag_006 = next(s for s in scenarios if s.scenario_id == "ag_006_long_chain_late_leak")
    assert ag_006.max_steps == 6
    assert "write_public_report" in ag_006.tools_used
    assert "fingerprint_leak_free" in ag_006.tested_dimensions
    assert "tool_result_governance" in ag_006.tested_dimensions

    ag_007 = next(s for s in scenarios if s.scenario_id == "ag_007_fallback_model_handoff")
    assert "delegate_to_large_model" in ag_007.tools_used
    assert "tool_input_clean" in ag_007.tested_dimensions
    assert "tool_args_minimal" in ag_007.tested_dimensions


def test_cloud_search_and_reverse_scenarios_cover_all_tracks():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    mt_cloud = next(s for s in scenarios if s.scenario_id == "mt_005_cloud_search_safe")
    mt_reverse = next(s for s in scenarios if s.scenario_id == "mt_006_reverse_leak_pressure")
    ag_cloud = next(s for s in scenarios if s.scenario_id == "ag_008_cloud_search_tool_leak")
    ag_reverse = next(s for s in scenarios if s.scenario_id == "ag_009_reverse_leak_pressure")

    assert "cloud_search_clean" in mt_cloud.tested_dimensions
    assert "reverse_leak_resisted" in mt_reverse.tested_dimensions
    assert "cloud_search_web" in ag_cloud.tools_used
    assert "cloud_search_clean" in ag_cloud.tested_dimensions
    assert "reverse_leak_resisted" in ag_reverse.tested_dimensions


def test_third_party_search_injection_scenarios_present():
    scenarios = load_scenarios(REPO_ROOT / "config" / "scenarios.yaml")
    result_injection = next(
        s for s in scenarios
        if s.scenario_id == "ag_010_search_result_third_party_injection"
    )
    tool_injection = next(
        s for s in scenarios
        if s.scenario_id == "ag_011_tool_description_third_party_injection"
    )

    assert "send_query_to_third_party" in result_injection.tools_used
    assert "third_party_exfiltration_resisted" in result_injection.tested_dimensions
    assert "compromised_cloud_search_web" in tool_injection.tools_used
    assert "send_query_to_third_party" in tool_injection.tools_used
    assert "third_party_exfiltration_resisted" in tool_injection.tested_dimensions
