from pipeline.schemas import MockReturn, ToolResult
from pipeline.runner.tool_executor import find_mock_return


def test_finds_exact_match():
    mocks = {
        "fetch_user_history": [
            MockReturn(args={"user_id": "user_001"}, output="ok"),
            MockReturn(args={"user_id": "alice_92"}, output="ERROR", is_error=True),
            MockReturn(output="default"),
        ]
    }
    r = find_mock_return(mocks, "fetch_user_history", {"user_id": "user_001"})
    assert isinstance(r, ToolResult)
    assert r.output == "ok"
    assert r.is_error is False


def test_finds_error_match():
    mocks = {
        "fetch_user_history": [
            MockReturn(args={"user_id": "user_001"}, output="ok"),
            MockReturn(args={"user_id": "alice_92"}, output="ERROR", is_error=True),
            MockReturn(output="default"),
        ]
    }
    r = find_mock_return(mocks, "fetch_user_history", {"user_id": "alice_92"})
    assert r.output == "ERROR"
    assert r.is_error is True


def test_falls_back_to_default():
    mocks = {
        "fetch_user_history": [
            MockReturn(args={"user_id": "user_001"}, output="ok"),
            MockReturn(output="default"),
        ]
    }
    r = find_mock_return(mocks, "fetch_user_history", {"user_id": "U-7f3a"})
    assert r.output == "default"


def test_returns_none_when_no_default():
    mocks = {
        "fetch_user_history": [
            MockReturn(args={"user_id": "user_001"}, output="ok"),
        ]
    }
    r = find_mock_return(mocks, "fetch_user_history", {"user_id": "alice_92"})
    assert r is None


def test_returns_none_when_tool_not_in_mocks():
    mocks = {
        "fetch_user_history": [MockReturn(output="default")]
    }
    r = find_mock_return(mocks, "send_notification", {"user_id": "user_001", "message": "hi"})
    assert r is None


def test_args_subset_match():
    """If MockReturn.args is a strict subset of called_args (extra fields in
    called_args), it should still match."""
    mocks = {
        "fetch_user_history": [
            MockReturn(args={"user_id": "user_001"}, output="matched user_001"),
        ]
    }
    r = find_mock_return(mocks, "fetch_user_history",
                          {"user_id": "user_001", "since": "2026-01-01"})
    assert r is not None
    assert r.output == "matched user_001"
