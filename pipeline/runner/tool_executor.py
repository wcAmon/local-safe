"""Tool executor — match tool_calls against scenario mock_returns."""

from __future__ import annotations
from typing import Any
from pipeline.schemas import MockReturn, ToolResult


def find_mock_return(
    mock_returns: dict[str, list[MockReturn]],
    tool_name: str,
    called_args: dict[str, Any],
) -> ToolResult | None:
    """Return the first matching mock_return as a ToolResult.

    Match rule: an entry's `args` must be a strict subset of `called_args`,
    with each declared key/value matching exactly. Extra keys in `called_args`
    (not in entry.args) are allowed. An empty `entry.args` (default {}) matches
    any call to this tool — typically used as the fallback last entry.
    Returns None if no entry matches OR the tool isn't in `mock_returns`.
    """
    entries = mock_returns.get(tool_name)
    if not entries:
        return None
    for entry in entries:
        if all(called_args.get(k) == v for k, v in entry.args.items()):
            return ToolResult(
                tool_name=tool_name,
                output=entry.output,
                is_error=entry.is_error,
            )
    return None
