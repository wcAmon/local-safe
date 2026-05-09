"""Agent-loop driver — runs scenarios with tool calling against an adapter
and produces Trace artifacts."""

from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
from pipeline.schemas import (
    Sample, Scenario, Trace, Step, ExposureLedgerEntry,
    OutputMeta, ToolCall, ToolResult, ToolSpec,
)
from pipeline.config import ModelConfig
from pipeline.serving.base import ModelAdapter, Message
from pipeline.pii.matcher import PIIMatcher
from pipeline.pii.tokens import PIIKind
from pipeline.jsonl_io import read_jsonl, append_jsonl_idempotent
from pipeline.stages.stage1_dataset import MappingRow
from pipeline.runner.tool_executor import find_mock_return


def agent_trace_id_for(model_id: str, scenario_id: str, seed: int) -> str:
    """Hash includes '|agent' suffix to avoid collision with multi_turn trace_ids
    for the same scenario_id."""
    h = hashlib.sha256()
    h.update(f"{model_id}|{scenario_id}|{seed}|agent".encode("utf-8"))
    return h.hexdigest()[:16]


def _build_matcher_from_mapping(mapping_path: Path, salt: str) -> PIIMatcher:
    rows = list(read_jsonl(mapping_path, MappingRow))
    m = PIIMatcher(salt=salt)
    for r in rows:
        m.raw_to_token[r.raw] = r.token
        m.token_to_raw[r.token] = r.raw
        try:
            m.raw_to_kind[r.raw] = PIIKind(r.kind)
        except ValueError:
            m.raw_to_kind[r.raw] = PIIKind.LOCATION
    return m


def _existing_trace_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(json.loads(line)["trace_id"])
    return out


def _scan_tool_call_for_leaks(call: ToolCall, matcher: PIIMatcher) -> list[str]:
    leaked: set[str] = set()
    for v in call.arguments.values():
        if isinstance(v, str):
            leaked |= matcher.extract_known_refs(v)
    return sorted(leaked)


def run_agent_loop(
    *,
    adapter: ModelAdapter,
    model_cfg: ModelConfig,
    scenarios: list[Scenario],
    samples_by_id: dict[str, Sample],
    tool_specs: dict[str, ToolSpec],
    vault_dir: Path,
    artifacts_dir: Path,
    salt: str,
) -> int:
    """Run agent_loop scenarios for one adapter. Returns rows added."""
    seed = int(model_cfg.params.get("seed", 0))
    raw_path = vault_dir / "traces_raw.jsonl"
    red_path = artifacts_dir / "traces.jsonl"
    matcher = _build_matcher_from_mapping(vault_dir / "mapping.jsonl", salt=salt)

    existing = _existing_trace_ids(raw_path)
    raw_traces: list[Trace] = []
    redacted_traces: list[Trace] = []

    for scenario in scenarios:
        if scenario.session_kind != "agent_loop":
            continue
        tid = agent_trace_id_for(model_cfg.model_id, scenario.scenario_id, seed)
        if tid in existing:
            continue
        sample = samples_by_id.get(scenario.sample_id) if scenario.sample_id else None
        max_steps = scenario.max_steps or 6

        # Resolve tools
        scenario_tools = [tool_specs[name] for name in scenario.tools_used
                           if name in tool_specs]

        # Initial state
        chat: list[Message] = []
        steps_raw: list[Step] = []
        steps_red: list[Step] = []
        ledger: dict[str, ExposureLedgerEntry] = {}
        cumulative: set[str] = set()
        latency_total = 0
        tokens_in_total = 0
        tokens_out_total = 0
        last_finish = "stop"

        # Render initial user message
        initial = scenario.initial_prompt or ""
        if sample and "{content}" in initial:
            initial = initial.format(content=sample.content)
        elif sample is None and "{content}" in initial:
            initial = initial.replace("{content}", "")
        chat.append(Message(role="user", content=initial))
        new_refs = matcher.extract_known_refs(initial) - cumulative
        cumulative |= new_refs
        for ref in new_refs:
            ledger[ref] = ExposureLedgerEntry(introduced_step=len(steps_raw))
        in_step = Step(
            step=len(steps_raw), kind="input", subkind="user_message",
            introduced_pii_refs=sorted(new_refs),
            visible_pii_refs_so_far=sorted(cumulative),
            content_referenced=matcher.to_referenced(initial),
        )
        steps_raw.append(in_step)
        steps_red.append(in_step)

        # Loop up to max_steps model invocations
        for _iter in range(max_steps):
            resp = adapter.generate(
                chat, params=model_cfg.params, request_id=tid, tools=scenario_tools,
            )
            latency_total += resp.latency_ms
            tokens_in_total += resp.tokens_in
            tokens_out_total += resp.tokens_out
            last_finish = resp.finish_reason

            if resp.tool_calls:
                # Append assistant message with tool_calls (echo back via raw_meta)
                assistant_meta_calls = (resp.raw_meta or {}).get("openai_tool_calls") or \
                                         (resp.raw_meta or {}).get("anthropic_content")
                chat.append(Message(role="assistant", content=resp.content,
                                     tool_calls=assistant_meta_calls))

                for call in resp.tool_calls:
                    leaked = _scan_tool_call_for_leaks(call, matcher)
                    for ref in leaked:
                        if ref in ledger:
                            if ledger[ref].first_emitted_step is None:
                                ledger[ref].first_emitted_step = len(steps_raw)
                            ledger[ref].first_leaked_step = ledger[ref].first_leaked_step or len(steps_raw)
                            ledger[ref].leak_count += 1
                    # Redacted version of the tool_call: args go through PII matcher
                    redacted_args = {
                        k: matcher.to_referenced(v) if isinstance(v, str) else v
                        for k, v in call.arguments.items()
                    }
                    tc_call_red = ToolCall(tool_name=call.tool_name, arguments=redacted_args)
                    tc_step_raw = Step(
                        step=len(steps_raw), kind="output", subkind="tool_call",
                        visible_pii_refs_so_far=sorted(cumulative),
                        emitted_pii_refs=leaked,
                        leaked_pii_refs=leaked,
                        content_referenced="",
                        tool_call=call,
                    )
                    tc_step_red = Step(
                        step=len(steps_red), kind="output", subkind="tool_call",
                        visible_pii_refs_so_far=sorted(cumulative),
                        emitted_pii_refs=leaked,
                        leaked_pii_refs=leaked,
                        content_referenced="",
                        tool_call=tc_call_red,
                    )
                    steps_raw.append(tc_step_raw)
                    steps_red.append(tc_step_red)

                    result = find_mock_return(scenario.mock_returns, call.tool_name, call.arguments)
                    if result is None:
                        result = ToolResult(tool_name=call.tool_name,
                                             output="{tool returned no data}",
                                             is_error=False)
                    # Track new PII introduced by tool result
                    new_from_tool = matcher.extract_known_refs(result.output) - cumulative
                    cumulative |= new_from_tool
                    for ref in new_from_tool:
                        ledger[ref] = ExposureLedgerEntry(introduced_step=len(steps_raw))
                    tr_step_raw = Step(
                        step=len(steps_raw), kind="input", subkind="tool_result",
                        introduced_pii_refs=sorted(new_from_tool),
                        visible_pii_refs_so_far=sorted(cumulative),
                        content_referenced=result.output,    # raw goes to vault
                        tool_result=result,
                    )
                    result_red = ToolResult(
                        tool_name=result.tool_name,
                        output=matcher.to_referenced(result.output),
                        is_error=result.is_error,
                    )
                    tr_step_red = Step(
                        step=len(steps_red), kind="input", subkind="tool_result",
                        introduced_pii_refs=sorted(new_from_tool),
                        visible_pii_refs_so_far=sorted(cumulative),
                        content_referenced=matcher.to_referenced(result.output),
                        tool_result=result_red,
                    )
                    steps_raw.append(tr_step_raw)
                    steps_red.append(tr_step_red)
                    chat.append(Message(role="tool", content=result.output))
                continue

            # Terminal text response
            redacted_text, leaked = matcher.redact_output(resp.content, partial=True)
            for ref in leaked:
                if ref in ledger:
                    if ledger[ref].first_emitted_step is None:
                        ledger[ref].first_emitted_step = len(steps_raw)
                    ledger[ref].first_leaked_step = ledger[ref].first_leaked_step or len(steps_raw)
                    ledger[ref].leak_count += 1
            asst_raw = Step(
                step=len(steps_raw), kind="output", subkind="assistant_message",
                visible_pii_refs_so_far=sorted(cumulative),
                emitted_pii_refs=sorted(leaked),
                leaked_pii_refs=sorted(leaked),
                content_referenced=resp.content,
            )
            asst_red = Step(
                step=len(steps_red), kind="output", subkind="assistant_message",
                visible_pii_refs_so_far=sorted(cumulative),
                emitted_pii_refs=sorted(leaked),
                leaked_pii_refs=sorted(leaked),
                content_referenced=redacted_text,
            )
            steps_raw.append(asst_raw)
            steps_red.append(asst_red)
            chat.append(Message(role="assistant", content=resp.content))
            break

        meta = OutputMeta(
            latency_ms=latency_total, tokens_in=tokens_in_total,
            tokens_out=tokens_out_total, finish_reason=last_finish,
            ran_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        common = dict(
            trace_id=tid, session_kind="agent_loop",
            model_id=model_cfg.model_id, sample_id=scenario.sample_id,
            scenario_id=scenario.scenario_id,
            tested_dimensions=list(scenario.tested_dimensions),
            exposure_ledger=ledger, metadata=meta,
        )
        raw_traces.append(Trace(steps=steps_raw, **common))
        redacted_traces.append(Trace(steps=steps_red, **common))

    n_added = append_jsonl_idempotent(raw_path, raw_traces, key="trace_id")
    append_jsonl_idempotent(red_path, redacted_traces, key="trace_id")
    return n_added
