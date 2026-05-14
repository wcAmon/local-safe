"""Budget tracking and stop_and_report guard for paid LLM judges."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml


class BudgetExceeded(Exception):
    """Raised on demand by callers; not raised automatically."""


class BudgetGuard:
    """Track per-judge and total spend, persist to cost.jsonl, halt politely."""

    def __init__(
        self,
        *,
        total_cap: float,
        per_judge_caps: dict[str, float],
        cost_log_path: Path,
        on_exceed: str = "stop_and_report",
    ):
        self.total_cap = total_cap
        self.per_judge_caps = per_judge_caps
        self.cost_log_path = cost_log_path
        self.on_exceed = on_exceed
        self._spent_total = 0.0
        self._spent_per_judge: dict[str, float] = {}
        # Recover state if a prior run wrote any rows.
        if cost_log_path.exists():
            for line in cost_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._spent_total += row["cost_usd"]
                jid = row["judge_id"]
                self._spent_per_judge[jid] = self._spent_per_judge.get(jid, 0.0) + row["cost_usd"]

    @classmethod
    def from_config(cls, budget_yaml_path: Path, cost_log_path: Path) -> "BudgetGuard":
        cfg = yaml.safe_load(budget_yaml_path.read_text(encoding="utf-8"))
        # per_judge_cap and per_model_cap share the same key space (actor_id);
        # the YAML split is organisational — judges vs under_test models.
        caps: dict[str, float] = {}
        for k, v in (cfg.get("per_judge_cap") or {}).items():
            caps[k] = float(v)
        for k, v in (cfg.get("per_model_cap") or {}).items():
            caps[k] = float(v)
        return cls(
            total_cap=float(cfg.get("total_usd_cap", 0.0)),
            per_judge_caps=caps,
            cost_log_path=cost_log_path,
            on_exceed=cfg.get("on_exceed", "stop_and_report"),
        )

    def check_before_call(self, judge_id: str) -> bool:
        if self._spent_total >= self.total_cap:
            return False
        cap = self.per_judge_caps.get(judge_id)
        if cap is not None and self._spent_per_judge.get(judge_id, 0.0) >= cap:
            return False
        return True

    def record(self, *, judge_id: str, cost_usd: float, **meta: Any) -> None:
        self._spent_total += cost_usd
        self._spent_per_judge[judge_id] = self._spent_per_judge.get(judge_id, 0.0) + cost_usd
        row = {"judge_id": judge_id, "cost_usd": cost_usd, **meta}
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cost_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
