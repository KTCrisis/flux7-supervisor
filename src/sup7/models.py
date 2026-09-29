"""Core data models for supervisor decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple


class Verdict(NamedTuple):
    """LLM or rule evaluation result."""

    action: str  # "approve" | "deny" | "escalate"
    confidence: float  # 0.0-1.0
    reasoning: str
    source: str = ""  # who answered, for the audit label; set by the provider chain
    meta: dict | None = None  # provenance from the provider: model version, question set, thresholds
    raw: dict | None = None  # the provider's raw answers, kept by the bench to recompute offline


@dataclass
class Decision:
    """Full decision record, written to JSONL and mem7."""

    timestamp: datetime
    approval_id: str
    agent_id: str
    tool: str
    decision: str  # "approved" | "denied" | "escalated"
    rule_matched: str | None
    reasoning: str
    confidence: float
    evaluation_ms: int
    injection_risk: bool = False
    evaluator: dict | None = None  # provenance of an LLM verdict (Verdict.meta)
    via: str = "mesh"  # "mesh": an approval polled from flux7-mesh; "evaluate": POST /evaluate

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class RuleConfig:
    """A single evaluation rule in the supervisor rule chain."""

    name: str
    condition: str | None = None  # None = catch-all
    action: str = "escalate"  # "approve" | "deny" | "escalate"
    confidence: float = 0.9
    description: str | None = None


@dataclass
class ApprovalContext:
    """Approval detail with surrounding context, for LLM evaluation."""

    id: str
    agent_id: str
    tool: str
    params: dict = field(default_factory=dict)
    policy_rule: str = ""
    injection_risk: bool = False
    recent_traces: list[dict] = field(default_factory=list)
    active_grants: list[dict] = field(default_factory=list)
    project_dirs: list[str] = field(default_factory=list)  # set by the evaluator from the config
