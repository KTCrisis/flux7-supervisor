"""FastMCP server exposing sup7.pending + sup7.verdict tools."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from sup7.models import Verdict

logger = logging.getLogger(__name__)

mcp = FastMCP("sup7")

_evaluator = None


def set_evaluator(evaluator: Any) -> None:
    """Bind the ClaudeCodeEvaluator instance to the MCP server."""
    global _evaluator
    _evaluator = evaluator


@mcp.tool()
def sup7_pending() -> list[dict]:
    """List evaluations waiting for Claude Code review.

    Returns a list of approval contexts that need evaluation.
    Each entry contains: id, agent_id, tool, params, policy_rule,
    injection_risk, recent_traces, active_grants.
    """
    if _evaluator is None:
        return []
    return _evaluator.list_pending()


@mcp.tool()
def sup7_verdict(
    approval_id: str,
    action: str,
    confidence: float,
    reasoning: str,
) -> str:
    """Submit evaluation verdict for a pending supervisor decision.

    Args:
        approval_id: The approval ID from sup7_pending
        action: "approve", "deny", or "escalate"
        confidence: 0.0-1.0 confidence score
        reasoning: One sentence explanation
    """
    if _evaluator is None:
        return "error: no evaluator configured"

    if action not in ("approve", "deny", "escalate"):
        return f"error: invalid action {action!r}, expected approve/deny/escalate"

    verdict = Verdict(action=action, confidence=confidence, reasoning=reasoning)
    ok = _evaluator.submit_verdict(approval_id, verdict)
    return "accepted" if ok else "error: unknown or expired approval_id"
