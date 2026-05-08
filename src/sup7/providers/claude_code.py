"""Claude Code MCP callback provider — async evaluation via MCP tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sup7.config import EvaluatorConfig
from sup7.models import ApprovalContext, Verdict

logger = logging.getLogger(__name__)


class ClaudeCodeEvaluator:
    """Queues evaluations and waits for Claude Code to respond via MCP callback.

    Exposes pending evaluations via sup7.pending MCP tool.
    Receives verdicts via sup7.verdict MCP tool.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self._timeout = config.callback_timeout
        self._pending: dict[str, asyncio.Future[Verdict]] = {}
        self._pending_context: dict[str, dict[str, Any]] = {}

    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Verdict] = loop.create_future()
        self._pending[approval.id] = future
        self._pending_context[approval.id] = {
            "id": approval.id,
            "agent_id": approval.agent_id,
            "tool": approval.tool,
            "params": approval.params,
            "policy_rule": approval.policy_rule,
            "injection_risk": approval.injection_risk,
            "recent_traces": approval.recent_traces[:5],
            "active_grants": approval.active_grants,
        }
        logger.info("queued evaluation %s for Claude Code (timeout=%.0fs)", approval.id, self._timeout)

        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.warning("Claude Code evaluation timed out for %s", approval.id)
            return None
        finally:
            self._pending.pop(approval.id, None)
            self._pending_context.pop(approval.id, None)

    def submit_verdict(self, approval_id: str, verdict: Verdict) -> bool:
        """Called by MCP server when Claude Code responds via sup7.verdict."""
        future = self._pending.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(verdict)
        logger.info("received verdict for %s: %s (%.2f)", approval_id, verdict.action, verdict.confidence)
        return True

    def list_pending(self) -> list[dict[str, Any]]:
        """Return all pending evaluations for the MCP tool."""
        return list(self._pending_context.values())

    async def close(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._pending_context.clear()
