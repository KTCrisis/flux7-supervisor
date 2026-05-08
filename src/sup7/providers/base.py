"""Evaluator protocol — interface for LLM providers."""

from __future__ import annotations

from typing import Protocol

from sup7.models import ApprovalContext, Verdict


class Evaluator(Protocol):
    """All LLM providers implement this interface."""

    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        """Evaluate an approval. Returns None on failure (triggers escalation)."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
