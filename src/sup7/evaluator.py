"""Rule-based approval evaluator with pluggable LLM fallback."""

from __future__ import annotations

import logging
import time

from sup7.config import RuleEntry, SupervisorConfig
from sup7.models import ApprovalContext, Decision, Verdict
from sup7.providers import Evaluator, create_evaluator
from sup7.rules import Predicate, parse_condition

logger = logging.getLogger(__name__)


class RuleEvaluator:
    """Evaluates approvals against a rule chain with optional LLM fallback.

    Flow:
    1. injection_risk → escalate (fast path)
    2. Rules evaluated in order, first match wins
    3. If catch-all matches and LLM is configured → LLM evaluation
    4. If LLM fails or is disabled → escalate to human
    """

    def __init__(self, config: SupervisorConfig) -> None:
        self._config = config
        self._compiled: list[tuple[RuleEntry, Predicate | None]] = []
        for rule in config.rules:
            pred = parse_condition(rule.condition) if rule.condition else None
            self._compiled.append((rule, pred))

        self._llm: Evaluator | None = None
        if config.evaluator.provider:
            self._llm = create_evaluator(config.evaluator)
            logger.info(
                "LLM evaluation enabled — provider=%s model=%s",
                config.evaluator.provider,
                config.evaluator.model,
            )

    async def evaluate(self, approval: ApprovalContext) -> Decision:
        start = time.monotonic()

        if approval.injection_risk:
            return self._decision(
                approval, start,
                decision="escalated",
                rule_matched="injection-risk",
                reasoning="injection risk detected by flux7-mesh",
                confidence=1.0,
            )

        for rule, predicate in self._compiled:
            if predicate is None or predicate(approval, self._config.project_dirs):
                if predicate is None and rule.action == "escalate" and self._llm:
                    return await self._evaluate_with_llm(approval, start)

                action = rule.action
                confidence = rule.confidence

                if action != "escalate" and confidence < self._config.evaluator.confidence_threshold:
                    action = "escalate"

                decision_str = {"approve": "approved", "deny": "denied", "escalate": "escalated"}
                return self._decision(
                    approval, start,
                    decision=decision_str.get(action, "escalated"),
                    rule_matched=rule.name,
                    reasoning=self._build_reasoning(rule, approval),
                    confidence=confidence,
                )

        return self._decision(
            approval, start,
            decision="escalated",
            rule_matched=None,
            reasoning="no rule matched",
            confidence=0.0,
        )

    async def _evaluate_with_llm(self, approval: ApprovalContext, start: float) -> Decision:
        assert self._llm is not None
        verdict = await self._llm.evaluate(approval)

        if verdict is None:
            return self._decision(
                approval, start,
                decision="escalated",
                rule_matched=f"{self._config.evaluator.provider}-fallback",
                reasoning="LLM evaluation failed, escalating to human",
                confidence=0.0,
            )

        decision_str = {"approve": "approved", "deny": "denied", "escalate": "escalated"}
        action = decision_str.get(verdict.action, "escalated")

        if action != "escalated" and verdict.confidence < self._config.evaluator.confidence_threshold:
            action = "escalated"
            reasoning = (
                f"LLM confidence {verdict.confidence:.2f} below threshold "
                f"{self._config.evaluator.confidence_threshold}: {verdict.reasoning}"
            )
        else:
            reasoning = verdict.reasoning

        return self._decision(
            approval, start,
            decision=action,
            rule_matched=f"{self._config.evaluator.provider}:{self._config.evaluator.model}",
            reasoning=reasoning,
            confidence=verdict.confidence,
        )

    async def close(self) -> None:
        if self._llm:
            await self._llm.close()

    def _decision(
        self, approval: ApprovalContext, start: float, *,
        decision: str, rule_matched: str | None, reasoning: str, confidence: float,
    ) -> Decision:
        return Decision(
            timestamp=Decision.now(),
            approval_id=approval.id,
            agent_id=approval.agent_id,
            tool=approval.tool,
            decision=decision,
            rule_matched=rule_matched,
            reasoning=reasoning,
            confidence=confidence,
            evaluation_ms=int((time.monotonic() - start) * 1000),
            injection_risk=approval.injection_risk,
        )

    def _build_reasoning(self, rule: RuleEntry, approval: ApprovalContext) -> str:
        parts = [f"rule '{rule.name}' matched"]
        if rule.condition and "params.path" in rule.condition:
            path = approval.params.get("path")
            if path:
                parts.append(f"path={path}")
        if rule.description:
            parts.append(rule.description)
        return "; ".join(parts)
