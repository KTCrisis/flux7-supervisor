"""Tests for the rule evaluator."""

import pytest

from sup7.config import EvaluatorConfig, RuleEntry, SupervisorConfig
from sup7.evaluator import RuleEvaluator
from sup7.models import ApprovalContext


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "test-1", "agent_id": "agent", "tool": "filesystem.read_file"}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


def _config(rules=None, **kwargs) -> SupervisorConfig:
    return SupervisorConfig(
        rules=rules or [],
        evaluator=EvaluatorConfig(provider="ollama", **kwargs),
    )


class TestRuleEvaluator:
    @pytest.mark.asyncio
    async def test_injection_risk_escalates(self):
        evaluator = RuleEvaluator(_config())
        ctx = _ctx(injection_risk=True)
        decision = await evaluator.evaluate(ctx)
        assert decision.decision == "escalated"
        assert decision.rule_matched == "injection-risk"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_rule_approve(self):
        rules = [RuleEntry(name="reads", condition="tool contains read", action="approve")]
        evaluator = RuleEvaluator(_config(rules=rules))
        decision = await evaluator.evaluate(_ctx(tool="filesystem.read_file"))
        assert decision.decision == "approved"
        assert decision.rule_matched == "reads"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_rule_deny(self):
        rules = [RuleEntry(name="no-delete", condition="tool contains delete", action="deny")]
        evaluator = RuleEvaluator(_config(rules=rules))
        decision = await evaluator.evaluate(_ctx(tool="filesystem.delete"))
        assert decision.decision == "denied"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_first_match_wins(self):
        rules = [
            RuleEntry(name="reads", condition="tool contains read", action="approve"),
            RuleEntry(name="all-fs", condition="tool contains filesystem", action="deny"),
        ]
        evaluator = RuleEvaluator(_config(rules=rules))
        decision = await evaluator.evaluate(_ctx(tool="filesystem.read_file"))
        assert decision.decision == "approved"
        assert decision.rule_matched == "reads"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_low_confidence_escalates(self):
        rules = [
            RuleEntry(name="weak", condition="tool contains read", action="approve", confidence=0.3),
        ]
        evaluator = RuleEvaluator(_config(rules=rules, confidence_threshold=0.8))
        decision = await evaluator.evaluate(_ctx(tool="filesystem.read_file"))
        assert decision.decision == "escalated"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_catch_all_without_llm_escalates(self):
        evaluator = RuleEvaluator(_config(rules=[]))
        decision = await evaluator.evaluate(_ctx(tool="unknown.tool"))
        assert decision.decision == "escalated"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_decision_fields(self):
        rules = [RuleEntry(name="reads", condition="tool contains read", action="approve")]
        evaluator = RuleEvaluator(_config(rules=rules))
        decision = await evaluator.evaluate(_ctx())
        assert decision.approval_id == "test-1"
        assert decision.agent_id == "agent"
        assert decision.tool == "filesystem.read_file"
        assert decision.evaluation_ms >= 0
        assert decision.timestamp is not None
        await evaluator.close()
