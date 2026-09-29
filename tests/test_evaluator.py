"""Tests for the rule evaluator."""

import pytest

from sup7.config import EvaluatorConfig, RuleEntry, SupervisorConfig
from sup7.evaluator import RuleEvaluator
from sup7.models import ApprovalContext


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "test-1", "agent_id": "agent", "tool": "filesystem.read_file"}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


def _config(rules=None, provider="none", **kwargs) -> SupervisorConfig:
    # rules only by default: no test may reach a real LLM (it loaded qwen3:14b on the GPU)
    return SupervisorConfig(
        rules=rules or [],
        evaluator=EvaluatorConfig(provider=provider, **kwargs),
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
        assert evaluator._llm is None
        decision = await evaluator.evaluate(_ctx(tool="unknown.tool"))
        assert decision.decision == "escalated"
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_provider_none_is_ignored_when_a_chain_is_set(self):
        config = SupervisorConfig(evaluator=EvaluatorConfig(
            provider="none", chain=[EvaluatorConfig(provider="jev")]))
        evaluator = RuleEvaluator(config)
        assert evaluator._llm is not None
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


class _Stub:
    def __init__(self, verdict):
        self.verdict = verdict

    async def evaluate(self, approval):
        return self.verdict

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_chain_decision_labelled_with_the_provider_that_answered():
    # regression: a Jev answer in a jev > ollama chain was recorded as "ollama:qwen3:14b"
    from sup7.models import Verdict
    from sup7.providers.chain import ChainEvaluator

    config = SupervisorConfig(evaluator=EvaluatorConfig(
        chain=[EvaluatorConfig(provider="jev"), EvaluatorConfig(provider="ollama")]))
    evaluator = RuleEvaluator(config)
    evaluator._llm = ChainEvaluator(
        [("jev", _Stub(Verdict("escalate", 0.4, "Jev: escalate"))), ("ollama", _Stub(None))],
        labels=["jev:cloudflare", "ollama:qwen3:14b"])
    decision = await evaluator.evaluate(_ctx(tool="unknown.tool"))
    assert decision.rule_matched == "jev:cloudflare"
    assert decision.reasoning.startswith("[jev]")


@pytest.mark.parametrize("config, label", [
    (EvaluatorConfig(provider="jev"), "jev:cloudflare"),
    (EvaluatorConfig(provider="ollama", model="qwen3:14b"), "ollama:qwen3:14b"),
    (EvaluatorConfig(chain=[EvaluatorConfig(provider="jev")]), "chain"),
])
def test_provider_label(config, label):
    from sup7.providers import provider_label
    assert provider_label(config) == label
