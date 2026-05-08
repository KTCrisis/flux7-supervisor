"""Tests for LLM providers."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sup7.config import EvaluatorConfig
from sup7.models import ApprovalContext, Verdict
from sup7.providers.claude_code import ClaudeCodeEvaluator
from sup7.providers.ollama import OllamaEvaluator


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "test-1", "agent_id": "agent", "tool": "filesystem.read_file"}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


def _eval_config(**kwargs) -> EvaluatorConfig:
    return EvaluatorConfig(**kwargs)


class TestOllamaParseResponse:
    def test_valid_response(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response(
            "DECISION: APPROVE | CONFIDENCE: 0.95 | REASONING: routine file read"
        )
        assert verdict is not None
        assert verdict.action == "approve"
        assert verdict.confidence == 0.95
        assert "routine" in verdict.reasoning

    def test_deny_response(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response(
            "DECISION: DENY | CONFIDENCE: 0.88 | REASONING: writes to system dir"
        )
        assert verdict is not None
        assert verdict.action == "deny"

    def test_escalate_response(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response(
            "DECISION: ESCALATE | CONFIDENCE: 0.4 | REASONING: unclear intent"
        )
        assert verdict is not None
        assert verdict.action == "escalate"

    def test_unparseable_approve_fallback(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response("I think we should approve this request.")
        assert verdict is not None
        assert verdict.action == "approve"
        assert verdict.confidence == 0.5

    def test_unparseable_deny_fallback(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response("This should be denied for safety.")
        assert verdict is not None
        assert verdict.action == "deny"

    def test_completely_unparseable(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response("Hello, how can I help you?")
        assert verdict is None

    def test_multiline_finds_decision(self):
        evaluator = OllamaEvaluator(_eval_config())
        verdict = evaluator._parse_response(
            "Let me think about this...\n"
            "DECISION: APPROVE | CONFIDENCE: 0.9 | REASONING: safe operation\n"
            "Hope that helps!"
        )
        assert verdict is not None
        assert verdict.action == "approve"


class TestClaudeCodeEvaluator:
    @pytest.mark.asyncio
    async def test_submit_verdict_resolves_future(self):
        evaluator = ClaudeCodeEvaluator(_eval_config(callback_timeout=5))
        ctx = _ctx()

        async def submit_after_delay():
            await asyncio.sleep(0.1)
            evaluator.submit_verdict("test-1", Verdict("approve", 0.95, "safe"))

        asyncio.create_task(submit_after_delay())
        verdict = await evaluator.evaluate(ctx)

        assert verdict is not None
        assert verdict.action == "approve"
        assert verdict.confidence == 0.95
        await evaluator.close()

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        evaluator = ClaudeCodeEvaluator(_eval_config(callback_timeout=0.1))
        verdict = await evaluator.evaluate(_ctx())
        assert verdict is None
        await evaluator.close()

    def test_list_pending(self):
        evaluator = ClaudeCodeEvaluator(_eval_config(callback_timeout=60))
        assert evaluator.list_pending() == []

    def test_submit_unknown_id_returns_false(self):
        evaluator = ClaudeCodeEvaluator(_eval_config(callback_timeout=60))
        ok = evaluator.submit_verdict("unknown", Verdict("approve", 0.9, "test"))
        assert not ok


class TestProviderFactory:
    def test_create_ollama(self):
        from sup7.providers import create_evaluator

        ev = create_evaluator(_eval_config(provider="ollama"))
        assert isinstance(ev, OllamaEvaluator)

    def test_create_claude_code(self):
        from sup7.providers import create_evaluator

        ev = create_evaluator(_eval_config(provider="claude-code"))
        assert isinstance(ev, ClaudeCodeEvaluator)

    def test_create_unknown_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _eval_config(provider="gpt-4")
