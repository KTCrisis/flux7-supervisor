"""Tests for the MCP server tools."""

from sup7.config import EvaluatorConfig
from sup7.mcp_server import set_evaluator, sup7_pending, sup7_verdict
from sup7.models import Verdict
from sup7.providers.claude_code import ClaudeCodeEvaluator


class TestMCPTools:
    def test_pending_empty_without_evaluator(self):
        set_evaluator(None)
        assert sup7_pending() == []

    def test_pending_with_evaluator(self):
        ev = ClaudeCodeEvaluator(EvaluatorConfig(callback_timeout=60))
        set_evaluator(ev)
        assert sup7_pending() == []

    def test_verdict_without_evaluator(self):
        set_evaluator(None)
        result = sup7_verdict("x", "approve", 0.9, "test")
        assert "no evaluator" in result

    def test_verdict_invalid_action(self):
        ev = ClaudeCodeEvaluator(EvaluatorConfig(callback_timeout=60))
        set_evaluator(ev)
        result = sup7_verdict("x", "invalid", 0.9, "test")
        assert "invalid action" in result

    def test_verdict_unknown_id(self):
        ev = ClaudeCodeEvaluator(EvaluatorConfig(callback_timeout=60))
        set_evaluator(ev)
        result = sup7_verdict("unknown", "approve", 0.9, "test")
        assert "unknown" in result
