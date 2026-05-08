"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Evaluator, Verdict

if TYPE_CHECKING:
    from sup7.config import EvaluatorConfig

__all__ = ["Evaluator", "Verdict", "create_evaluator"]


def create_evaluator(config: EvaluatorConfig) -> Evaluator:
    """Factory: instantiate the configured LLM provider."""
    if config.provider == "ollama":
        from .ollama import OllamaEvaluator

        return OllamaEvaluator(config)
    elif config.provider == "anthropic":
        from .anthropic import AnthropicEvaluator

        return AnthropicEvaluator(config)
    elif config.provider == "claude-code":
        from .claude_code import ClaudeCodeEvaluator

        return ClaudeCodeEvaluator(config)
    else:
        raise ValueError(f"unknown evaluator provider: {config.provider!r}")
