"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Evaluator, Verdict

if TYPE_CHECKING:
    from sup7.config import EvaluatorConfig

__all__ = ["Evaluator", "Verdict", "create_evaluator", "provider_label"]


def provider_label(config: EvaluatorConfig) -> str:
    """Audit label of one provider: what a decision's rule_matched records."""
    if config.chain:
        return "chain"
    if config.provider == "jev":
        return f"jev:{config.jev.model or config.jev.backend}"
    return f"{config.provider}:{config.model}"


def create_evaluator(config: EvaluatorConfig) -> Evaluator:
    """Factory: instantiate the configured LLM provider."""
    if config.chain:
        from .chain import ChainEvaluator

        return ChainEvaluator(
            [(c.provider, create_evaluator(c)) for c in config.chain],
            labels=[provider_label(c) for c in config.chain],
            failures=config.breaker_failures,
            cooldown=config.breaker_cooldown,
        )
    if config.provider == "ollama":
        from .ollama import OllamaEvaluator

        return OllamaEvaluator(config)
    elif config.provider == "anthropic":
        from .anthropic import AnthropicEvaluator

        return AnthropicEvaluator(config)
    elif config.provider == "jev":
        from .jev import JevEvaluator

        return JevEvaluator(config)
    elif config.provider == "claude-code":
        from .claude_code import ClaudeCodeEvaluator

        return ClaudeCodeEvaluator(config)
    else:
        raise ValueError(f"unknown evaluator provider: {config.provider!r}")
