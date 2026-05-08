"""Anthropic Claude API provider — cloud LLM evaluation."""

from __future__ import annotations

import json
import logging
import re

from sup7.config import EvaluatorConfig
from sup7.models import ApprovalContext, Verdict

logger = logging.getLogger(__name__)

_RESPONSE_RE = re.compile(
    r"DECISION:\s*(APPROVE|DENY|ESCALATE)\s*\|\s*"
    r"CONFIDENCE:\s*([\d.]+)\s*\|\s*"
    r"REASONING:\s*(.+)",
    re.IGNORECASE,
)


class AnthropicEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install 'flux7-supervisor[anthropic]'"
            )
        self._config = config
        self._client = anthropic.AsyncAnthropic()

    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        context = {
            "tool": approval.tool,
            "agent_id": approval.agent_id,
            "params": approval.params,
            "policy_rule": approval.policy_rule,
            "injection_risk": approval.injection_risk,
            "recent_traces": approval.recent_traces[:5],
            "active_grants": approval.active_grants,
        }
        prompt = f"Evaluate this pending approval request:\n\n```json\n{json.dumps(context, indent=2)}\n```"

        try:
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=256,
                system=self._config.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return None

        raw = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw += block.text

        return self._parse_response(raw)

    async def close(self) -> None:
        pass

    def _parse_response(self, raw: str) -> Verdict | None:
        for line in raw.strip().splitlines():
            match = _RESPONSE_RE.search(line)
            if match:
                action = match.group(1).lower()
                try:
                    confidence = max(0.0, min(1.0, float(match.group(2))))
                except ValueError:
                    confidence = 0.5
                return Verdict(action=action, confidence=confidence, reasoning=match.group(3).strip())

        logger.warning("could not parse Anthropic response: %s", raw[:200])
        return None
