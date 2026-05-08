"""Ollama LLM provider — local HTTP evaluation."""

from __future__ import annotations

import json
import logging
import re

import httpx

from sup7.config import EvaluatorConfig
from sup7.models import ApprovalContext, Verdict

logger = logging.getLogger(__name__)

_RESPONSE_RE = re.compile(
    r"DECISION:\s*(APPROVE|DENY|ESCALATE)\s*\|\s*"
    r"CONFIDENCE:\s*([\d.]+)\s*\|\s*"
    r"REASONING:\s*(.+)",
    re.IGNORECASE,
)


class OllamaEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.url.rstrip("/"),
            timeout=config.timeout,
        )

    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        prompt = self._build_prompt(approval)
        try:
            resp = await self._client.post("/api/generate", json={
                "model": self._config.model,
                "prompt": prompt,
                "system": self._config.system_prompt,
                "stream": False,
            })
        except (httpx.ConnectError, httpx.ConnectTimeout):
            logger.warning("cannot connect to Ollama at %s", self._config.url)
            return None

        if resp.status_code != 200:
            logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
            return None

        raw = resp.json().get("response", "")
        return self._parse_response(raw)

    async def close(self) -> None:
        await self._client.aclose()

    def _build_prompt(self, approval: ApprovalContext) -> str:
        context = {
            "tool": approval.tool,
            "agent_id": approval.agent_id,
            "params": approval.params,
            "policy_rule": approval.policy_rule,
            "injection_risk": approval.injection_risk,
            "recent_traces": approval.recent_traces[:5],
            "active_grants": approval.active_grants,
        }
        return f"Evaluate this pending approval request:\n\n```json\n{json.dumps(context, indent=2)}\n```"

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

        raw_lower = raw.lower()
        has_approve = "approve" in raw_lower
        has_deny = "deny" in raw_lower or "denied" in raw_lower
        if has_approve and not has_deny:
            return Verdict("approve", 0.5, f"LLM suggested approve (unparsed): {raw[:100]}")
        if has_deny and not has_approve:
            return Verdict("deny", 0.5, f"LLM suggested deny (unparsed): {raw[:100]}")

        logger.warning("could not parse Ollama response: %s", raw[:200])
        return None
