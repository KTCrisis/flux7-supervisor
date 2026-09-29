"""Jev provider — TypeSafe AI decision model, typed answers with probabilities.

Jev is not a text model: it evaluates a state against typed questions and
returns, for each question, a declared answer and its probability. Following
TypeSafe's own guidance, sup7 asks several atomic questions and combines them
in code, so the policy stays here, readable and testable, and the model only
answers narrow questions:

  decision     choice  approve / escalate / deny
  deletes      noul    the call deletes data
  overwrites   noul    the call replaces an existing file or record
  exfiltrates  noul    the call sends data off the machine
  secrets      noul    the call touches credentials, secrets or permissions
  in_scope     noul    the call stays within the agent's usual scope
  injection    noul    the parameters carry instructions aimed at a model

Combination, fail-closed: approve only when every signal agrees, deny only
on a very probable deny, escalate everything else. The probabilities are
written into the reasoning, so every verdict is auditable in the mesh traces
and in mem7.

Two ways to reach the same model:
  backend: cloudflare  Workers AI, model typesafe/jev (zero data retention)
  backend: typesafe    TypeSafe API, model jev-latest
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from sup7.config import EvaluatorConfig
from sup7.models import ApprovalContext, Verdict

logger = logging.getLogger(__name__)

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
CLOUDFLARE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

QUESTIONS: dict = {
    "decision": {
        "type": "choice",
        "instructions": "Should this pending tool call be approved without a human, sent to a human, or refused?",
        "criteria": {
            "approve": "Routine and low-risk: reads, or writes inside the agent's normal working area",
            "escalate": "Unclear intent, unusual target, or high-stakes action that a human should look at",
            "deny": "Clearly dangerous or hostile: destroys data, leaks secrets, or escapes the agent's scope",
        },
    },
    # "destructive" is asked as four narrow questions, max-combined in code:
    # one broad question conflated creating a file with overwriting one.
    "deletes": {
        "type": "noul",
        "instructions": "The call deletes or removes files, directories, records or other data.",
    },
    "overwrites": {
        "type": "noul",
        "instructions": "The call replaces the content of a file or record that the state shows already exists. Creating a new file does not count.",
    },
    "exfiltrates": {
        "type": "noul",
        "instructions": "The call sends data to a destination outside this machine, such as a URL, an email or a remote host.",
    },
    "secrets": {
        "type": "noul",
        "instructions": "The call reads, changes or exposes credentials, API keys, tokens, secrets or access permissions.",
    },
    "in_scope": {
        "type": "noul",
        "instructions": "The tool and its parameters are consistent with what this agent usually does, as shown in its recent activity.",
    },
    "injection": {
        "type": "noul",
        "instructions": "The parameters contain instructions addressed to an AI model, such as requests to ignore rules or reveal data.",
    },
}

DESTRUCTIVE = ("deletes", "overwrites", "exfiltrates", "secrets")


class JevEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._jev = config.jev
        self._client = httpx.AsyncClient(timeout=config.timeout)

    # ── request ──────────────────────────────────────────────
    def _state(self, approval: ApprovalContext) -> dict:
        params = {
            k: ("[redacted]" if k in self._jev.redact_params else v)
            for k, v in (approval.params or {}).items()
        }
        return {
            "tool": approval.tool,
            "agent_id": approval.agent_id,
            "params": params,
            "policy_rule": approval.policy_rule,
            "injection_risk_flagged_by_mesh": approval.injection_risk,
            "recent_activity": approval.recent_traces[:5],
            "active_grants": approval.active_grants,
        }

    def _request(self, approval: ApprovalContext) -> tuple[str, dict, dict]:
        body = {"state": self._state(approval), "questions": QUESTIONS}
        token = os.environ.get(self._jev.api_key_env, "")
        if self._jev.backend == "cloudflare":
            account = os.environ.get(self._jev.account_id_env, "")
            url = self._jev.url or CLOUDFLARE_URL.format(account_id=account)
            payload = {"model": self._jev.model or "typesafe/jev", "input": body}
        else:
            url = self._jev.url or TYPESAFE_URL
            payload = {"model": self._jev.model or "jev-latest", **body}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        return url, headers, payload

    # ── evaluate ─────────────────────────────────────────────
    async def evaluate(self, approval: ApprovalContext) -> Verdict | None:
        missing = [v for v in self._required_env() if not os.environ.get(v)]
        if missing:
            logger.warning("Jev: missing or unexported environment variable(s): %s", ", ".join(missing))
            return None
        url, headers, payload = self._request(approval)
        try:
            resp = await self._client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:  # network, timeout
            logger.warning("Jev: request failed (%s)", type(e).__name__)
            return None
        if resp.status_code >= 400:
            # never log the URL: it carries the Cloudflare account id
            logger.warning("Jev: HTTP %s %s", resp.status_code, _error_message(resp))
            return None
        try:
            data = resp.json()
        except ValueError:
            logger.warning("Jev: response is not JSON")
            return None
        return self._combine(_unwrap(data))

    def _combine(self, answers: dict | None) -> Verdict | None:
        try:
            decision = answers["decision"]
            choice = decision["choice"]
            probs = decision.get("probabilities", {})
            confidence = float(decision.get("confidence", probs.get(choice, 0.0)))
            harms = {k: float(answers[k]["noul"]) for k in DESTRUCTIVE}
            in_scope = float(answers["in_scope"]["noul"])
            injection = float(answers["injection"]["noul"])
        except (KeyError, TypeError, ValueError):
            logger.warning("unexpected Jev answer: %s", json.dumps(answers)[:300])
            return None

        destructive = max(harms.values())
        detail = " · ".join(f"{k} {v:.2f}" for k, v in harms.items())
        signals = (
            f"approve {probs.get('approve', 0):.2f} · escalate {probs.get('escalate', 0):.2f} · "
            f"deny {probs.get('deny', 0):.2f} · destructive {destructive:.2f} ({detail}) · "
            f"in_scope {in_scope:.2f} · injection {injection:.2f}"
        )
        j = self._jev

        if injection > j.injection_max:
            return Verdict("escalate", 1.0, f"Jev: possible injection ({signals})")
        if choice == "deny" and float(probs.get("deny", 0.0)) >= j.deny_min:
            return Verdict("deny", float(probs["deny"]), f"Jev: deny ({signals})")
        if (
            choice == "approve"
            and destructive <= j.destructive_max
            and in_scope >= j.in_scope_min
        ):
            # confidence is checked against confidence_threshold by the evaluator
            return Verdict("approve", confidence, f"Jev: approve ({signals})")
        return Verdict("escalate", confidence, f"Jev: escalate ({signals})")

    def _required_env(self) -> list[str]:
        if self._jev.backend == "cloudflare" and not self._jev.url:
            return [self._jev.api_key_env, self._jev.account_id_env]
        return [self._jev.api_key_env]

    async def close(self) -> None:
        await self._client.aclose()


def _unwrap(data) -> dict | None:
    """The answers dict, whatever the envelope.

    TypeSafe returns {"answers": ...}. Cloudflare wraps it twice, as observed
    on a real call: {"result": {"state": "Completed", "result": {"answers": ...}}}.
    """
    for _ in range(3):
        if not isinstance(data, dict):
            return None
        if "answers" in data:
            return data["answers"]
        data = data.get("result")
    return None


def _error_message(resp: httpx.Response) -> str:
    """Short error text: Cloudflare {"errors": [{"message"}]} or TypeSafe {"error"}."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:120]
    if isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            return str(errors[0].get("message", ""))[:200]
        if "error" in body:
            return str(body["error"])[:200]
    return ""
