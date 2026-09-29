"""Jev provider — TypeSafe AI decision model, typed answers with probabilities.

Jev is not a text model: it evaluates a state against typed questions and
returns, for each question, a declared answer and its probability. Following
TypeSafe's own guidance, sup7 asks narrow factual questions and decides in
code, so the policy stays here, readable and testable, and the model only
reports what the call does.

The questions live in YAML (sup7/questions.py): the shipped socle
(data/socle.yaml: deletes, overwrites, exfiltrates, secrets, target_zone,
in_scope, injection) plus the business packs listed in evaluator.jev.questions,
each applying to some agents or tools. Decision, fail-closed, from the groups:
  a manipulation answer above its threshold (injection_max)     escalate
  a counted danger >= deny_min and scope < deny_in_scope_max     deny
  every counted danger <= its threshold (destructive_max) and
      scope >= in_scope_min                                      approve, with
      confidence min(1 - highest danger, 1 - highest manipulation)
  anything else                                                  escalate
A danger with ignore_when does not count when the named choice option is
probable enough (overwrites inside the project, P(project) >= project_min).
A broad approve/escalate/deny choice was asked until 2026-09-29: on real calls
it stayed soft (0.61-0.79) where the narrow questions answered 0.99.
The probabilities are written into the reasoning, so every verdict is
auditable in the mesh traces and in mem7.

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
from sup7.questions import Selection, load_packs, select

logger = logging.getLogger(__name__)

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
CLOUDFLARE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"

# The shipped socle, for callers that need the default set (tests, scripts).
_SOCLE = select(load_packs([]), "", "")
QUESTIONS: dict = _SOCLE.wire()
QUESTIONS_SHA = _SOCLE.sha
DESTRUCTIVE = tuple(q.name for q in _SOCLE.questions if q.group == "danger")


class JevEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._jev = config.jev
        self._client: httpx.AsyncClient | None = None  # created on first call: a bench recompute never needs it
        # loaded once: a question set that fails validation stops sup7 at start
        self._packs = load_packs(config.jev.questions)

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
            "project_dirs": approval.project_dirs,
            "policy_rule": approval.policy_rule,
            "injection_risk_flagged_by_mesh": approval.injection_risk,
            "recent_activity": approval.recent_traces[:5],
            "active_grants": approval.active_grants,
        }

    def _selection(self, approval: ApprovalContext) -> Selection:
        return select(self._packs, approval.agent_id, approval.tool)

    def _request(self, approval: ApprovalContext) -> tuple[str, dict, dict]:
        body = {"state": self._state(approval), "questions": self._selection(approval).wire()}
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
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout)
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
        payload = _payload(data)
        return self._combine(_unwrap(data), model=str((payload or {}).get("model", "")),
                             selection=self._selection(approval))

    def _combine(self, answers: dict | None, model: str = "",
                 selection: Selection | None = None) -> Verdict | None:
        sel = selection or select(self._packs, "", "")
        try:
            values: dict[str, float] = {}
            choices: dict[str, dict] = {}
            for q in sel.questions:
                a = answers[q.name]
                if q.type == "noul":
                    values[q.name] = float(a["noul"])
                elif q.type == "choice":
                    choices[q.name] = {k: float(v) for k, v in a["probabilities"].items()}
        except (KeyError, TypeError, ValueError, AttributeError):
            logger.warning("unexpected Jev answer: %s", json.dumps(answers)[:300])
            return None

        j = self._jev
        dangers = [q for q in sel.questions if q.group == "danger"]
        manipulation = [q for q in sel.questions if q.group == "manipulation"]
        scope_q = next((q for q in sel.questions if q.role == "scope"), None)
        scope = values[scope_q.name] if scope_q else None

        # a danger with ignore_when does not count where it is the agent's job
        # (overwriting inside the project)
        shown: dict[str, float] = {}
        counted = []
        for q in dangers:
            iw = q.ignore_when
            if iw:
                p = choices.get(iw["question"], {}).get(iw["option"], 0.0)
                shown[iw["option"]] = p
                if p >= iw.get("min", j.project_min):
                    continue
            counted.append(q)
        destructive = max((values[q.name] for q in counted), default=0.0)
        manip = max((values[q.name] for q in manipulation), default=0.0)

        detail = " · ".join(f"{q.name} {values[q.name]:.2f}" for q in dangers)
        tail = [f"{k} {v:.2f}" for k, v in shown.items()]
        tail += [f"{q.name} {values[q.name]:.2f}" for q in sel.questions
                 if q.group == "context" and q.type == "noul"]
        tail += [f"{q.name} {values[q.name]:.2f}" for q in manipulation]
        signals = f"destructive {destructive:.2f} ({detail}) · " + " · ".join(tail)
        # provenance: which model, which questions, which thresholds decided
        meta = {
            "model": model or "unknown",
            "questions": sel.sha,
            "packs": sel.packs,
            "thresholds": {"injection_max": j.injection_max, "deny_min": j.deny_min,
                           "destructive_max": j.destructive_max, "in_scope_min": j.in_scope_min,
                           "deny_in_scope_max": j.deny_in_scope_max,
                           "project_min": j.project_min},
        }
        own = {q.name: q.threshold for q in sel.questions if q.threshold is not None}
        if own:
            meta["thresholds"]["per_question"] = own
        who = f"Jev {meta['model']} q={sel.sha}"

        def limit(q, default):
            return q.threshold if q.threshold is not None else default

        hit = next((q for q in manipulation if values[q.name] > limit(q, j.injection_max)), None)
        if hit:
            return Verdict("escalate", manip, f"{who}: possible {hit.name} ({signals})", meta=meta, raw=answers)
        if scope is not None and destructive >= j.deny_min and scope < j.deny_in_scope_max:
            return Verdict("deny", destructive, f"{who}: deny ({signals})", meta=meta, raw=answers)
        if all(values[q.name] <= limit(q, j.destructive_max) for q in counted) and (
                scope is None or scope >= j.in_scope_min):
            # confidence is checked against confidence_threshold by the evaluator;
            # scope gates approval but stays out of the confidence: on the bench
            # it separated nothing and mostly produced false alerts (2026-09-29)
            confidence = min(1 - destructive, 1 - manip)
            return Verdict("approve", round(confidence, 4), f"{who}: approve ({signals})", meta=meta, raw=answers)
        return Verdict("escalate", 1 - destructive, f"{who}: escalate ({signals})", meta=meta, raw=answers)

    def _required_env(self) -> list[str]:
        if self._jev.backend == "cloudflare" and not self._jev.url:
            return [self._jev.api_key_env, self._jev.account_id_env]
        return [self._jev.api_key_env]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def _payload(data) -> dict | None:
    """The object holding answers and model, whatever the envelope.

    TypeSafe returns {"model", "answers", ...}. Cloudflare wraps it twice, as
    observed on a real call: {"result": {"state": "Completed", "result": {...}}}.
    """
    for _ in range(3):
        if not isinstance(data, dict):
            return None
        if "answers" in data:
            return data
        data = data.get("result")
    return None


def _unwrap(data) -> dict | None:
    """The answers dict, whatever the envelope."""
    payload = _payload(data)
    return payload.get("answers") if payload else None


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
