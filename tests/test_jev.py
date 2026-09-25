"""Tests for the Jev provider: request shape per backend, and the fail-closed combination."""

import json

import httpx
import pytest

from sup7.config import EvaluatorConfig, JevConfig
from sup7.models import ApprovalContext
from sup7.providers import create_evaluator
from sup7.providers.jev import JevEvaluator


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "a-1", "agent_id": "claude-code", "tool": "filesystem.write_file",
                "params": {"path": "/home/u/project/notes.md", "content": "hello"}}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


def _answers(choice="approve", probs=None, confidence=0.9, destructive=0.05, in_scope=0.9, injection=0.02):
    probs = probs or {"approve": 0.9, "escalate": 0.08, "deny": 0.02}
    return {
        "decision": {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probs},
        "destructive": {"type": "noul", "noul": destructive},
        "in_scope": {"type": "noul", "noul": in_scope},
        "injection": {"type": "noul", "noul": injection},
    }


def _evaluator(handler, **jev) -> JevEvaluator:
    ev = JevEvaluator(EvaluatorConfig(provider="jev", jev=JevConfig(**jev)))
    ev._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ev


def _ok(body):
    return lambda request: httpx.Response(200, json=body)


# ── wiring ────────────────────────────────────────────────────
def test_factory_returns_jev():
    assert isinstance(create_evaluator(EvaluatorConfig(provider="jev")), JevEvaluator)


async def test_cloudflare_request_shape(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {"answers": _answers()}, "success": True})

    verdict = await _evaluator(handler).evaluate(_ctx())
    assert seen["url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run"
    assert seen["auth"] == "Bearer cf-token"
    assert seen["body"]["model"] == "typesafe/jev"
    assert set(seen["body"]["input"]["questions"]) == {"decision", "destructive", "in_scope", "injection"}
    assert seen["body"]["input"]["state"]["tool"] == "filesystem.write_file"
    assert verdict.action == "approve"


async def test_typesafe_request_shape(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": _answers()})

    await _evaluator(handler, backend="typesafe", api_key_env="TYPESAFE_API_KEY").evaluate(_ctx())
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["body"]["model"] == "jev-latest"
    assert "state" in seen["body"] and "input" not in seen["body"]


async def test_redacted_params_never_sent():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": _answers()})

    ctx = _ctx(params={"path": "/tmp/x", "content": "SECRET"})
    await _evaluator(handler, redact_params=["content"]).evaluate(ctx)
    sent = json.dumps(seen["body"])
    assert "SECRET" not in sent and "[redacted]" in sent


# ── combination ───────────────────────────────────────────────
async def test_approve_when_all_signals_agree():
    v = await _evaluator(_ok({"answers": _answers()})).evaluate(_ctx())
    assert v.action == "approve" and v.confidence == 0.9
    assert "destructive 0.05" in v.reasoning  # probabilities are kept for audit


async def test_destructive_blocks_approval():
    v = await _evaluator(_ok({"answers": _answers(destructive=0.6)})).evaluate(_ctx())
    assert v.action == "escalate"


async def test_out_of_scope_blocks_approval():
    v = await _evaluator(_ok({"answers": _answers(in_scope=0.3)})).evaluate(_ctx())
    assert v.action == "escalate"


async def test_injection_escalates_even_if_approve():
    v = await _evaluator(_ok({"answers": _answers(injection=0.8)})).evaluate(_ctx())
    assert v.action == "escalate" and "injection" in v.reasoning


async def test_deny_only_when_very_probable():
    likely = _answers(choice="deny", probs={"approve": 0.0, "escalate": 0.05, "deny": 0.95}, destructive=0.9)
    unsure = _answers(choice="deny", probs={"approve": 0.1, "escalate": 0.3, "deny": 0.6}, destructive=0.9)
    assert (await _evaluator(_ok({"answers": likely})).evaluate(_ctx())).action == "deny"
    assert (await _evaluator(_ok({"answers": unsure})).evaluate(_ctx())).action == "escalate"


async def test_escalate_choice_passes_through():
    ans = _answers(choice="escalate", probs={"approve": 0.2, "escalate": 0.7, "deny": 0.1})
    v = await _evaluator(_ok({"answers": ans})).evaluate(_ctx())
    assert v.action == "escalate"


# ── failures: None means the evaluator escalates to a human ──
@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(429, json={"error": "rate limited"}),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"answers": {"decision": {"choice": "approve"}}}),  # missing questions
])
async def test_failures_return_none(response):
    assert await _evaluator(lambda request: response).evaluate(_ctx()) is None


async def test_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("down")
    assert await _evaluator(handler).evaluate(_ctx()) is None
