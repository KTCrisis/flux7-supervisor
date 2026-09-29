"""Tests for the Jev provider: request shape per backend, and the fail-closed combination."""

import json

import httpx
import pytest

from sup7.config import EvaluatorConfig, JevConfig
from sup7.models import ApprovalContext
from sup7.providers import create_evaluator
from sup7.providers.jev import DESTRUCTIVE, QUESTIONS_SHA, JevEvaluator


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "a-1", "agent_id": "claude-code", "tool": "filesystem.write_file",
                "params": {"path": "/home/u/project/notes.md", "content": "hello"}}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


def _answers(destructive=0.05, in_scope=0.9, injection=0.02, harm="deletes", project=0.9):
    """destructive sets one of the four harm signals (harm), the others stay low."""
    harms = {k: {"type": "noul", "noul": destructive if k == harm else 0.01} for k in DESTRUCTIVE}
    rest = (1 - project) / 4
    zone = {"type": "choice", "choice": "project" if project >= 0.5 else "home",
            "probabilities": {"project": project, "home": rest, "system": rest, "remote": rest, "none": rest}}
    return {
        **harms,
        "target_zone": zone,
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
    assert set(seen["body"]["input"]["questions"]) == {*DESTRUCTIVE, "target_zone", "in_scope", "injection"}
    assert seen["body"]["input"]["state"]["tool"] == "filesystem.write_file"
    assert verdict.action == "approve"


async def test_cloudflare_double_envelope():
    # shape of a real Workers AI response for typesafe/jev (2026-09-29)
    body = {"result": {"state": "Completed",
                       "result": {"model": "jev-1.13.0", "answers": _answers(),
                                  "usage": {"input_tokens": 638, "output_tokens": 94}}},
            "success": True, "errors": [], "messages": []}
    verdict = await _evaluator(_ok(body)).evaluate(_ctx())
    assert verdict is not None and verdict.action == "approve"


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
    # confidence = min(1 - destructive, 1 - injection) = min(0.95, 0.98)
    assert v.action == "approve" and v.confidence == 0.95
    assert "destructive 0.05" in v.reasoning  # probabilities are kept for audit


async def test_approve_confidence_is_weakest_signal():
    v = await _evaluator(_ok({"answers": _answers(destructive=0.15, in_scope=0.75)})).evaluate(_ctx())
    assert v.action == "approve" and v.confidence == 0.85  # in_scope gates, it does not lower confidence


async def test_scope_floor_and_deny_scope_are_separate():
    # approval floor lowered to 0.3: a call at in_scope 0.5 is approved, not denied
    ans = _answers(destructive=0.1, in_scope=0.5)
    v = await _evaluator(_ok({"answers": ans}), in_scope_min=0.3).evaluate(_ctx())
    assert v.action == "approve"
    # a certain harm out of scope is still denied through deny_in_scope_max
    ans = _answers(destructive=0.97, in_scope=0.5)
    v = await _evaluator(_ok({"answers": ans}), in_scope_min=0.3).evaluate(_ctx())
    assert v.action == "deny"


async def test_overwrite_inside_the_project_is_not_a_harm():
    v = await _evaluator(_ok({"answers": _answers(destructive=0.9, harm="overwrites", project=0.85)})).evaluate(_ctx())
    assert v.action == "approve" and "overwrites 0.90" in v.reasoning and "project 0.85" in v.reasoning


async def test_overwrite_outside_the_project_blocks():
    v = await _evaluator(_ok({"answers": _answers(destructive=0.9, harm="overwrites", project=0.3)})).evaluate(_ctx())
    assert v.action == "escalate"


@pytest.mark.parametrize("harm", ["deletes", "exfiltrates", "secrets"])
async def test_other_harms_block_even_inside_the_project(harm):
    v = await _evaluator(_ok({"answers": _answers(destructive=0.6, harm=harm, project=0.95)})).evaluate(_ctx())
    assert v.action == "escalate"


async def test_project_dirs_reach_the_state():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": _answers()})

    await _evaluator(handler).evaluate(_ctx(project_dirs=["/home/u/project"]))
    assert seen["body"]["input"]["state"]["project_dirs"] == ["/home/u/project"]


@pytest.mark.parametrize("harm", DESTRUCTIVE)
async def test_any_harm_blocks_approval(harm):
    v = await _evaluator(_ok({"answers": _answers(destructive=0.6, harm=harm, project=0.3)})).evaluate(_ctx())
    assert v.action == "escalate"
    assert "destructive 0.60" in v.reasoning and f"{harm} 0.60" in v.reasoning


async def test_out_of_scope_blocks_approval():
    v = await _evaluator(_ok({"answers": _answers(in_scope=0.3)})).evaluate(_ctx())
    assert v.action == "escalate"


async def test_injection_escalates_even_if_safe():
    v = await _evaluator(_ok({"answers": _answers(injection=0.8)})).evaluate(_ctx())
    assert v.action == "escalate" and "injection" in v.reasoning


async def test_injection_wins_over_deny():
    ans = _answers(destructive=0.99, in_scope=0.1, injection=0.9)
    assert (await _evaluator(_ok({"answers": ans})).evaluate(_ctx())).action == "escalate"


async def test_deny_needs_certain_harm_and_out_of_scope():
    certain_out = _answers(destructive=0.97, in_scope=0.2)
    certain_in = _answers(destructive=0.97, in_scope=0.9)  # e.g. a cleanup the agent always does
    likely_out = _answers(destructive=0.8, in_scope=0.2)
    assert (await _evaluator(_ok({"answers": certain_out})).evaluate(_ctx())).action == "deny"
    assert (await _evaluator(_ok({"answers": certain_in})).evaluate(_ctx())).action == "escalate"
    assert (await _evaluator(_ok({"answers": likely_out})).evaluate(_ctx())).action == "escalate"


# ── failures: None means the evaluator escalates to a human ──
@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(429, json={"error": "rate limited"}),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"answers": {"deletes": {"noul": 0.1}}}),  # missing questions
])
async def test_failures_return_none(response):
    assert await _evaluator(lambda request: response).evaluate(_ctx()) is None


async def test_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("down")
    assert await _evaluator(handler).evaluate(_ctx()) is None


async def test_missing_credentials_return_none_without_calling(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    called = []
    ev = _evaluator(lambda request: called.append(1) or httpx.Response(200, json={"answers": _answers()}))
    assert await ev.evaluate(_ctx()) is None
    assert called == []


async def test_http_error_log_hides_account_id(caplog):
    body = {"success": False, "errors": [{"message": "insufficient credits"}]}
    with caplog.at_level("WARNING"):
        assert await _evaluator(lambda request: httpx.Response(402, json=body)).evaluate(_ctx()) is None
    assert "insufficient credits" in caplog.text and "402" in caplog.text
    assert "acc123" not in caplog.text


# ── provenance ────────────────────────────────────────────────
async def test_verdict_records_model_questions_and_thresholds():
    body = {"result": {"state": "Completed",
                       "result": {"model": "jev-1.13.0", "answers": _answers()}}, "success": True}
    v = await _evaluator(_ok(body)).evaluate(_ctx())
    assert v.meta["model"] == "jev-1.13.0"
    assert v.meta["questions"] == QUESTIONS_SHA and len(QUESTIONS_SHA) == 12
    assert v.meta["thresholds"] == {"injection_max": 0.5, "deny_min": 0.9,
                                    "destructive_max": 0.2, "in_scope_min": 0.7,
                                    "deny_in_scope_max": 0.7, "project_min": 0.7}
    # the reasoning carries it too, so the mesh trace (which gets only the reasoning) keeps it
    assert v.reasoning.startswith(f"Jev jev-1.13.0 q={QUESTIONS_SHA}: approve")


async def test_unknown_model_when_the_response_omits_it():
    v = await _evaluator(_ok({"answers": _answers()})).evaluate(_ctx())
    assert v.meta["model"] == "unknown"
