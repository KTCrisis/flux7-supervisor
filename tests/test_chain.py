"""Tests for the provider chain: order, fallback on failure, breaker, config wiring."""

import pytest

from sup7.config import EvaluatorConfig, SupervisorConfig
from sup7.models import ApprovalContext, Verdict
from sup7.providers import create_evaluator
from sup7.providers.chain import ChainEvaluator


class Fake:
    """Evaluator stub: returns the queued results in order (None = failure)."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0
        self.closed = False

    async def evaluate(self, approval):
        self.calls += 1
        r = self.results.pop(0) if self.results else None
        if isinstance(r, Exception):
            raise r
        return r

    async def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _ctx():
    return ApprovalContext(id="c-1", agent_id="agent", tool="filesystem.read_file")


APPROVE = Verdict("approve", 0.9, "routine")
ESCALATE = Verdict("escalate", 0.6, "unsure")


async def test_first_answer_wins_and_is_labelled():
    a, b = Fake(APPROVE), Fake(ESCALATE)
    v = await ChainEvaluator([("jev", a), ("ollama", b)]).evaluate(_ctx())
    assert v.action == "approve" and v.reasoning == "[jev] routine"
    assert b.calls == 0


async def test_escalate_is_an_answer_not_a_failure():
    a, b = Fake(ESCALATE), Fake(APPROVE)
    v = await ChainEvaluator([("jev", a), ("ollama", b)]).evaluate(_ctx())
    assert v.action == "escalate" and b.calls == 0


async def test_fallback_on_failure_and_on_exception():
    a, b, c = Fake(None), Fake(RuntimeError("boom")), Fake(APPROVE)
    v = await ChainEvaluator([("jev", a), ("anthropic", b), ("ollama", c)]).evaluate(_ctx())
    assert v.action == "approve"
    assert v.reasoning == "[ollama, jev failed, anthropic failed] routine"


async def test_all_fail_returns_none():
    v = await ChainEvaluator([("jev", Fake(None)), ("ollama", Fake(None))]).evaluate(_ctx())
    assert v is None


async def test_breaker_skips_then_retries_after_cooldown():
    clock = Clock()
    jev = Fake(None, None, APPROVE)
    ollama = Fake(ESCALATE, ESCALATE, ESCALATE, ESCALATE)
    chain = ChainEvaluator([("jev", jev), ("ollama", ollama)], failures=2, cooldown=60, clock=clock)

    await chain.evaluate(_ctx())          # jev fails (1)
    await chain.evaluate(_ctx())          # jev fails (2) -> tripped
    v = await chain.evaluate(_ctx())      # jev skipped
    assert jev.calls == 2 and v.reasoning.startswith("[ollama, jev skipped]")

    clock.t += 61                          # cooldown over: jev tried again
    v = await chain.evaluate(_ctx())
    assert jev.calls == 3 and v.reasoning == "[jev] routine"


async def test_success_resets_failure_count():
    clock = Clock()
    jev = Fake(None, APPROVE, None, APPROVE)
    chain = ChainEvaluator([("jev", jev), ("ollama", Fake(ESCALATE, ESCALATE))], failures=2, clock=clock)
    for _ in range(4):
        await chain.evaluate(_ctx())
    assert jev.calls == 4  # never tripped: failures were not consecutive


async def test_close_closes_every_provider():
    a, b = Fake(), Fake()
    await ChainEvaluator([("a", a), ("b", b)]).close()
    assert a.closed and b.closed


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        ChainEvaluator([])


# ── config wiring ─────────────────────────────────────────────
def test_config_chain_builds_chain_evaluator():
    cfg = EvaluatorConfig(chain=[{"provider": "jev"}, {"provider": "ollama", "model": "qwen3:14b"}])
    ev = create_evaluator(cfg)
    assert isinstance(ev, ChainEvaluator)
    assert [name for name, _ in ev._providers] == ["jev", "ollama"]


def test_single_provider_config_unchanged():
    ev = create_evaluator(EvaluatorConfig(provider="ollama"))
    assert not isinstance(ev, ChainEvaluator)


def test_yaml_shape_parses():
    cfg = SupervisorConfig(evaluator={
        "confidence_threshold": 0.8,
        "breaker_failures": 2,
        "chain": [{"provider": "jev", "jev": {"backend": "cloudflare"}}, {"provider": "ollama"}],
    })
    assert cfg.evaluator.chain[0].jev.backend == "cloudflare"
    assert cfg.evaluator.breaker_failures == 2


async def test_answer_carries_its_label():
    v = await ChainEvaluator([("jev", Fake(None)), ("ollama", Fake(APPROVE))],
                             labels=["jev:cloudflare", "ollama:qwen3:14b"]).evaluate(_ctx())
    assert v.source == "ollama:qwen3:14b" and v.reasoning.startswith("[ollama, jev failed]")
