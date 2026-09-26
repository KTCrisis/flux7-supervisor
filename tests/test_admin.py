"""Tests for the admin API: routes, auth, pause/resume, and the runner's reporting."""

from starlette.testclient import TestClient

from sup7.admin import create_admin_app
from sup7.config import SupervisorConfig
from sup7.models import Decision
from sup7.runner import SupervisorRunner


class FakeTarget:
    def __init__(self):
        self.paused = False
        self.decisions = [{"approval_id": str(i), "decision": "approved"} for i in range(10)]

    def status(self):
        return {"state": "paused" if self.paused else "running"}

    def config_summary(self):
        return {"rules": [{"name": "reads"}]}

    def recent_decisions(self, limit):
        return self.decisions[:limit]

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def _client(token=""):
    target = FakeTarget()
    return target, TestClient(create_admin_app(target, token=token))


# ── routes ────────────────────────────────────────────────────
def test_routes_without_token():
    target, c = _client()
    assert c.get("/health").json() == {"ok": True}
    assert c.get("/status").json() == {"state": "running"}
    assert c.get("/config").json()["rules"][0]["name"] == "reads"
    assert len(c.get("/decisions?limit=3").json()["decisions"]) == 3


def test_pause_and_resume():
    target, c = _client()
    assert c.post("/pause").json()["state"] == "paused" and target.paused
    assert c.post("/resume").json()["state"] == "running" and not target.paused


def test_decisions_limit_validation():
    _, c = _client()
    assert c.get("/decisions?limit=abc").status_code == 400
    assert len(c.get("/decisions?limit=0").json()["decisions"]) == 1  # clamped to 1


def test_token_required_when_set():
    target, c = _client(token="s3cret")
    assert c.get("/health").status_code == 200  # liveness stays open
    assert c.get("/status").status_code == 401
    assert c.post("/pause").status_code == 401 and not target.paused
    ok = {"Authorization": "Bearer s3cret"}
    assert c.get("/status", headers=ok).status_code == 200
    assert c.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_writes_need_post():
    _, c = _client()
    assert c.get("/pause").status_code == 405


# ── runner reporting ──────────────────────────────────────────
def _runner(**evaluator):
    cfg = SupervisorConfig(evaluator=evaluator or {"provider": "ollama", "model": "qwen3:14b"},
                           rules=[{"name": "reads", "condition": "tool contains read", "action": "approve"}])
    return SupervisorRunner(cfg)


def _decision(kind="approved"):
    return Decision(
        timestamp=Decision.now(), approval_id="a-1", agent_id="claude-code", tool="filesystem.read_file",
        decision=kind, rule_matched=None, reasoning="[jev] Jev: approve (approve 0.92 · destructive 0.03)",
        confidence=0.92, evaluation_ms=120,
    )


def test_runner_status_single_provider():
    r = _runner()
    st = r.status()
    assert st["state"] == "running" and st["evaluator"]["mode"] == "single"
    assert st["evaluator"]["providers"][0]["name"] == "ollama"
    assert st["evaluator"]["providers"][0]["detail"] == "qwen3:14b"


def test_runner_status_chain_and_config_without_secrets():
    r = _runner(chain=[{"provider": "jev", "jev": {"api_key_env": "CF_TOKEN"}}, {"provider": "ollama"}])
    st = r.status()
    assert st["evaluator"]["mode"] == "chain"
    assert [p["name"] for p in st["evaluator"]["providers"]] == ["jev", "ollama"]
    assert st["evaluator"]["providers"][0]["state"] == "ok"
    cfg = r.config_summary()
    assert cfg["evaluator"]["providers"][0]["backend"] == "cloudflare"
    assert "CF_TOKEN" not in str(cfg) and "api_key_env" not in str(cfg)
    assert cfg["rules"][0]["name"] == "reads"


def test_runner_remembers_decisions_and_pauses():
    r = _runner()
    r._remember(_decision("approved"))
    r._remember(_decision("escalated"))
    assert r.status()["decisions"]["approved"] == 1 and r.status()["decisions"]["escalated"] == 1
    recent = r.recent_decisions(10)
    assert recent[0]["decision"] == "escalated"  # most recent first
    assert "destructive 0.03" in recent[1]["reasoning"]
    r.pause()
    assert r.status()["state"] == "paused"
    r.resume()
    assert r.status()["state"] == "running"
