"""Tests for POST /evaluate: sup7 as a decision service outside the mesh queue."""

import json

import httpx
import pytest

from sup7.admin import create_admin_app
from sup7.config import SupervisorConfig
from sup7.models import Verdict
from sup7.runner import SupervisorRunner


class _Stub:
    def __init__(self, verdict):
        self.verdict = verdict
        self.seen = []

    async def evaluate(self, approval):
        self.seen.append(approval)
        return self.verdict

    async def close(self):
        pass


@pytest.fixture
def runner(tmp_path):
    cfg = SupervisorConfig(
        decision_log=str(tmp_path / "d.jsonl"),
        project_dirs=["/home/u/project"],
        evaluator={"provider": "none"},
        rules=[{"name": "project-writes", "condition": "params.path starts_with project_dir",
                "action": "approve", "confidence": 0.9}],
    )
    r = SupervisorRunner(cfg)
    r._logger.open()
    yield r
    r._logger.close()


def _client(runner, token="t"):
    app = create_admin_app(runner, token=token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sup7",
                             headers={"Authorization": f"Bearer {token}"} if token else {})


async def test_a_rule_decides_and_the_caller_gets_approve(runner, tmp_path):
    async with _client(runner) as c:
        r = await c.post("/evaluate", json={"agent_id": "bot", "tool": "fs.write",
                                            "params": {"path": "/home/u/project/a.txt"}})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "approve" and body["rule_matched"] == "project-writes"
    assert body["id"].startswith("eval-")
    logged = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[-1])
    assert logged["via"] == "evaluate" and logged["tool"] == "fs.write"


async def test_the_evaluator_answers_what_no_rule_settles(runner):
    stub = _Stub(Verdict("deny", 0.99, "Jev: deny (destructive 0.99)", "jev:cloudflare",
                         {"model": "jev-1.13.0", "questions": "abc"}))
    runner._evaluator._llm = stub
    async with _client(runner) as c:
        r = await c.post("/evaluate", json={"agent_id": "bot", "tool": "Bash",
                                            "params": {"command": "rm -rf ~/.ssh"},
                                            "recent_traces": [{"tool": "Read"}] * 9})
    body = r.json()
    assert body["decision"] == "deny" and body["evaluator"]["model"] == "jev-1.13.0"
    assert body["rule_matched"] == "jev:cloudflare"
    assert len(stub.seen[0].recent_traces) == 5  # context capped as for a polled approval
    assert stub.seen[0].project_dirs == ["/home/u/project"]  # the evaluator adds the project


async def test_paused_escalates_without_evaluating(runner):
    stub = _Stub(Verdict("approve", 0.99, "x"))
    runner._evaluator._llm = stub
    runner.pause()
    async with _client(runner) as c:
        body = (await c.post("/evaluate", json={"tool": "Bash", "params": {}})).json()
    assert body["decision"] == "escalate" and "paused" in body["reasoning"] and stub.seen == []


@pytest.mark.parametrize("payload", [{"params": {}}, {"tool": "", "params": {}}, {"tool": "x", "params": "no"}, [1]])
async def test_bad_input_is_refused(runner, payload):
    async with _client(runner) as c:
        r = await c.post("/evaluate", json=payload)
    assert r.status_code == 400


async def test_needs_the_token_when_one_is_set(runner):
    app = create_admin_app(runner, token="secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as bare:
        r = await bare.post("/evaluate", json={"tool": "x", "params": {}})
    assert r.status_code == 401
