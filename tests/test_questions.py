"""Tests for YAML question sets: validation, packs by agent or tool, decision with packs."""

import json

import httpx
import pytest

from sup7.config import EvaluatorConfig, JevConfig
from sup7.models import ApprovalContext
from sup7.providers.jev import QUESTIONS_SHA, JevEvaluator
from sup7.questions import QuestionError, load_packs, parse_pack, select, socle_text

FINANCE = """
pack: finance
applies_to: ["agent:compta-*", "tool:bank.*"]
questions:
  payment:
    type: noul
    group: danger
    threshold: 0.1
    instructions: The call initiates or approves a payment.
    criteria:
      true: Creates or releases a payment order
      false: Reads balances or prepares a draft
"""


def test_shipped_socle_keeps_the_measured_wording():
    sel = select(load_packs([]), "claude", "Bash")
    assert sel.packs == ["socle"]
    assert sel.sha == QUESTIONS_SHA == "9bf9f0bc317f"  # the questions measured on the bench


def test_unquoted_true_false_criteria_become_strings():
    pack = parse_pack(FINANCE, "finance.yaml")
    assert pack.questions[0].criteria == {"true": "Creates or releases a payment order",
                                          "false": "Reads balances or prepares a draft"}


@pytest.mark.parametrize("text, message", [
    ("questions: {}", "non-empty"),
    ("pack: x\nquestions:\n  a: {type: yesno, group: danger, instructions: x}", "type must be"),
    ("pack: x\nquestions:\n  a: {type: noul, group: risk, instructions: x}", "group must be"),
    ("pack: x\nquestions:\n  a: {type: choice, group: danger, instructions: x}", "must be a noul"),
    ("pack: x\nquestions:\n  a: {type: noul, group: danger}", "instructions are required"),
    ("pack: x\nquestions:\n  a: {type: noul, group: danger, instructions: x, threshold: 2}", "between 0 and 1"),
    ("pack: x\napplies_to: ['team:x']\nquestions:\n  a: {type: noul, group: danger, instructions: x}", "applies_to"),
    ("pack: x\nquestions:\n  a: {type: choice, group: context, instructions: x, criteria: {one: y}}", "two options"),
    ("pack: x\nquestions: [", "invalid YAML"),
])
def test_invalid_sets_are_refused_with_a_reason(text, message):
    with pytest.raises(QuestionError, match=message):
        parse_pack(text, "x.yaml")


def test_duplicate_question_and_bad_ignore_when_are_refused(tmp_path):
    (tmp_path / "socle.yaml").write_text(socle_text())
    (tmp_path / "dup.yaml").write_text("pack: dup\nquestions:\n  deletes: {type: noul, group: danger, instructions: x}")
    with pytest.raises(QuestionError, match="defined twice"):
        load_packs([str(tmp_path / "socle.yaml"), str(tmp_path / "dup.yaml")])
    (tmp_path / "bad.yaml").write_text(
        "pack: bad\nquestions:\n  x: {type: noul, group: danger, instructions: x,"
        " ignore_when: {question: target_zone, option: moon}}")
    with pytest.raises(QuestionError, match="ignore_when"):
        load_packs([str(tmp_path / "socle.yaml"), str(tmp_path / "bad.yaml")])


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(QuestionError, match="cannot read"):
        load_packs([str(tmp_path / "nope.yaml")])


def test_packs_apply_by_agent_or_tool(tmp_path):
    (tmp_path / "socle.yaml").write_text(socle_text())
    (tmp_path / "finance.yaml").write_text(FINANCE)
    packs = load_packs([str(tmp_path / "socle.yaml"), str(tmp_path / "finance.yaml")])
    assert select(packs, "claude", "Bash").packs == ["socle"]
    assert select(packs, "compta-bot", "erp.read").packs == ["socle", "finance"]
    assert select(packs, "claude", "bank.transfer").packs == ["socle", "finance"]
    # another wording, another fingerprint
    assert select(packs, "claude", "bank.transfer").sha != QUESTIONS_SHA


def _answers(payment=0.05, project=0.9):
    a = {k: {"type": "noul", "noul": 0.02} for k in ("deletes", "overwrites", "exfiltrates", "secrets", "injection")}
    a["in_scope"] = {"type": "noul", "noul": 0.9}
    r = (1 - project) / 4
    a["target_zone"] = {"type": "choice", "choice": "project",
                        "probabilities": {"project": project, "home": r, "system": r, "remote": r, "none": r}}
    a["payment"] = {"type": "noul", "noul": payment}
    return a


def _evaluator(tmp_path, handler):
    (tmp_path / "socle.yaml").write_text(socle_text())
    (tmp_path / "finance.yaml").write_text(FINANCE)
    jev = JevConfig(questions=[str(tmp_path / "socle.yaml"), str(tmp_path / "finance.yaml")])
    ev = JevEvaluator(EvaluatorConfig(provider="jev", jev=jev))
    ev._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ev


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")


async def test_pack_questions_are_sent_only_where_they_apply(tmp_path):
    sent = []

    def handler(request):
        sent.append(set(json.loads(request.content)["input"]["questions"]))
        return httpx.Response(200, json={"answers": _answers()})

    ev = _evaluator(tmp_path, handler)
    await ev.evaluate(ApprovalContext(id="1", agent_id="claude", tool="Bash"))
    await ev.evaluate(ApprovalContext(id="2", agent_id="claude", tool="bank.transfer"))
    assert "payment" not in sent[0] and "payment" in sent[1]


async def test_pack_threshold_is_its_own(tmp_path):
    # payment 0.15: under the default destructive_max (0.2), over its own 0.1
    ev = _evaluator(tmp_path, lambda r: httpx.Response(200, json={"answers": _answers(payment=0.15)}))
    v = await ev.evaluate(ApprovalContext(id="1", agent_id="claude", tool="bank.transfer"))
    assert v.action == "escalate" and "payment 0.15" in v.reasoning
    assert v.meta["packs"] == ["socle", "finance"]
    assert v.meta["thresholds"]["per_question"] == {"payment": 0.1}
