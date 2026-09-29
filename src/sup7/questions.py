"""Question sets for the Jev provider, loaded from YAML.

A question set ("pack") says what Jev is asked and how sup7 reads the answer.
The wording (type, instructions, criteria) is sent to Jev; the policy fields
(group, role, threshold, ignore_when) stay in sup7. The base set, `socle`,
ships with sup7 (data/socle.yaml); business packs are extra files listed
in the configuration, each applying to some agents or tools.

Loading validates everything up front: a set that loads is a set sup7 can
decide with, so a bad edit is refused before it reaches production.
"""

from __future__ import annotations

import fnmatch
import glob as globmod
import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

TYPES = {"noul", "choice", "score"}
GROUPS = {"danger", "context", "manipulation"}
WIRE_FIELDS = ("type", "instructions", "criteria")  # what Jev receives


class QuestionError(ValueError):
    """A question set that cannot be used, with the reason."""


@dataclass
class Question:
    name: str
    type: str
    group: str
    instructions: str
    criteria: dict[str, str] = field(default_factory=dict)
    role: str = ""  # "scope": gates approval and deny
    threshold: float | None = None  # None: the group default from JevConfig
    ignore_when: dict | None = None  # {"question", "option", "min"}

    def wire(self) -> dict:
        """The question as Jev receives it."""
        q = {"type": self.type, "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = dict(self.criteria)
        return q


@dataclass
class Pack:
    name: str
    applies_to: list[str]
    questions: list[Question]
    source: str = ""

    def applies(self, agent_id: str, tool: str) -> bool:
        if not self.applies_to:
            return True
        for pattern in self.applies_to:
            kind, _, glob = pattern.partition(":")
            value = {"agent": agent_id, "tool": tool}.get(kind)
            if value is not None and fnmatch.fnmatchcase(value, glob):
                return True
        return False


def _criteria(raw, where: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise QuestionError(f"{where}: criteria must be a mapping")
    # an unquoted `true:` key is read by YAML as a boolean
    return {(str(k).lower() if isinstance(k, bool) else str(k)): str(v) for k, v in raw.items()}


def parse_pack(text: str, source: str = "") -> Pack:
    """Parse and validate one question set."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise QuestionError(f"{source}: invalid YAML: {e}") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("questions"), dict) or not doc["questions"]:
        raise QuestionError(f"{source}: expected `pack`, `applies_to` and a non-empty `questions` mapping")
    name = str(doc.get("pack") or Path(source).stem or "pack")
    applies_to = doc.get("applies_to") or []
    if not isinstance(applies_to, list) or any(
        not isinstance(p, str) or p.partition(":")[0] not in ("agent", "tool") for p in applies_to
    ):
        raise QuestionError(f"{source}: applies_to takes patterns like 'agent:compta-*' or 'tool:bank.*'")

    questions = []
    for qname, raw in doc["questions"].items():
        where = f"{source}: question {qname!r}"
        if not isinstance(raw, dict):
            raise QuestionError(f"{where}: expected a mapping")
        qtype, group = raw.get("type"), raw.get("group")
        if qtype not in TYPES:
            raise QuestionError(f"{where}: type must be one of {sorted(TYPES)}")
        if group not in GROUPS:
            raise QuestionError(f"{where}: group must be one of {sorted(GROUPS)}")
        if group in ("danger", "manipulation") and qtype != "noul":
            raise QuestionError(f"{where}: a {group} question must be a noul (a yes/no probability)")
        if not str(raw.get("instructions") or "").strip():
            raise QuestionError(f"{where}: instructions are required")
        criteria = _criteria(raw.get("criteria"), where)
        if qtype == "choice" and len(criteria) < 2:
            raise QuestionError(f"{where}: a choice needs at least two options under criteria")
        threshold = raw.get("threshold")
        if threshold is not None and not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            raise QuestionError(f"{where}: threshold must be a number between 0 and 1")
        role = raw.get("role", "")
        if role not in ("", "scope") or (role == "scope" and qtype != "noul"):
            raise QuestionError(f"{where}: role can only be 'scope', on a noul")
        questions.append(Question(
            name=str(qname), type=qtype, group=group, instructions=str(raw["instructions"]),
            criteria=criteria, role=role, threshold=threshold, ignore_when=raw.get("ignore_when"),
        ))
    return Pack(name=name, applies_to=applies_to, questions=questions, source=source)


def socle_text() -> str:
    return resources.files("sup7").joinpath("data/socle.yaml").read_text()


def expand(paths: list[str], extra: list[str] | None = None) -> list[Path]:
    """Configured entries as files: `~` expanded, globs matched and sorted.

    A glob lets a new pack be added by creating a file in its directory;
    `extra` are files about to be created, kept when a glob would match them.
    """
    out: list[Path] = []
    for entry in paths:
        pattern = str(Path(entry).expanduser())
        if globmod.has_magic(pattern):
            found = set(globmod.glob(pattern))
            found |= {x for x in (extra or []) if fnmatch.fnmatch(x, pattern)}
            out += [Path(x) for x in sorted(found)]
        else:
            out.append(Path(pattern))
    return out


def load_packs(paths: list[str], overrides: dict[str, str] | None = None) -> list[Pack]:
    """The configured packs, or the shipped socle when none is configured.

    `overrides` maps a file path to the text it is about to hold: an edit is
    validated with every other pack, before it is written.
    """
    overrides = {str(Path(k).expanduser()): v for k, v in (overrides or {}).items()}
    if not paths:
        packs = [parse_pack(socle_text(), "socle.yaml (shipped)")]
    else:
        packs = []
        for path in expand(paths, extra=list(overrides)):
            if str(path) in overrides:
                text = overrides[str(path)]
            else:
                try:
                    text = path.read_text()
                except OSError as e:
                    raise QuestionError(f"{path}: cannot read ({e.strerror})") from None
            packs.append(parse_pack(text, str(path)))
        if not packs:
            raise QuestionError(f"no question file matches {paths}")
    check(packs)
    return packs


def check(packs: list[Pack]) -> None:
    """Cross-pack validation: unique names, ignore_when pointing at a real option."""
    seen: dict[str, str] = {}
    for pack in packs:
        for q in pack.questions:
            if q.name in seen:
                raise QuestionError(f"question {q.name!r} is defined twice ({seen[q.name]}, {pack.name})")
            seen[q.name] = pack.name
    choices = {q.name: q for pack in packs for q in pack.questions if q.type == "choice"}
    for pack in packs:
        for q in pack.questions:
            iw = q.ignore_when
            if iw is None:
                continue
            target = choices.get(iw.get("question", "") if isinstance(iw, dict) else "")
            if target is None or iw.get("option") not in target.criteria:
                raise QuestionError(
                    f"{pack.name}: {q.name}.ignore_when must name a choice question and one of its options")


@dataclass
class Selection:
    """The questions asked for one call: every pack that applies to it."""

    packs: list[str]
    questions: list[Question]

    def wire(self) -> dict:
        return {q.name: q.wire() for q in self.questions}

    @property
    def sha(self) -> str:
        """Fingerprint of the wording sent to Jev (policy fields excluded)."""
        return hashlib.sha256(json.dumps(self.wire(), sort_keys=True).encode()).hexdigest()[:12]


def select(packs: list[Pack], agent_id: str, tool: str) -> Selection:
    chosen = [p for p in packs if p.applies(agent_id, tool)]
    return Selection(packs=[p.name for p in chosen], questions=[q for p in chosen for q in p.questions])
