"""Editing sup7's configuration and question sets as text, safely.

The admin API lets flux7-console change what sup7 does: its rules,
thresholds, provider chain, and the questions Jev is asked. Changing who
decides is a governance act, so an edit is:

  validated   the whole resulting configuration must load (schema, rules,
              every question set with the edit in place) before anything is
              written; a bad edit leaves production untouched
  guarded     the client sends the fingerprint it read (If-Match); a file
              changed on disk since then is not overwritten
  kept        the previous version is copied to <file>.bak-<timestamp>
  atomic      written to a temporary file, then renamed
  masked      token values never leave in a read; the mask is put back on write

Files are named by id: "config" for sup7.yaml, "questions/<file>.yaml" for a
question set listed (directly or through a glob) in evaluator.jev.questions.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from sup7.config import EvaluatorConfig, SupervisorConfig, parse_config
from sup7.questions import QuestionError, expand, load_packs
from sup7.rules import parse_condition

MASK = "***redacted***"
TOKEN_LINE = re.compile(r"^(\s*token:\s*)(.*?)\s*$", re.MULTILINE)
PACK_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\.ya?ml$")
# sections read once at start: changing them needs a restart of sup7
RESTART_FIELDS = ("mesh", "memory", "admin", "mcp_server", "decision_log")


class EditError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def mask(text: str) -> str:
    """Hide every non-empty `token:` value."""
    def repl(m: re.Match) -> str:
        value = m.group(2).strip()
        return m.group(0) if value in ("", '""', "''") else f"{m.group(1)}{MASK}"
    return TOKEN_LINE.sub(repl, text)


def unmask(new: str, current: str) -> str:
    """Put the real token values back where the client kept the mask."""
    values = [m.group(2) for m in TOKEN_LINE.finditer(current) if m.group(2).strip() not in ("", '""', "''")]
    lines = [m for m in TOKEN_LINE.finditer(new) if m.group(2).strip() == MASK]
    if not lines:
        return new
    if len(lines) != len(values):
        raise EditError(400, "masked tokens were moved or added; change tokens in the file on disk")
    out, last = [], 0
    for m, value in zip(lines, values):
        out.append(new[last:m.start(2)] + value)
        last = m.end(2)
    return "".join(out) + new[last:]


def _jev_entries(ev: EvaluatorConfig) -> list[EvaluatorConfig]:
    return [c for c in [*ev.chain, ev] if c.provider == "jev"]


def question_patterns(config: SupervisorConfig) -> list[str]:
    seen: list[str] = []
    for c in _jev_entries(config.evaluator):
        for p in c.jev.questions:
            if p not in seen:
                seen.append(p)
    return seen


def validate(config_text: str, overrides: dict[str, str] | None = None) -> SupervisorConfig:
    """The configuration that would run, or EditError(400) with the reason."""
    try:
        config = parse_config(config_text)
    except Exception as e:  # yaml or pydantic: the message says where
        raise EditError(400, f"configuration: {e}") from None
    for rule in config.rules:
        if rule.condition:
            try:
                parse_condition(rule.condition)
            except ValueError as e:
                raise EditError(400, f"rule {rule.name!r}: {e}") from None
    for c in _jev_entries(config.evaluator):
        try:
            load_packs(c.jev.questions, overrides)
        except QuestionError as e:
            raise EditError(400, f"questions: {e}") from None
    return config


@dataclass
class FileRef:
    id: str
    kind: str  # "config" | "questions"
    path: Path
    exists: bool

    def describe(self) -> dict:
        text = self.path.read_text() if self.exists else ""
        return {"id": self.id, "kind": self.kind, "path": str(self.path),
                "sha": sha(text) if self.exists else None}


class ConfigFiles:
    """The editable files of a running sup7."""

    def __init__(self, config_path: str, config: SupervisorConfig) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = config

    def files(self) -> list[FileRef]:
        refs = [FileRef("config", "config", self.config_path, self.config_path.exists())]
        for path in expand(question_patterns(self.config)):
            refs.append(FileRef(f"questions/{path.name}", "questions", path, path.exists()))
        return refs

    def _new_pack_path(self, name: str) -> Path | None:
        """Where a new pack file named `name` would go: a configured glob must match it."""
        if not PACK_NAME.match(name):
            return None
        for pattern in question_patterns(self.config):
            p = Path(pattern).expanduser()
            candidate = p.parent / name
            if any(ch in p.name for ch in "*?[") and fnmatch.fnmatch(str(candidate), str(p)):
                return candidate
        return None

    def resolve(self, file_id: str, creating: bool = False) -> FileRef:
        for ref in self.files():
            if ref.id == file_id:
                return ref
        if creating and file_id.startswith("questions/"):
            path = self._new_pack_path(file_id.split("/", 1)[1])
            if path is not None:
                return FileRef(file_id, "questions", path, False)
            raise EditError(400, "a new question file needs a name like 'finance.yaml' and a glob "
                                 "in evaluator.jev.questions that matches it")
        raise EditError(404, f"no editable file {file_id!r}")

    def read(self, file_id: str) -> tuple[str, str]:
        ref = self.resolve(file_id)
        text = ref.path.read_text()
        return (mask(text) if ref.kind == "config" else text), sha(text)

    def write(self, file_id: str, text: str, if_match: str | None) -> tuple[SupervisorConfig, dict]:
        """Validate, back up and write one file; returns the configuration that now applies."""
        ref = self.resolve(file_id, creating=True)
        current = ref.path.read_text() if ref.exists else None
        if current is None:
            if if_match not in (None, "", "new"):
                raise EditError(409, f"{file_id} does not exist; create it with If-Match: new")
        elif if_match != sha(current):
            raise EditError(409, f"{file_id} changed since it was read (now {sha(current)}); reload it first")
        if ref.kind == "config":
            text = unmask(text, current or "")
            config = validate(text)
        else:
            config = validate(self.config_path.read_text(), {str(ref.path): text})
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if current is not None:
            shutil.copy2(ref.path, f"{ref.path}.bak-{stamp}")
        tmp = ref.path.with_name(f".{ref.path.name}.tmp")
        # the file holds tokens: the new version keeps the old one's mode (600),
        # a new file gets 600; created closed, never readable in between
        mode = stat.S_IMODE(ref.path.stat().st_mode) if current is not None else 0o600
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, mode)  # O_CREAT mode is filtered by the umask
        os.replace(tmp, ref.path)
        return config, {"file": file_id, "path": str(ref.path),
                        "sha_before": sha(current) if current is not None else None,
                        "sha_after": sha(text), "backup": f"{ref.path}.bak-{stamp}" if current else None}


def split_restart(old: SupervisorConfig, new: SupervisorConfig) -> tuple[SupervisorConfig, list[str]]:
    """The new configuration with the start-only sections kept as running, and those that differ."""
    pending = [f for f in RESTART_FIELDS if getattr(old, f) != getattr(new, f)]
    applied = new.model_copy(update={f: getattr(old, f) for f in RESTART_FIELDS})
    return applied, pending
