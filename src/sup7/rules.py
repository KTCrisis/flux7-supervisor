"""Condition parser and predicate evaluation for supervisor rules."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from sup7.models import ApprovalContext

Predicate = Callable[[ApprovalContext, list[str]], bool]

OPERATORS = {"starts_with", "equals", "contains", "not_equals", "in", "==", "!="}


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'params.path' against an object or dict."""
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def in_project(path: str, project_dirs: list[str]) -> bool:
    """The path is one of the project dirs or inside one, after normalisation.

    A raw prefix test let "/p/flux7-mesh/../.bashrc" and "/p/flux7-mesh-backup/x"
    pass for "/p/flux7-mesh": the path is normalised (".." resolved, "~"
    expanded) and must continue with a separator. A relative path never matches.
    """
    p = os.path.normpath(os.path.expanduser(path))
    if not os.path.isabs(p):
        return False
    for d in project_dirs:
        root = os.path.normpath(os.path.expanduser(d))
        if p == root or p.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def parse_condition(condition: str) -> Predicate:
    """Parse a condition string into a predicate function.

    Supported forms:
        "params.path starts_with project_dir"
        "injection_risk == true"
        "tool equals filesystem.read_file"
        "tool contains write"
        "tool not_equals filesystem.delete"
        "tool in filesystem.read_file, filesystem.list_directory"   (exact names, comma-separated)

    `contains` is a substring test: "tool contains read" also matches
    gmail_mark_as_read, which writes. Prefer `in` with exact names for approvals.
    """
    parts = condition.split()

    op_idx = None
    for i, part in enumerate(parts):
        if part in OPERATORS:
            op_idx = i
            break

    if op_idx is None or op_idx == 0:
        raise ValueError(f"cannot parse condition: {condition!r}")

    left_path = ".".join(parts[:op_idx])
    op = parts[op_idx]
    right_value = " ".join(parts[op_idx + 1 :])

    def predicate(approval: ApprovalContext, project_dirs: list[str]) -> bool:
        left = _resolve_path(approval, left_path)
        if left is None:
            return False

        left_str = str(left).lower()

        if right_value == "project_dir":
            if op == "starts_with":
                return in_project(str(left), project_dirs)
            return False

        rv = right_value.lower()

        if op in ("equals", "=="):
            return left_str == rv
        elif op in ("not_equals", "!="):
            return left_str != rv
        elif op == "starts_with":
            return left_str.startswith(rv)
        elif op == "contains":
            return rv in left_str
        elif op == "in":
            return left_str in {v.strip() for v in rv.split(",") if v.strip()}
        return False

    return predicate
