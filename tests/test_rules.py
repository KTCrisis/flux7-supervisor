"""Tests for rule condition parsing and evaluation."""

import pytest

from sup7.models import ApprovalContext
from sup7.rules import parse_condition


def _ctx(**kwargs) -> ApprovalContext:
    defaults = {"id": "test-1", "agent_id": "agent", "tool": "filesystem.read_file"}
    defaults.update(kwargs)
    return ApprovalContext(**defaults)


class TestParseCondition:
    def test_tool_contains(self):
        pred = parse_condition("tool contains read")
        assert pred(_ctx(tool="filesystem.read_file"), [])
        assert not pred(_ctx(tool="filesystem.write_file"), [])

    def test_tool_equals(self):
        pred = parse_condition("tool equals filesystem.write_file")
        assert pred(_ctx(tool="filesystem.write_file"), [])
        assert not pred(_ctx(tool="filesystem.read_file"), [])

    def test_tool_not_equals(self):
        pred = parse_condition("tool not_equals filesystem.delete")
        assert pred(_ctx(tool="filesystem.read_file"), [])
        assert not pred(_ctx(tool="filesystem.delete"), [])

    def test_params_path_starts_with(self):
        pred = parse_condition("params.path starts_with /home/user/project")
        ctx = _ctx(params={"path": "/home/user/project/src/main.py"})
        assert pred(ctx, [])

    def test_params_path_starts_with_project_dir(self):
        pred = parse_condition("params.path starts_with project_dir")
        ctx = _ctx(params={"path": "/home/fluxart/flux7-mesh/main.go"})
        assert pred(ctx, ["/home/fluxart/flux7-mesh", "/home/fluxart/flux7-console"])
        assert not pred(ctx, ["/opt/other"])

    def test_injection_risk_equals(self):
        pred = parse_condition("injection_risk == true")
        assert pred(_ctx(injection_risk=True), [])
        assert not pred(_ctx(injection_risk=False), [])

    def test_operator_double_equals(self):
        pred = parse_condition("tool == filesystem.read_file")
        assert pred(_ctx(tool="filesystem.read_file"), [])

    def test_operator_not_equals_symbol(self):
        pred = parse_condition("tool != filesystem.delete")
        assert pred(_ctx(tool="filesystem.read_file"), [])

    def test_missing_field_returns_false(self):
        pred = parse_condition("params.nonexistent contains foo")
        assert not pred(_ctx(), [])

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            parse_condition("invalid")

    def test_no_operator_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            parse_condition("tool")


class TestHardening:
    """Two approval rules were broader than they read (2026-09-29)."""

    @pytest.mark.parametrize("path, ok", [
        ("/home/fluxart/flux7-mesh/main.go", True),
        ("/home/fluxart/flux7-mesh", True),
        ("~/flux7-mesh/docs/a.md", True),
        ("/home/fluxart/flux7-mesh/../.bashrc", False),      # traversal out of the project
        ("/home/fluxart/flux7-mesh-backup/x", False),        # sibling with the same prefix
        ("flux7-mesh/main.go", False),                       # relative: no anchor
        ("/home/fluxart/flux7-mesh/./a/../b.go", True),
    ])
    def test_project_dir_is_normalised(self, path, ok, monkeypatch):
        monkeypatch.setenv("HOME", "/home/fluxart")
        pred = parse_condition("params.path starts_with project_dir")
        assert pred(_ctx(params={"path": path}), ["/home/fluxart/flux7-mesh/"]) is ok

    def test_in_matches_exact_names_only(self):
        pred = parse_condition("tool in filesystem.read_file, filesystem.list_directory")
        assert pred(_ctx(tool="filesystem.read_file"), [])
        assert pred(_ctx(tool="filesystem.list_directory"), [])
        assert not pred(_ctx(tool="gmail.gmail_mark_as_read"), [])
        assert not pred(_ctx(tool="filesystem.read_file_and_delete"), [])

    def test_contains_read_is_a_substring_test(self):
        # documented pitfall: kept as is, but no longer used for approvals in the examples
        assert parse_condition("tool contains read")(_ctx(tool="gmail.gmail_mark_as_read"), [])
