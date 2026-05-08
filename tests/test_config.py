"""Tests for supervisor configuration."""

import pytest
import yaml

from sup7.config import SupervisorConfig, _parse_duration, load_config


class TestParseDuration:
    def test_seconds(self):
        assert _parse_duration("2s") == 2.0

    def test_milliseconds(self):
        assert _parse_duration("500ms") == 0.5

    def test_minutes(self):
        assert _parse_duration("1m") == 60.0

    def test_hours(self):
        assert _parse_duration("1h") == 3600.0

    def test_invalid(self):
        with pytest.raises(ValueError, match="invalid duration"):
            _parse_duration("10x")


class TestSupervisorConfig:
    def test_defaults(self):
        cfg = SupervisorConfig()
        assert cfg.mesh.url == "http://localhost:9090"
        assert cfg.mesh.agent_id == "supervisor"
        assert cfg.evaluator.provider == "ollama"
        assert cfg.evaluator.confidence_threshold == 0.8
        assert cfg.poll.interval == 2.0

    def test_catch_all_appended(self):
        cfg = SupervisorConfig(rules=[])
        assert len(cfg.rules) == 1
        assert cfg.rules[-1].name == "default"
        assert cfg.rules[-1].action == "escalate"

    def test_catch_all_not_duplicated(self):
        from sup7.config import RuleEntry

        cfg = SupervisorConfig(rules=[RuleEntry(name="catch-all", action="escalate")])
        assert len(cfg.rules) == 1

    def test_poll_interval_duration_string(self):
        cfg = SupervisorConfig(poll={"interval": "500ms"})
        assert cfg.poll.interval == 0.5

    def test_evaluator_providers(self):
        for provider in ("ollama", "anthropic", "claude-code"):
            cfg = SupervisorConfig(evaluator={"provider": provider})
            assert cfg.evaluator.provider == provider


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "sup7.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"url": "http://mesh:9090", "agent_id": "test-sup"},
            "evaluator": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            "poll": {"interval": "3s"},
            "rules": [
                {"name": "reads", "condition": "tool contains read", "action": "approve"},
            ],
        }))

        cfg = load_config(str(config_file))
        assert cfg.mesh.url == "http://mesh:9090"
        assert cfg.mesh.agent_id == "test-sup"
        assert cfg.evaluator.provider == "anthropic"
        assert cfg.poll.interval == 3.0
        assert len(cfg.rules) == 2  # reads + auto catch-all

    def test_load_nested_supervisor_key(self, tmp_path):
        config_file = tmp_path / "sup7.yaml"
        config_file.write_text(yaml.dump({
            "supervisor": {
                "mesh": {"agent_id": "nested"},
            }
        }))

        cfg = load_config(str(config_file))
        assert cfg.mesh.agent_id == "nested"
