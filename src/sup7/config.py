"""Supervisor configuration — YAML loading and Pydantic validation."""

from __future__ import annotations

import re
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def _parse_duration(value: str) -> float:
    """Parse a duration string like '2s', '500ms', '1m' to seconds."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", value.strip())
    if not match:
        raise ValueError(f"invalid duration: {value!r}")
    num, unit = float(match.group(1)), match.group(2)
    return num * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


class MeshConfig(BaseModel):
    url: str = "http://localhost:9090"
    agent_id: str = "supervisor"


class MemoryConfig(BaseModel):
    url: str = "http://localhost:9070"
    token: str = ""
    enabled: bool = False
    store_decisions: bool = True
    recall_on_start: bool = True
    recall_limit: int = 20
    tags: list[str] = Field(default_factory=lambda: ["supervisor", "decision"])


class JevConfig(BaseModel):
    """Jev (TypeSafe AI) settings; used when evaluator.provider is "jev"."""

    backend: Literal["cloudflare", "typesafe"] = "cloudflare"
    model: str = ""  # default: typesafe/jev (cloudflare) or jev-latest (typesafe)
    url: str = ""  # default: the backend's public endpoint
    api_key_env: str = "CLOUDFLARE_API_TOKEN"  # TYPESAFE_API_KEY for backend typesafe
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID"  # cloudflare only
    injection_max: float = 0.5  # above: escalate
    destructive_max: float = 0.2  # above: never auto-approve
    in_scope_min: float = 0.7  # below: never auto-approve
    deny_in_scope_max: float = 0.7  # deny needs the call to be this out of scope (in_scope below)
    deny_min: float = 0.9  # deny only when a harm is this probable and the call is out of scope
    project_min: float = 0.7  # above: the call acts in the project, overwriting is not a harm
    redact_params: list[str] = Field(default_factory=list)  # param names never sent


class EvaluatorConfig(BaseModel):
    provider: Literal["ollama", "anthropic", "claude-code", "jev", "none"] = "ollama"  # none: rules only
    model: str = "qwen3:14b"
    url: str = "http://localhost:11434"
    timeout: float = 30.0
    callback_timeout: float = 120.0
    confidence_threshold: float = 0.8
    jev: JevConfig = Field(default_factory=JevConfig)
    # Provider chain: tried in order, next one only when the previous fails
    # (no answer). Empty = the single provider above. Each entry is a full
    # evaluator config; confidence_threshold stays the top-level one.
    chain: list["EvaluatorConfig"] = Field(default_factory=list)
    breaker_failures: int = 3  # consecutive failures before a provider is skipped
    breaker_cooldown: float = 300.0  # seconds a tripped provider stays skipped
    system_prompt: str = (
        "You are a supervisor agent evaluating tool call approval requests. "
        "You receive a JSON description of a pending approval including the tool name, "
        "parameters, the agent's recent activity, and active grants.\n\n"
        "Respond with EXACTLY one line in this format:\n"
        "DECISION: <APPROVE|DENY|ESCALATE> | CONFIDENCE: <0.0-1.0> | REASONING: <one sentence>\n\n"
        "Guidelines:\n"
        "- APPROVE if the action is routine and low-risk\n"
        "- DENY if the action is clearly dangerous\n"
        "- ESCALATE if you are unsure or the action is high-stakes\n"
        "- Base your decision on STRUCTURAL properties (paths, tool names, patterns)\n"
        "- Be conservative: when in doubt, ESCALATE"
    )


EvaluatorConfig.model_rebuild()

class MCPServerConfig(BaseModel):
    enabled: bool = True
    transport: Literal["stdio", "http"] = "stdio"
    port: int = 9095



class AdminConfig(BaseModel):
    """HTTP admin API (status, config, recent decisions, pause/resume).

    Off by default. Binds to loopback; set a token before binding elsewhere.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9096
    token: str = ""  # required as "Authorization: Bearer <token>" when set
    recent_decisions: int = 200  # kept in memory for GET /decisions

class PollConfig(BaseModel):
    interval: float = 2.0
    tool_scopes: list[str] = Field(default_factory=list)

    @field_validator("interval", mode="before")
    @classmethod
    def parse_interval(cls, v: str | float | int) -> float:
        if isinstance(v, str):
            return _parse_duration(v)
        return float(v)


class RuleEntry(BaseModel):
    name: str
    condition: str | None = None
    action: Literal["approve", "deny", "escalate"] = "escalate"
    confidence: float = 0.9
    description: str | None = None


class SupervisorConfig(BaseModel):
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    mcp_server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    poll: PollConfig = Field(default_factory=PollConfig)
    rules: list[RuleEntry] = Field(default_factory=list)
    project_dirs: list[str] = Field(default_factory=list)
    decision_log: str = "sup7-decisions.jsonl"

    def model_post_init(self, __context: object) -> None:
        if not self.rules or self.rules[-1].condition is not None:
            self.rules.append(
                RuleEntry(name="default", action="escalate", confidence=1.0)
            )


def load_config(path: str) -> SupervisorConfig:
    """Load supervisor config from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "supervisor" in data:
        data = data["supervisor"]
    return SupervisorConfig(**data)
