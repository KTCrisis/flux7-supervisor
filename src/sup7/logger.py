"""JSONL decision logger."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO

from sup7.models import Decision

logger = logging.getLogger(__name__)


class DecisionLogger:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._file: IO[str] | None = None

    def open(self) -> None:
        self._file = open(self._path, "a")
        logger.info("decision log: %s", self._path)

    def log(self, decision: Decision) -> None:
        if self._file is None:
            return
        record = {
            "timestamp": decision.timestamp.isoformat(),
            "approval_id": decision.approval_id,
            "agent_id": decision.agent_id,
            "tool": decision.tool,
            "decision": decision.decision,
            "rule_matched": decision.rule_matched,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "evaluation_ms": decision.evaluation_ms,
            "injection_risk": decision.injection_risk,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
