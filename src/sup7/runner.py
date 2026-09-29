"""Supervisor main loop — poll, evaluate, resolve, log."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections import deque

from mesh7 import AgentMesh

from sup7.config import SupervisorConfig
from sup7.evaluator import RuleEvaluator
from sup7.logger import DecisionLogger
from sup7 import __version__
from sup7.models import ApprovalContext

logger = logging.getLogger(__name__)


class SupervisorRunner:
    """Polls flux7-mesh for pending approvals and resolves them.

    Expects flux7-mesh (mesh7 serve) to be running independently.
    Optionally stores/recalls decisions via flux7-memory.
    """

    def __init__(self, config: SupervisorConfig) -> None:
        self._config = config
        self._mesh = AgentMesh(
            url=config.mesh.url,
            agent=config.mesh.agent_id,
        )
        self._evaluator = RuleEvaluator(config)
        self._logger = DecisionLogger(config.decision_log)
        self._shutdown = False
        self._seen_escalated: set[str] = set()
        self._semaphore = asyncio.Semaphore(10)
        self._mesh_alive = False

        # Admin API state
        self._paused = False
        self._started_at = time.time()
        self._recent: deque[dict] = deque(maxlen=config.admin.recent_decisions)
        self._counts = {"approved": 0, "denied": 0, "escalated": 0}
        self._last_decision_at: str | None = None
        self._admin_server = None
        self._admin_task: asyncio.Task | None = None

        self._mem7 = None
        if config.memory.enabled:
            from mem7 import Mem7

            self._mem7 = Mem7(config.memory.url, token=config.memory.token)

    async def start(self) -> None:
        self._logger.open()
        logger.info(
            "sup7 starting — mesh=%s agent=%s interval=%.1fs provider=%s",
            self._config.mesh.url,
            self._config.mesh.agent_id,
            self._config.poll.interval,
            " > ".join(c.provider for c in self._config.evaluator.chain) or self._config.evaluator.provider,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.shutdown)

        await self._start_admin()
        await self._wait_for_mesh()

        if self._config.evaluator.provider == "claude-code":
            await self._start_mcp_server()

        await self._recall_memory()

        try:
            await self._run()
        finally:
            await self._stop_admin()
            await self._evaluator.close()
            self._logger.close()
            logger.info("sup7 stopped")

    async def _wait_for_mesh(self) -> None:
        if self._mesh.is_healthy():
            logger.info("flux7-mesh connected")
            self._mesh_alive = True
            return

        logger.info("waiting for flux7-mesh (mesh7 serve) ...")
        while not self._shutdown:
            await asyncio.sleep(self._config.poll.interval)
            if self._mesh.is_healthy():
                logger.info("flux7-mesh connected")
                self._mesh_alive = True
                return

    async def _start_mcp_server(self) -> None:
        if not self._config.mcp_server.enabled:
            return

        from sup7.mcp_server import set_evaluator
        from sup7.providers.claude_code import ClaudeCodeEvaluator

        if isinstance(self._evaluator._llm, ClaudeCodeEvaluator):
            set_evaluator(self._evaluator._llm)
            logger.info("MCP server configured for Claude Code callback")

    async def _recall_memory(self) -> None:
        if self._mem7 is None or not self._mesh_alive:
            return

        try:
            tags = self._config.memory.tags
            result = self._mem7.recall(
                tags=tags,
                agent=self._config.mesh.agent_id,
                limit=self._config.memory.recall_limit,
            )
            if result:
                logger.info("recalled decisions from memory")
        except Exception:
            logger.debug("memory recall failed (mem7 may not be ready)")

    async def _run(self) -> None:
        while not self._shutdown:
            if self._paused:
                # approvals stay pending in the mesh, where a human decides
                await asyncio.sleep(self._config.poll.interval)
                continue
            try:
                pending = self._poll()
                if pending:
                    await self._process_batch(pending)
            except Exception:
                logger.exception("unexpected error in poll loop")

            await asyncio.sleep(self._config.poll.interval)

    def _poll(self) -> list[dict]:
        seen_ids: set[str] = set()
        results: list[dict] = []

        scopes = self._config.poll.tool_scopes or [None]
        for scope in scopes:
            try:
                approvals = self._mesh.pending(tool_scope=scope)
            except Exception:
                if self._mesh_alive:
                    logger.warning("flux7-mesh connection lost")
                    self._mesh_alive = False
                continue

            if not self._mesh_alive:
                logger.info("flux7-mesh reconnected")
            self._mesh_alive = True

            for a in approvals:
                aid = a.get("id", "")
                if aid and aid not in seen_ids and aid not in self._seen_escalated:
                    seen_ids.add(aid)
                    results.append(a)

        return results

    async def _process_batch(self, approvals: list[dict]) -> None:
        tasks = [self._process_one(a) for a in approvals]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_one(self, summary: dict) -> None:
        async with self._semaphore:
            approval_id = summary.get("id", "")
            tool = summary.get("tool", "")

            try:
                detail = self._mesh.approval_detail(approval_id)
            except Exception:
                logger.warning("could not fetch detail for %s", approval_id)
                return

            context = ApprovalContext(
                id=approval_id,
                agent_id=detail.get("agent_id", ""),
                tool=detail.get("tool", tool),
                params=detail.get("params", {}),
                policy_rule=detail.get("policy_rule", ""),
                injection_risk=detail.get("injection_risk", False),
                recent_traces=detail.get("recent_traces", []),
                active_grants=detail.get("active_grants", []),
            )

            decision = await self._evaluator.evaluate(context)
            resolved_by = f"supervisor:{self._config.mesh.agent_id}"

            if decision.decision == "approved":
                ok = self._mesh.resolve(
                    approval_id, "approve",
                    resolved_by=resolved_by,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence,
                )
                if ok:
                    logger.info("approved %s (%s) — %s", approval_id, tool, decision.reasoning)

            elif decision.decision == "denied":
                ok = self._mesh.resolve(
                    approval_id, "deny",
                    resolved_by=resolved_by,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence,
                )
                if ok:
                    logger.info("denied %s (%s) — %s", approval_id, tool, decision.reasoning)

            else:
                self._seen_escalated.add(approval_id)
                logger.info("escalated %s (%s) — %s", approval_id, tool, decision.reasoning)

            self._logger.log(decision)
            self._remember(decision)
            self._store_decision(decision)

    def _store_decision(self, decision) -> None:
        if self._mem7 is None or not self._config.memory.store_decisions:
            return

        try:
            key = f"supervisor:decision:{decision.approval_id}"
            value = json.dumps({
                "approval_id": decision.approval_id,
                "agent_id": decision.agent_id,
                "tool": decision.tool,
                "decision": decision.decision,
                "rule_matched": decision.rule_matched,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat(),
                "evaluator": decision.evaluator,
            })
            tags = self._config.memory.tags + [decision.decision, decision.tool.split(".")[0]]
            self._mem7.store(key, value, tags=tags, agent=self._config.mesh.agent_id)
        except Exception:
            logger.debug("failed to store decision in memory")

    # ── admin API ─────────────────────────────────────────────
    async def _start_admin(self) -> None:
        cfg = self._config.admin
        if not cfg.enabled:
            return
        import uvicorn

        from sup7.admin import create_admin_app

        class _Server(uvicorn.Server):
            # sup7 owns SIGINT/SIGTERM; uvicorn must not replace the handlers
            def capture_signals(self):  # type: ignore[override]
                from contextlib import nullcontext
                return nullcontext()

        app = create_admin_app(self, token=cfg.token)
        self._admin_server = _Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning"))
        self._admin_task = asyncio.create_task(self._admin_server.serve())
        logger.info("admin API on http://%s:%d", cfg.host, cfg.port)

    async def _stop_admin(self) -> None:
        if self._admin_server is None or self._admin_task is None:
            return
        self._admin_server.should_exit = True
        try:
            await asyncio.wait_for(self._admin_task, timeout=5)
        except Exception:
            self._admin_task.cancel()

    def _remember(self, decision) -> None:
        self._counts[decision.decision] = self._counts.get(decision.decision, 0) + 1
        self._last_decision_at = decision.timestamp.isoformat()
        self._recent.appendleft({
            "timestamp": decision.timestamp.isoformat(),
            "approval_id": decision.approval_id,
            "agent_id": decision.agent_id,
            "tool": decision.tool,
            "decision": decision.decision,
            "rule_matched": decision.rule_matched,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "evaluation_ms": decision.evaluation_ms,
            "evaluator": decision.evaluator,
        })

    def _providers(self) -> list[dict]:
        ev = self._config.evaluator
        llm = self._evaluator._llm
        described = ev.chain or [ev]
        states = llm.status() if hasattr(llm, "status") else [
            {"name": ev.provider, "state": "ok", "consecutive_failures": 0, "skipped_for_s": 0}
        ]
        out = []
        for cfg, st in zip(described, states):
            detail = cfg.jev.backend if cfg.provider == "jev" else cfg.model
            out.append({**st, "detail": detail})
        return out

    def status(self) -> dict:
        return {
            "version": __version__,
            "state": "paused" if self._paused else "running",
            "uptime_s": round(time.time() - self._started_at),
            "mesh": {"url": self._config.mesh.url, "reachable": self._mesh_alive},
            "memory": {"enabled": self._mem7 is not None},
            "evaluator": {
                "mode": "chain" if self._config.evaluator.chain else "single",
                "providers": self._providers() if self._evaluator._llm else [],
            },
            "decisions": {**self._counts, "last_at": self._last_decision_at},
        }

    def config_summary(self) -> dict:
        ev = self._config.evaluator

        def provider(cfg):
            # the threshold that applies to this provider's verdicts: its own when set
            own = cfg is not ev and "confidence_threshold" in cfg.model_fields_set
            d = {"provider": cfg.provider,
                 "confidence_threshold": cfg.confidence_threshold if own else ev.confidence_threshold}
            if cfg.provider == "jev":
                j = cfg.jev
                d.update(backend=j.backend, destructive_max=j.destructive_max, in_scope_min=j.in_scope_min,
                         deny_in_scope_max=j.deny_in_scope_max, project_min=j.project_min,
                         injection_max=j.injection_max, deny_min=j.deny_min, redact_params=j.redact_params)
            elif cfg.provider in ("ollama", "anthropic"):
                d["model"] = cfg.model
            return d

        return {
            "rules": [
                {"name": r.name, "condition": r.condition, "action": r.action, "confidence": r.confidence}
                for r in self._config.rules
            ],
            "evaluator": {
                "confidence_threshold": ev.confidence_threshold,
                "breaker_failures": ev.breaker_failures,
                "breaker_cooldown_s": ev.breaker_cooldown,
                "providers": [provider(c) for c in (ev.chain or [ev])],
            },
            "poll_interval_s": self._config.poll.interval,
            "project_dirs": self._config.project_dirs,
        }

    def recent_decisions(self, limit: int) -> list[dict]:
        return list(self._recent)[:limit]

    def pause(self) -> None:
        if not self._paused:
            logger.info("paused by admin API: approvals now go to humans")
        self._paused = True

    def resume(self) -> None:
        if self._paused:
            logger.info("resumed by admin API")
        self._paused = False

    def shutdown(self) -> None:
        logger.info("shutdown requested")
        self._shutdown = True
