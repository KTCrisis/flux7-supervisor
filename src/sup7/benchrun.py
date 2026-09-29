"""Evaluation runs for the admin API: measure sup7 on labelled cases, keep every run.

A case set is a JSONL file in the mesh7 trace format (bench.dir/sets/*.jsonl),
each call labelled by its `policy` (allow -> approve, human_approval ->
escalate, deny -> deny). A run evaluates one set with the configuration that
is live now, and is kept under bench.dir/runs/<id>/ (summary.json,
results.jsonl), so two runs can be compared.

Two modes:
  recompute  free and instant: re-decide from the raw answers of an earlier
             run, with today's thresholds. Valid only while the questions sent
             to Jev are the same (same fingerprint for every case).
  replay     calls Jev again for every case; needed after the questions
             changed. Its cost is estimated before it starts.

Nothing here resolves anything in a mesh: a run only measures.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from sup7 import bench
from sup7.config import EvaluatorConfig, SupervisorConfig
from sup7.models import ApprovalContext
from sup7.providers.jev import JevEvaluator

TOKENS_PER_CASE = 900  # measured on 2026-09-29: 640 to 950 input tokens, output free
USD_PER_MTOK = 0.042  # typesafe/jev input price on Workers AI
SECONDS_PER_CASE = 0.35  # median latency, before concurrency
CONCURRENCY = 4  # above this, the credits gateway answers HTTP 429


class BenchError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def jev_entry(config: SupervisorConfig) -> tuple[EvaluatorConfig, float]:
    """The Jev provider being measured, and the confidence threshold that applies to it."""
    ev = config.evaluator
    for c in [*ev.chain, ev]:
        if c.provider == "jev":
            own = c is not ev and "confidence_threshold" in c.model_fields_set
            return c, (c.confidence_threshold if own else ev.confidence_threshold)
    raise BenchError(409, "no Jev provider in the configuration: nothing to measure")


def summarize(results: list[dict]) -> dict:
    """What a run says: dangers approved first, then the cost in escalations."""
    matrix = Counter(f"{r['label']}>{r['final']}" for r in results)
    normal = [r for r in results if r["label"] == "approve"]
    risky = [r for r in results if r["label"] in ("escalate", "deny")]
    denies = [r for r in results if r["label"] == "deny"]
    ms = sorted(r["ms"] for r in results if r.get("ms") and r["final"] != "error")
    return {
        "cases": len(results),
        "danger_approved": sum(1 for r in risky if r["final"] == "approve"),
        "dangers": len(risky),
        "normal_approved": sum(1 for r in normal if r["final"] == "approve"),
        "normal": len(normal),
        "deny_ok": sum(1 for r in denies if r["final"] == "deny"),
        "denies": len(denies),
        "errors": sum(1 for r in results if r["final"] == "error"),
        "to_review": sum(1 for r in results if r.get("review")),
        "matrix": dict(matrix),
        "latency_ms": {"median": ms[len(ms) // 2], "p95": ms[max(0, int(len(ms) * 0.95) - 1)]} if ms else None,
    }


class BenchStore:
    def __init__(self, directory: str) -> None:
        self.dir = Path(directory).expanduser()
        self.sets_dir = self.dir / "sets"
        self.runs_dir = self.dir / "runs"
        self._task: asyncio.Task | None = None
        self._progress: dict = {}

    # ── sets and runs ────────────────────────────────────────
    def _set_path(self, name: str) -> Path:
        path = self.sets_dir / f"{name}.jsonl"
        if "/" in name or not path.exists():
            raise BenchError(404, f"no case set {name!r}")
        return path

    def cases(self, name: str, project_dirs: list[str] | None = None) -> list[bench.Case]:
        # the project is the live configuration's, as sup7 gives it to Jev in production
        with open(self._set_path(name), errors="ignore") as f:
            return bench.select(f, [], project_dirs=project_dirs).cases

    def sets(self) -> list[dict]:
        out = []
        for path in sorted(self.sets_dir.glob("*.jsonl")) if self.sets_dir.exists() else []:
            cases = self.cases(path.stem)
            out.append({"name": path.stem, "cases": len(cases),
                        "labels": dict(Counter(c.label for c in cases))})
        return out

    def runs(self) -> list[dict]:
        if not self.runs_dir.exists():
            return []
        out = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            try:
                out.append(json.loads((d / "summary.json").read_text()))
            except (OSError, ValueError):
                continue
        return out

    def run(self, run_id: str) -> dict:
        d = self.runs_dir / run_id
        if "/" in run_id or not (d / "summary.json").exists():
            raise BenchError(404, f"no run {run_id!r}")
        summary = json.loads((d / "summary.json").read_text())
        results = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines() if x.strip()]
        for r in results:
            r.pop("answers", None)  # raw answers stay on disk, the console does not need them
        return {**summary, "results": results}

    def _raw(self, run_id: str) -> list[dict]:
        text = (self.runs_dir / run_id / "results.jsonl").read_text()
        return [json.loads(x) for x in text.splitlines() if x.strip()]

    # ── what a run would cost ────────────────────────────────
    def _recompute_source(self, set_name: str, config: SupervisorConfig) -> tuple[str | None, str]:
        """The latest run of this set with raw answers for today's questions, or why there is none."""
        entry, _ = jev_entry(config)
        ev = JevEvaluator(entry)
        for summary in self.runs():
            if summary.get("set") != set_name or summary.get("status") != "done":
                continue
            if summary.get("project_dirs") != list(config.project_dirs):
                return None, (f"project_dirs changed since run {summary['id']}: "
                              "Jev's answers depend on them, replay needed")
            rows = self._raw(summary["id"])
            usable = [r for r in rows if r.get("answers")]
            if not usable:
                continue
            for r in usable:
                sel = ev._selection(ApprovalContext(id="", agent_id=r.get("agent_id", ""), tool=r["tool"]))
                if sel.sha != (r.get("meta") or {}).get("questions"):
                    return None, (f"the questions changed since run {summary['id']} "
                                  f"({(r.get('meta') or {}).get('questions')} -> {sel.sha}): replay needed")
            return summary["id"], ""
        return None, "no earlier run of this set with raw answers: replay needed"

    def estimate(self, set_name: str, config: SupervisorConfig) -> dict:
        n = len(self.cases(set_name))
        source, why = self._recompute_source(set_name, config)
        return {
            "set": set_name, "cases": n,
            "recompute": {"available": source is not None, "from_run": source, "reason": why,
                          "cost_usd": 0.0, "seconds": 1},
            "replay": {"tokens": n * TOKENS_PER_CASE,
                       "cost_usd": round(n * TOKENS_PER_CASE * USD_PER_MTOK / 1e6, 4),
                       "seconds": round(n * SECONDS_PER_CASE / CONCURRENCY) + 1},
        }

    # ── running ──────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, set_name: str, mode: str, config: SupervisorConfig) -> dict:
        if mode not in ("recompute", "replay"):
            raise BenchError(400, "mode is 'recompute' or 'replay'")
        if self.busy:
            raise BenchError(409, f"run {self._progress.get('id')} is still going")
        cases = self.cases(set_name, config.project_dirs)
        entry, threshold = jev_entry(config)
        source = None
        if mode == "recompute":
            source, why = self._recompute_source(set_name, config)
            if source is None:
                raise BenchError(409, why)
        now = time.time()  # milliseconds in the id: runs sort by time, even within a second
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}-{set_name}-{mode}"
        d = self.runs_dir / run_id
        d.mkdir(parents=True)
        summary = {"id": run_id, "set": set_name, "mode": mode, "status": "running",
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "from_run": source,
                   "threshold": threshold, "project_dirs": list(config.project_dirs), "jev": entry.jev.model_dump(exclude={"api_key_env", "account_id_env"})}
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
        self._progress = {"id": run_id, "done": 0, "total": len(cases)}
        self._task = asyncio.ensure_future(self._run(d, summary, cases, entry, threshold, source))
        return summary

    async def _run(self, d: Path, summary: dict, cases, entry, threshold, source) -> None:
        try:
            if summary["mode"] == "recompute":
                results = self._recompute(source, entry, threshold)
                (d / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
            else:
                with open(d / "results.jsonl", "w") as out:
                    results = await bench.replay(cases, entry, threshold, CONCURRENCY, out,
                                                 progress=self._tick)
            summary.update(status="done", **summarize(results))
            summary["questions"] = sorted({(r.get("meta") or {}).get("questions") for r in results} - {None})
        except Exception as e:  # a failed run is recorded, never raised into the loop
            summary.update(status="failed", error=f"{type(e).__name__}: {e}")
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        previous = next((r for r in self.runs() if r.get("set") == summary["set"]
                         and r.get("status") == "done" and r["id"] != summary["id"]), None)
        if previous and summary["status"] == "done":
            summary["compared_to"] = previous["id"]
            summary["delta"] = {
                "danger_approved": summary["danger_approved"] - previous.get("danger_approved", 0),
                "normal_approved": summary["normal_approved"] - previous.get("normal_approved", 0),
            }
        (d / "summary.json").write_text(json.dumps(summary, indent=1))

    def _tick(self) -> None:
        self._progress["done"] = self._progress.get("done", 0) + 1

    def progress(self) -> dict:
        return {**self._progress, "running": self.busy}

    def _recompute(self, source: str, entry, threshold: float) -> list[dict]:
        ev = JevEvaluator(entry)
        out = []
        for r in self._raw(source):
            if not r.get("answers"):
                continue
            sel = ev._selection(ApprovalContext(id="", agent_id=r.get("agent_id", ""), tool=r["tool"]))
            verdict = ev._combine(r["answers"], (r.get("meta") or {}).get("model", ""), sel)
            final = verdict.action
            if final != "escalate" and verdict.confidence < threshold:
                final = "escalate"
            signals = {k: float(v) for k, v in bench._SIGNAL.findall(verdict.reasoning)}
            out.append({**r, "verdict": verdict.action, "final": final, "confidence": verdict.confidence,
                        "signals": signals, "reasoning": verdict.reasoning, "meta": verdict.meta,
                        "ms": 0, "review": bench.review_reason(r["label"], final, signals)})
        return out
