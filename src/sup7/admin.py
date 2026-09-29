"""HTTP admin API — lets flux7-console see and steer the supervisor.

  GET  /health      liveness
  GET  /status      running or paused, mesh reachability, counters, provider states
  GET  /config      rules, thresholds, provider chain (never secrets)
  GET  /decisions   most recent decisions with their reasoning (?limit=)
  POST /pause       stop evaluating: approvals stay pending in the mesh, for a human
  POST /resume      evaluate again
  GET  /files              editable files: sup7.yaml and the question sets, with fingerprints
  GET  /files/{id}         one file as text (token values masked), ETag = fingerprint
  PUT  /files/{id}         replace it: If-Match required ("new" to create a question set);
                           validated whole, backed up, applied without restart where possible
  POST /evaluate           judge one tool call on demand: {"agent_id", "tool", "params",
                           "recent_traces"?, ...} -> {"decision": approve|deny|escalate, ...};
                           sup7 advises, the caller enforces
  GET  /bench/sets         labelled case sets (bench.dir/sets/*.jsonl)
  GET  /bench/runs         evaluation runs, newest first; GET /bench/runs/{id} with its results
  GET  /bench/estimate     ?set=: cost of a replay, and whether a free recompute is possible
  POST /bench/runs         {"set", "mode": "recompute" | "replay"}: measure the live config
  GET  /bench/progress     the run in progress

Served in-process by uvicorn next to the poll loop. Off unless `admin.enabled`.
When `admin.token` is set, every route except /health requires
`Authorization: Bearer <token>`. Writing files always requires it: with no
token configured, PUT /files answers 403, even on loopback.
"""

from __future__ import annotations

import hmac
from typing import Protocol

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route


class AdminTarget(Protocol):
    """What the admin API needs from the runner."""

    def status(self) -> dict: ...
    def config_summary(self) -> dict: ...
    def recent_decisions(self, limit: int) -> list[dict]: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def list_files(self) -> list[dict]: ...
    def read_file(self, file_id: str) -> tuple[str, str]: ...
    def write_file(self, file_id: str, text: str, if_match: str | None, by: str) -> dict: ...
    def bench_sets(self) -> list[dict]: ...
    def bench_runs(self) -> list[dict]: ...
    def bench_run(self, run_id: str) -> dict: ...
    def bench_estimate(self, set_name: str) -> dict: ...
    def bench_start(self, set_name: str, mode: str) -> dict: ...
    def bench_progress(self) -> dict: ...
    async def evaluate_call(self, call: dict) -> dict: ...


def create_admin_app(target: AdminTarget, token: str = "") -> Starlette:
    def authorized(request: Request) -> bool:
        if not token:
            return True
        header = request.headers.get("authorization", "")
        return hmac.compare_digest(header, f"Bearer {token}")

    def guarded(handler):
        async def wrapper(request: Request):
            if not authorized(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await handler(request)
        return wrapper

    async def health(request: Request):
        return JSONResponse({"ok": True})

    async def status(request: Request):
        return JSONResponse(target.status())

    async def config(request: Request):
        return JSONResponse(target.config_summary())

    async def decisions(request: Request):
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)
        limit = max(1, min(limit, 500))
        return JSONResponse({"decisions": target.recent_decisions(limit)})

    async def pause(request: Request):
        target.pause()
        return JSONResponse(target.status())

    async def resume(request: Request):
        target.resume()
        return JSONResponse(target.status())

    def edit_error(e):
        from sup7.benchrun import BenchError
        from sup7.editing import EditError

        if isinstance(e, (EditError, BenchError)):
            return JSONResponse({"error": e.message}, status_code=e.status)
        raise e

    def answer(fn):
        try:
            return JSONResponse(fn())
        except Exception as e:
            return edit_error(e)

    async def bench_sets(request: Request):
        return answer(lambda: {"sets": target.bench_sets()})

    async def bench_runs(request: Request):
        return answer(lambda: {"runs": target.bench_runs()})

    async def bench_run(request: Request):
        return answer(lambda: target.bench_run(request.path_params["run_id"]))

    async def bench_estimate(request: Request):
        return answer(lambda: target.bench_estimate(request.query_params.get("set", "")))

    async def bench_progress(request: Request):
        return answer(target.bench_progress)

    async def evaluate(request: Request):
        try:
            call = await request.json()
        except ValueError:
            return JSONResponse({"error": 'expected JSON {"tool", "params", ...}'}, status_code=400)
        if not isinstance(call, dict):
            return JSONResponse({"error": 'expected a JSON object {"tool", "params", ...}'}, status_code=400)
        try:
            return JSONResponse(await target.evaluate_call(call))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def bench_start(request: Request):
        # a replay spends credits: starting a run needs the token, like an edit
        if not token:
            return JSONResponse({"error": "running an evaluation needs admin.token in sup7.yaml"},
                                status_code=403)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "expected JSON {\"set\", \"mode\"}"}, status_code=400)
        return answer(lambda: target.bench_start(str(body.get("set", "")), str(body.get("mode", ""))))

    async def files(request: Request):
        try:
            return JSONResponse({"files": target.list_files()})
        except Exception as e:
            return edit_error(e)

    async def read_file(request: Request):
        try:
            text, fingerprint = target.read_file(request.path_params["file_id"])
        except Exception as e:
            return edit_error(e)
        return PlainTextResponse(text, media_type="text/yaml", headers={"ETag": fingerprint})

    async def write_file(request: Request):
        if not token:
            return JSONResponse({"error": "editing needs admin.token in sup7.yaml "
                                          "(and SUP7_ADMIN_TOKEN on the console side)"}, status_code=403)
        text = (await request.body()).decode("utf-8", errors="replace")
        try:
            result = target.write_file(request.path_params["file_id"], text,
                                       request.headers.get("if-match"), by="admin-api")
        except Exception as e:
            return edit_error(e)
        return JSONResponse(result)

    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/status", guarded(status), methods=["GET"]),
        Route("/config", guarded(config), methods=["GET"]),
        Route("/decisions", guarded(decisions), methods=["GET"]),
        Route("/pause", guarded(pause), methods=["POST"]),
        Route("/resume", guarded(resume), methods=["POST"]),
        Route("/files", guarded(files), methods=["GET"]),
        Route("/files/{file_id:path}", guarded(read_file), methods=["GET"]),
        Route("/files/{file_id:path}", guarded(write_file), methods=["PUT"]),
        Route("/evaluate", guarded(evaluate), methods=["POST"]),
        Route("/bench/sets", guarded(bench_sets), methods=["GET"]),
        Route("/bench/runs", guarded(bench_runs), methods=["GET"]),
        Route("/bench/runs", guarded(bench_start), methods=["POST"]),
        Route("/bench/runs/{run_id}", guarded(bench_run), methods=["GET"]),
        Route("/bench/estimate", guarded(bench_estimate), methods=["GET"]),
        Route("/bench/progress", guarded(bench_progress), methods=["GET"]),
    ])
