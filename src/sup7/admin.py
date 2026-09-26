"""HTTP admin API — lets flux7-console see and steer the supervisor.

  GET  /health      liveness
  GET  /status      running or paused, mesh reachability, counters, provider states
  GET  /config      rules, thresholds, provider chain (never secrets)
  GET  /decisions   most recent decisions with their reasoning (?limit=)
  POST /pause       stop evaluating: approvals stay pending in the mesh, for a human
  POST /resume      evaluate again

Served in-process by uvicorn next to the poll loop. Off unless `admin.enabled`.
When `admin.token` is set, every route except /health requires
`Authorization: Bearer <token>`.
"""

from __future__ import annotations

import hmac
from typing import Protocol

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


class AdminTarget(Protocol):
    """What the admin API needs from the runner."""

    def status(self) -> dict: ...
    def config_summary(self) -> dict: ...
    def recent_decisions(self, limit: int) -> list[dict]: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


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

    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/status", guarded(status), methods=["GET"]),
        Route("/config", guarded(config), methods=["GET"]),
        Route("/decisions", guarded(decisions), methods=["GET"]),
        Route("/pause", guarded(pause), methods=["POST"]),
        Route("/resume", guarded(resume), methods=["POST"]),
    ])
