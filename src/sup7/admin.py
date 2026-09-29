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
        from sup7.editing import EditError

        if isinstance(e, EditError):
            return JSONResponse({"error": e.message}, status_code=e.status)
        raise e

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
    ])
