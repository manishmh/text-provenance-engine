"""Vercel ASGI entrypoint for the existing Text Provenance API.

Vercel sends requests for this function below ``/api``.  The product's API
contract intentionally remains ``/v1/*``, ``/health``, and ``/ready`` inside
the application, so this small ASGI adapter removes only that deployment
prefix.  It delegates lifespan, middleware, routes, and repository startup to
the normal application factory; no API construction is duplicated here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from provenance.api.app import create_app

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]


class ApiPrefixAdapter:
    """Expose an ASGI app below ``/api`` while preserving its route contract."""

    def __init__(self, application: Callable[..., Awaitable[None]], prefix: str = "/api") -> None:
        self.application = application
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        # Lifespan messages have no URL path and must reach the factory-built
        # application unchanged so repository initialization still runs.
        if scope["type"] not in {"http", "websocket"}:
            await self.application(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        if path == self.prefix:
            inner_path = "/"
        elif path.startswith(self.prefix + "/"):
            inner_path = path[len(self.prefix):]
        else:
            # ``vercel dev`` and production function dispatch both preserve
            # the request path.  Keep this fallback useful for direct ASGI
            # tests without accepting an unrelated prefix as a valid route.
            inner_path = path

        inner_scope = dict(scope)
        inner_scope["path"] = inner_path
        raw_path = scope.get("raw_path")
        if isinstance(raw_path, bytes) and raw_path.startswith(self.prefix.encode()):
            inner_scope["raw_path"] = raw_path[len(self.prefix):] or b"/"
        inner_scope["root_path"] = f"{scope.get('root_path', '')}{self.prefix}"
        await self.application(inner_scope, receive, send)


# Vercel detects an ASGI object named ``app`` in api/index.py.
app = ApiPrefixAdapter(create_app())
