"""L4 — Serving layer package.

Exposes the ASGI app factory so the process entry point can do:

    from server import create_app
    app = create_app()

or reference `server.app:app` directly for uvicorn.
"""
from __future__ import annotations

from server.app import app, create_app

__all__ = ["app", "create_app"]
