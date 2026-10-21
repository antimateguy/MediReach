"""Run the web app: `python -m medireach.web` (host and port from MEDIREACH_HOST / PORT)."""
from __future__ import annotations

import os

import uvicorn

from medireach.web.server import create_app


def main() -> None:
    host = os.environ.get("MEDIREACH_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
