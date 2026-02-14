"""Run the FastAPI backend for the React UI."""

from __future__ import annotations

import argparse

import uvicorn

from src.ui_api.server import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live simulation API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

