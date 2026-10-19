#!/bin/bash
# Start the StepLens server (port 8781; 8765 is used by another app on this machine).
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn steplens.server:app --host 127.0.0.1 --port "${PORT:-8781}"
