#!/bin/sh
# Production start. Binds every interface and the platform port (Render sets PORT).
# Local use: sh scripts/start_api.sh   (defaults to port 8000)
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT/backend"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PORT="${PORT:-8000}"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --proxy-headers \
  --forwarded-allow-ips="*"
