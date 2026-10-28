#!/usr/bin/env bash
# Start the Falsify web app. Used by PM2:  pm2 start deploy/run-web.sh --name falsify-web --interpreter bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec .venv/bin/uvicorn falsify.web:app --host 127.0.0.1 --port "${FALSIFY_PORT:-8097}" --proxy-headers --workers 1
