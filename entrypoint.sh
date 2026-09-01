#!/bin/bash
set -e

TOKEN_FILE="${TOKEN_FILE:-/root/.config/open-webui-proxy/token.json}"
PROFILE_DIR="${BROWSER_PROFILE_DIR:-/root/.config/open-webui-proxy/browser-profile}"

mkdir -p "$(dirname "$TOKEN_FILE")" "$PROFILE_DIR"

echo "Checking for existing token..."
if [ ! -f "$TOKEN_FILE" ]; then
    echo "No token found. Launching browser login..."
    echo "Approve Duo push on your phone when prompted."
    python /app/playwright_login.py
    if [ ! -f "$TOKEN_FILE" ]; then
        echo "ERROR: Login failed. No token file created." >&2
        exit 1
    fi
    echo "Login complete. Token saved to $TOKEN_FILE"
fi

echo "Starting cron daemon..."
cron

echo "Starting proxy server..."
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}"
