#!/bin/bash
set -euo pipefail

TOKEN_FILE="${TOKEN_FILE:-/data/token.json}"
PROFILE_DIR="${BROWSER_PROFILE_DIR:-/data/browser-profile}"

mkdir -p "$(dirname "$TOKEN_FILE")" "$PROFILE_DIR"

die() {
    echo "" >&2
    echo "──────────────────────────────────────────────────────────────" >&2
    echo " Setup required" >&2
    echo "──────────────────────────────────────────────────────────────" >&2
    printf '%s\n' "$@" >&2
    echo "──────────────────────────────────────────────────────────────" >&2
    echo "" >&2
    exit 1
}

if [ -z "${OPEN_WEBUI_URL:-}" ]; then
    die "OPEN_WEBUI_URL is not set." \
        "" \
        "On the host, create your config and fill it in:" \
        "" \
        "    cp .env.example .env" \
        "    \$EDITOR .env" \
        "" \
        "Then: docker compose up -d" \
        "" \
        "See GETTING-STARTED.md for details."
fi

# Log in at startup unless a usable token already exists, so the first request does
# not have to absorb a refresh. A present-but-expired USER_TOKEN does not count.
have_usable_token() {
    python - <<'PY'
import os, sys, time
sys.path.insert(0, "/app")
try:
    from src.auth import get_current_token, get_token_expiry
    token = get_current_token()
except Exception:
    sys.exit(1)
exp = get_token_expiry(token)
# No exp claim means we cannot prove staleness; assume usable and let the 401 path decide.
sys.exit(0 if exp is None or time.time() < exp else 1)
PY
}

if ! have_usable_token; then
    if [ -z "${UA_NETID:-}" ] || [ -z "${UA_NETID_PASSWORD:-}" ]; then
        die "No usable token, and no way to obtain one." \
            "" \
            "Set UA_NETID and UA_NETID_PASSWORD in .env so the proxy can log in" \
            "for you and keep the token renewed automatically:" \
            "" \
            "    UA_NETID=your-netid" \
            "    UA_NETID_PASSWORD=your-password" \
            "" \
            "Then: docker compose up -d" \
            "" \
            "Alternatively paste a fresh JWT as USER_TOKEN, but it will expire and" \
            "cannot be renewed without the credentials above." \
            "" \
            "See GETTING-STARTED.md for details."
    fi

    echo "No usable token. Logging in to ${OPEN_WEBUI_URL} ..."
    echo "If Duo prompts, approve the push on your phone (usually only the first time)."
    if ! python /app/playwright_login.py; then
        die "Automatic login failed." \
            "" \
            "Check that UA_NETID and UA_NETID_PASSWORD in .env are correct," \
            "then retry:" \
            "" \
            "    docker compose up -d --force-recreate" \
            "" \
            "To see the full browser log:  docker compose logs proxy"
    fi
    echo "Login complete. Token saved."
fi

echo "Starting token refresh timer (every ${REFRESH_INTERVAL_SECONDS:-7200}s)..."
# A background loop replaces cron so the container needs no root privileges. The
# sidecar's own flock guard makes an overlapping proxy-triggered refresh a no-op.
(
    while true; do
        sleep "${REFRESH_INTERVAL_SECONDS:-7200}"
        python /app/playwright_login.py || echo "Scheduled refresh failed; will retry next interval" >&2
    done
) &

echo "Starting proxy on port ${PORT:-8000} ..."
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}"
