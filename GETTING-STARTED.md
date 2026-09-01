# Getting Started

Run an OpenAI-compatible API in front of your Open WebUI instance.

```sh
git clone <this-repo> && cd open-webui-openai-compatible
cp .env.example .env
$EDITOR .env          # fill in 3 values (below)
docker compose up -d
```

That's it. No Python install, no manual token copying.

---

## What to put in `.env`

Only three values matter to start:

```sh
OPEN_WEBUI_URL=https://your-open-webui-instance.example.com
UA_NETID=your-netid
UA_NETID_PASSWORD=your-password
```

The credentials let the proxy log in on your behalf with a headless browser and
keep the token renewed, so you never paste a JWT by hand.

> **Not using SSO?** Leave `UA_NETID`/`UA_NETID_PASSWORD` blank and paste a JWT into
> `USER_TOKEN` instead. It works, but it will expire and cannot be auto-renewed.

If something is missing, the container tells you exactly what to set and exits —
it will not fail silently.

---

## First start

The first `up -d` takes a few minutes: it builds the image (including a headless
Chromium) and performs the initial login.

If your instance uses Duo, **approve the push on your phone** during that first
login. Watch it happen:

```sh
docker compose logs -f proxy
```

Once you see `Uvicorn running`, you're live:

```sh
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

---

## Using it

Point any OpenAI-compatible client at `http://localhost:8000/v1`. The API key is
ignored — the proxy handles auth itself.

```sh
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello!"}]}'
```

Get a valid `MODEL_ID` from `/v1/models` — **do not guess it**. Model IDs are
specific to your instance and an unknown one returns `400 Model not found`.

```sh
curl -s http://localhost:8000/v1/models | python -m json.tool | grep '"id"'
```

Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
print(client.chat.completions.create(
    model="MODEL_ID",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

Anthropic clients can use `POST /v1/messages` at the same host.

---

## How renewal works

Tokens expire. You should never have to think about it:

1. A background timer refreshes the token every 2 hours.
2. If a request hits an expired token first, the proxy returns `503`, refreshes in
   the background, and the next request succeeds. **Retry once on 503.**
3. Renewal reuses the browser session stored in the `proxy-data` volume, so it is
   normally silent — no Duo prompt.

Duo only prompts again if that stored session *and* the Duo trusted-browser cookie
both lapse (typically weeks).

> **`docker compose down -v` deletes that volume**, which forces a full interactive
> login next start. Use plain `down` to keep it.

---

## Everyday commands

| Task | Command |
|------|---------|
| Start | `docker compose up -d` |
| Stop (keep session) | `docker compose down` |
| Follow logs | `docker compose logs -f proxy` |
| Restart | `docker compose restart proxy` |
| Apply code changes | `docker compose up -d --build` |
| Force a login now | `docker compose exec proxy python /app/playwright_login.py` |
| Reset everything | `docker compose down -v` |

---

## Troubleshooting

**Container exits right away.** Read the message — it names the missing variable.

```sh
docker compose logs proxy
```

**`503 ... token refresh initiated`.** Working as designed on an expired token.
Wait a few seconds and retry. If it persists, the login is failing:

```sh
docker compose logs proxy | grep -v "Still waiting"
```

**`400 Model not found`.** The `model` value isn't on your instance. List real IDs:

```sh
curl -s http://localhost:8000/v1/models | python -m json.tool | grep '"id"'
```

**Login times out waiting for Duo.** The push wasn't approved within 5 minutes.
Retry and approve promptly:

```sh
docker compose restart proxy && docker compose logs -f proxy
```

**Credentials changed** (e.g. after a password rotation). Update `.env`, then:

```sh
docker compose up -d --force-recreate
```

**Start over completely:**

```sh
docker compose down -v && docker compose up -d
```

---

## Notes on secrets

- `.env` holds your password in plain text and is git-ignored. Keep it that way.
- The proxy loads it as a `SecretStr`, so it is masked in logs and error output.
- Nothing is transmitted anywhere except your own Open WebUI instance during login.

---

## Configuration reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPEN_WEBUI_URL` | Yes | — | Base URL of your instance (no trailing slash) |
| `UA_NETID` | For renewal | — | Username for automatic login |
| `UA_NETID_PASSWORD` | For renewal | — | Password for automatic login |
| `USER_TOKEN` | No | — | Manual JWT fallback; expires, not renewable |
| `PORT` | No | `8000` | Listen port |
| `REQUEST_TIMEOUT` | No | `300` | Upstream timeout, seconds (10–3600) |
| `STREAM_EMPTY_RETRY_MAX` | No | `3` | Retries for empty streams (0–10) |
| `REFRESH_INTERVAL_SECONDS` | No | `7200` | Background refresh interval |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |

See [README.md](README.md) for architecture and API details.
