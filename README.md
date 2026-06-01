# Rasa Service — Runbook

This README only covers:

- How to run Rasa as a developer
- How to run it in production

---

## Related repositories

- Webapp: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Webapp
- Action: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Action
- Rasa (this repo): https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Rasa
- SSOT: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/SSOT

---

## 1) Development setup

### Prerequisites (recommended path)

- Docker + Docker Compose
- VS Code + Dev Containers extension

### Dev Container workflow (recommended)

1. Open this repository in VS Code.
2. Choose **Reopen in Container**.
3. Wait for post-create setup to finish (`pip install -e . && mypy --strict . && ruff check .`).
4. Configure `.devcontainer/.env` as shown below.
5. Train a locale model and run the service command.

The dev container also starts sidecars for `redis` and `duckling`.

### Option B: Local machine

Use local Python/Rasa tooling and ensure dependency services are reachable (Redis, Duckling, Action).

### Configure environment (`.devcontainer/.env`)

For local API/shell runs, set:

```bash
ACTION_ENDPOINT_URL=http://host.docker.internal:5055/webhook
TRACKER_STORE_URL=redis
TRACKER_STORE_DB=0
LOCK_STORE_URL=redis
LOCK_STORE_DB=1
RASA_AUTH_TOKEN=<shared-rasa-token>
DUCKLING_ENDPOINT_URL=http://duckling:8000
RASA_SHELL_STREAM_READING_TIMEOUT_IN_SECONDS=3600
```

### Run locally

Train model (language-aware layering):

```bash
bash scripts/layer_rasa_lang.sh en/US
```

Examples:

```bash
bash scripts/layer_rasa_lang.sh cs/CZ
bash scripts/layer_rasa_lang.sh el/GR
```

Dry-run merge without training:

```bash
bash scripts/layer_rasa_lang.sh --dry-run=stdout en/US
```

Run API:

```bash
rasa run --enable-api --auth-token "$RASA_AUTH_TOKEN" --model models --endpoints src/core/endpoints.yml --request-timeout 300 --response-timeout 300
```

Run interactive shell:

```bash
rasa shell --model models --endpoints src/core/endpoints.yml --request-timeout 300 --response-timeout 300
```

### VS Code tasks (optional)

This repo includes `.vscode/tasks.json` with tasks for dry-run, training, and runtime, including:

- `Rasa: Dry Run (lang spec)`
- `Rasa: Train (lang spec)`
- `Rasa: Run (latest)`
- `Rasa: Shell (latest)`

Run from VS Code: **Terminal → Run Task**.

---

## 2) Production run

### Required dependencies

- Action service webhook endpoint
- Redis (tracker + lock stores)
- Duckling (if enabled in your pipeline)

### Recommended image tags

This repo builds one image per language/region:

- `ghcr.io/<org>/rasa:<lang>-<region>-latest` (example: `en-US-latest`)
- `ghcr.io/<org>/rasa:<lang>-<region>-<git-sha>`

### Required environment variables (per Rasa container)

```bash
ACTION_ENDPOINT_URL=http://<action-service>:5055/webhook
TRACKER_STORE_URL=<redis-host>
TRACKER_STORE_DB=<unique-int>
LOCK_STORE_URL=<redis-host>
LOCK_STORE_DB=<unique-int>
RASA_AUTH_TOKEN=<shared-rasa-token>
```

For multi-language deployment, run one container per locale image and assign unique Redis tracker/lock DB pairs.
Keep Rasa on a private service network behind the Webapp; the published image
ships with CORS disabled for browser clients.

### Minimal production compose snippet (single locale)

```yaml
services:
  rasa-en:
    image: ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/rasa:en-US-latest
    depends_on: [action, redis, duckling]
    environment:
      ACTION_ENDPOINT_URL: http://action:5055/webhook
      TRACKER_STORE_URL: redis
      TRACKER_STORE_DB: 0
      LOCK_STORE_URL: redis
      LOCK_STORE_DB: 1
      RASA_AUTH_TOKEN: "<shared-rasa-token>"

  action:
    image: ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/action:latest

  redis:
    image: redis:alpine

  duckling:
    image: rasa/duckling:latest
```

Production image runtime command:

```bash
rasa run --enable-api --auth-token "$RASA_AUTH_TOKEN" --endpoints src/core/endpoints.yml --request-timeout 300 --response-timeout 300
```

---

## 3) Quick verification

- Check auth enforcement: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5005/status` should return `401`
- Check authenticated health endpoint: `curl -s "http://localhost:5005/status?token=<shared-rasa-token>"`
- Verify Action is reachable from each Rasa container
- Verify Redis DB allocation does not overlap across locales
- Send a test message through REST webhook and confirm Action callbacks

---

## 4) Common commands

Train locale model:

```bash
bash scripts/layer_rasa_lang.sh <lang>/<REGION>
```

Dry-run locale merge:

```bash
bash scripts/layer_rasa_lang.sh --dry-run=stdout <lang>/<REGION>
```

Run API with latest local model:

```bash
rasa run --enable-api --auth-token "$RASA_AUTH_TOKEN" --model models --endpoints src/core/endpoints.yml --request-timeout 300 --response-timeout 300
```

Run local interactive shell:

```bash
rasa shell --model models --endpoints src/core/endpoints.yml
```
