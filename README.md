# Rasa

Run instructions for:

- Development using the Dev Container
- Production using GitHub workflow-built images

## Service Wiring

- Webapp calls Rasa REST API (`/webhooks/rest/webhook`, tracker endpoints)
- Rasa calls Action at `ACTION_ENDPOINT_URL`
- Rasa uses Redis for tracker and lock stores
- Duckling can be used if enabled in pipeline configuration

## Required Environment Variables

- `ACTION_ENDPOINT_URL` (example: `http://action:5055/webhook`)
- `TRACKER_STORE_URL`
- `TRACKER_STORE_DB`
- `LOCK_STORE_URL`
- `LOCK_STORE_DB`
- `RASA_AUTH_TOKEN`

Recommended:

- `RASA_CORS` as a single explicit origin (no wildcard)
- `RASA_REQUEST_TIMEOUT` and `RASA_RESPONSE_TIMEOUT` aligned (example: `300`)

## Development (Dev Container)

1. Open this repository in VS Code.
2. Reopen in container.
3. Train a locale model (example):

```bash
bash scripts/layer_rasa_lang.sh en/US
```

4. Run Rasa:

```bash
python -m src.run_rasa
```

The dev container definition is in `.devcontainer/Dockerfile`.

## Production (Workflow-built images)

GitHub workflows build and publish locale-specific Rasa images to GHCR.

Typical tags:

- `ghcr.io/<org>/rasa:en-US-latest`
- `ghcr.io/<org>/rasa:en-US-<git-sha>`

Run example:

```bash
docker run --rm -p 5005:5005 \
  -e ACTION_ENDPOINT_URL=http://action:5055/webhook \
  -e TRACKER_STORE_URL=redis \
  -e TRACKER_STORE_DB=0 \
  -e LOCK_STORE_URL=redis \
  -e LOCK_STORE_DB=1 \
  -e RASA_AUTH_TOKEN=<shared-rasa-token> \
  -e RASA_CORS=http://localhost:3000 \
  -e RASA_REQUEST_TIMEOUT=300 \
  -e RASA_RESPONSE_TIMEOUT=300 \
  ghcr.io/<org>/rasa:en-US-latest
```
