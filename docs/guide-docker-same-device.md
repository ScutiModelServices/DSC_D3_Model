# DSC_D3_Model: Docker Desktop on this PC

First start: open `/admin/` → **Updates** → **Download**; see [the web UI guide](guide-web-ui.md#updating-the-model-files). Startup and requests load downloaded files only; an empty installation keeps health/admin available and returns 503 for inference.

Run d3 API replicas on port 29221, all connected to the same Ollama.

## Requirements

- Docker Desktop with Compose v2, plus a running Ollama accessible from containers.
- Pull `d3:2b` on that Ollama first. The API image contains no model weights.
- Memory for the selected GGUF on the Ollama host; API memory usage has not been measured.
- External `citadel_core` network, created by the Citadel stack.

## Settings

Copy `.env.example` to `.env.compose`; saved admin settings take precedence over their environment defaults.

| Variable | Default | Effect |
|---|---|---|
| `SERVICE_ADMIN_PASSWORD` | empty | Enables authenticated public calls and admin access. |
| `SERVICE_PORT` | `29221` | Published load-balancer port. |
| `SERVICE_OLLAMA_URL` | `http://host.docker.internal:11434` | Required Ollama address; explicitly empty values fail startup. |
| `SERVICE_OLLAMA_MODEL` | `d3:2b` | Model name, read once at startup. |
| `SERVICE_OLLAMA_PARALLEL` | `2` | Decisions one replica runs at once; set it to the `OLLAMA_NUM_PARALLEL` of your Ollama. |
| `SERVICE_REPLICAS` | `1` | API replicas sharing Ollama. |
| `SERVICE_MEM_LIMIT` | `0` | Optional API container memory cap; does not cap Ollama. |

Admin settings: `update_check_hours` (6), `max_inflight` (0 = 4), `cache_size` (256), `warm` (true, restart). CPU thread values have no effect on Ollama. Details: [admin page](guide-web-ui.md#settings).

## Start

1. Start Ollama on a container-accessible interface and pull the configured model.
2. Set the admin password and backend address in `.env.compose`.
3. Start the API:

```bash
docker compose --env-file .env.compose up -d --build
```

Tokens, settings and replica update markers live in `dsc_d3_model_data`. Ollama manages its own model storage. The former service weights volume is unused; see [migration notes](../CHANGELOG.md).

## Check

```bash
curl http://127.0.0.1:29221/health
```

Warm-up runs in the background. The health endpoint and admin page remain available if Ollama is down; `/v1/decide` returns 503. A successful load or retry sets `ready` true. Open `http://127.0.0.1:29221/admin/` and create a public token. Container callers on `citadel_core` use `http://dsc_d3_model_lb:29221`.

## Updating

```bash
docker compose --env-file .env.compose up -d --build
docker compose restart dsc_d3_model_lb
```

Model pulls use the [Updates card](guide-web-ui.md#updating-the-model-files). Docker builds have not been verified for this migration.

## Problems

| Symptom | Fix |
|---|---|
| 503 or `ready: false` | Start Ollama and pull the configured model; check the backend URL and its container reachability. |
| Empty backend URL error | Set `SERVICE_OLLAMA_URL` or remove the variable to use the default. |
| 401 | Supply a valid public token; admin routes require the admin password. |
| 422 | Unknown field or invalid value; `images`, `project` and any unknown field are unsupported; `criteria` that does not fit its `type` is 422. |
| 429 | Wait for `Retry-After`, or adjust the in-flight cap/rate limit. |
| nginx 502 after scaling | Restart `dsc_d3_model_lb` so nginx resolves new replica addresses. |
