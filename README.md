# DSC_D3_Model

First start: open `/admin/` → **Updates** → **Download**; see [the web UI guide](docs/guide-web-ui.md#updating-the-model-files). Startup and requests load downloaded files only; an empty installation keeps health/admin available and returns 503 for inference.

The d3 decision model (`d3:2b` by vLLM Semantic Router, Apache 2.0) over HTTP through Ollama, port 29221. It exists to be compared with `DSC_Laya_Model` on the same questions. Callers share one Ollama model through API replicas, with admin settings, tokens, rate limits, metrics and an answer cache. No tool service is required.

## API

| Call | Purpose |
|---|---|
| `POST /v1/decide` | `state` plus `questions`; returns `answers`, `model`, `usage`, `duration_ms`. |
| `GET /health` | `status`, `model`, `ready`, `num_thread`; no token required. |

**Input.** `state` is text (at most 100,000 characters). `questions` is 1 to 16 named questions, each `{type, instructions, criteria}`:

- `choice`: `criteria` maps each option to its meaning. The answer has `choice`, `probabilities` and `confidence`.
- `noul`: `criteria` has exactly `true` and `false`. The answer has `noul`, the probability of true.
- `score`: `criteria` lists 2 to 10 level descriptions, lowest first. The answer has `score` (expected level), `legend`, `probabilities` and `confidence`.

Unknown fields, including `images`, return 422: this service passes text only, although d3 also reads still images. Scores are model outputs, not guaranteed frequencies; check them on your own examples before setting thresholds.

**Result.** The answers exactly as d3 returns them, one per question, no prose. An unavailable Ollama or missing model returns 503 with a hint to start Ollama and pull the model. The API and admin page remain available during failed warm-up.

**Cache.** Scoring is deterministic, so every request is kept, per replica, with `X-Cache: hit|miss`. The key includes the model generation, state and questions. Download invalidates answers across replicas. There is no streaming and no `temperature`.

**Tokens.** With `SERVICE_ADMIN_PASSWORD` set, public calls require `X-Internal-Token`. Tokens support expiry, rate limits and revocation. Token project restrictions do not exist: lists return 422; omitted or null `projects` stays compatible with the shared editor.

**Versus Laya.** Laya takes `/v1/ask` with typed questions and strategies; d3 takes `/v1/decide` with the `criteria` shape above, so a caller needs a small adapter to send one question to both. d3 loads about 3.6 GB in Ollama (Laya's two checkpoints are about 1.7 GB of weights) and wants a GPU or a fast CPU.

## Admin page

Open `/admin/` with `SERVICE_ADMIN_PASSWORD`. Tokens, usage, replicas, Settings and Updates share the standard model-service page. CPU thread values are saved and reported only; Ollama chooses its own threads. Settings retain update-check hours, in-flight cap, answer cache size and warm-up. The model name is the startup value `SERVICE_OLLAMA_MODEL`.

The Updates card compares the downloaded Ollama manifest digest with the online registry manifest at startup, every `update_check_hours` and on **Check now**. Checks fetch metadata only; only **Download** runs `ollama pull`. A successful download invalidates every replica through shared storage; each replica reloads on its next request.

## Run

Start Ollama and the API, then download from `/admin/` → **Updates** → **Download**:

```bash
curl -X POST http://127.0.0.1:29221/admin/updates/weights -H "X-Internal-Token: <ADMIN_PASSWORD>"
```

```bash
docker compose up -d --build
```

Docker defaults to `http://host.docker.internal:11434`. A local run defaults to `http://127.0.0.1:11434` when `SERVICE_OLLAMA_URL` is unset. An explicitly empty URL is an error. Without Docker use `run.bat run` with a running Ollama; when copying `.env.example`, change the Docker URL to the local one.

```bash
curl -X POST http://127.0.0.1:29221/v1/decide -H "Content-Type: application/json" -H "X-Internal-Token: d3_..." -d '{"state":"A customer says their card was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Invoices and charges","returns":"Refunds and replacements"}}}}'
```

## Tests

```bash
uv run pytest
```

Browser tests need Playwright and Chromium. Checked against a real Ollama 0.40.15 with `d3:2b`: one `choice` and one `score` question answered in about a second. Docker builds have not been run, and d3 has not been compared with Laya yet.

## Limits worth knowing

- One decision per API replica; excess waiting calls return 429 with `Retry-After: 5`.
- Rate limits are per replica. Ollama owns model memory and scheduling; API replicas share its model.
- Network timeout is 120 seconds. Readiness becomes true once Ollama reports the model installed and false after a failed call. Ollama loads the model into memory on the first decision, which is slower.
- Ollama's page says its d3 packages use a 4,096-token context (the tag list says 256K). Keep `state` and questions short until this is verified.
- No per-project model selection and no project checkpoints.

## Try it from the admin page

Use **Try it** with a public token kept only in memory. Decide sends one `choice` question (state, question text, two options). Status, latency, errors, cancellation and redacted curl appear in the panel. See [web UI guide](docs/guide-web-ui.md#try-it-send-a-real-request).

## Documentation

- [Purpose](docs/purpose.md)
- [Deployment](deployment.md)
- [Changelog](CHANGELOG.md)
- [Admin page](docs/guide-web-ui.md)
- [Docker on this PC](docs/guide-docker-same-device.md)
- [Docker on another machine](docs/guide-docker-deployment-device.md)
- [Without Docker](docs/guide-no-docker-same-device.md)
- [Command-line calls](docs/guide-cli.md)
