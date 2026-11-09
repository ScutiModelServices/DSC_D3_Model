# DSC_D3_Model

First start: open `/admin/` → **Updates** → **Download**; see [the web UI guide](docs/guide-web-ui.md#updating-the-model-files). Startup and requests load downloaded files only; an empty installation keeps health/admin available and returns 503 for inference.

The d3 decision model (`d3:2b` by vLLM Semantic Router, Apache 2.0) over HTTP through Ollama, port 29221. It exists to be compared with `DSC_Laya_Model` on the same questions. Callers share one Ollama model through API replicas, with admin settings, tokens, rate limits, metrics and an answer cache. No tool service is required.

## API

| Call | Purpose |
|---|---|
| `POST /v1/decide` | `state` plus `questions`; returns `answers`, `model`, `usage`, `duration_ms`. |
| `POST /v1/ask` | Laya-shaped: `text`, `questions`, optional `strategy`, `min_confidence`, `thresholds`, `prune_ratio`, `trace`, `stream`; returns `strategy`, `model`, `fallback`, `project_used`, `answers`, `uncertain`, `calls`. |
| `POST /v1/validate` | `{"questions"}` → `{"ok": true}` or 422 saying what is wrong. Same rules as Laya (1 to 8 questions, 2 to 10 options; `noul` criteria may be left out). |
| `POST /v1/combine` | `{"voters", "options", "prune_ratio", "pair_ratio"}`: mixes a caller's own voter into d3's answer, as Laya's does. |
| `GET /v1/strategies` | `{"default", "strategies"}`. |
| `GET /health` | `status`, `model`, `ready`, `num_thread`; no token required. |

**Input.** `state` is text (at most 100,000 characters). `questions` is 1 to 16 named questions, each `{type, instructions, criteria}`:

- `choice`: `criteria` maps each option to its meaning. The answer has `choice`, `probabilities` and `confidence`.
- `noul`: `criteria` has exactly `true` and `false`. The answer has `noul`, the probability of true.
- `score`: `criteria` lists 2 to 10 level descriptions, lowest first. The answer has `score` (expected level), `legend`, `probabilities` and `confidence`.

Unknown fields, including `images`, return 422: this service passes text only, although d3 also reads still images. Scores are model outputs, not guaranteed frequencies; check them on your own examples before setting thresholds.

**Result.** The answers exactly as d3 returns them, one per question, no prose. An unavailable Ollama or missing model returns 503 with a hint to start Ollama and pull the model. The API and admin page remain available during failed warm-up.

### `/v1/ask` and its strategies

`/v1/ask` answers like Laya's: per question the pick (`choice`, `score` or `noul`), `answer_confidence`, `uncertain` and `answered_by` (`d3`); `trace: true` adds `rounds`. `stream: true` sends one NDJSON `answer` line per question, then `done`. It shares the `max_inflight` cap and the cache with `/v1/decide`.

| Strategy | What it does | Cost |
|---|---|---|
| `d3-only` | One call for all questions; `uncertain` when under the threshold. | 1 call |
| `d3-narrowing` | A choice question that is not settled drops clearly worse options and is asked again (up to 3 rounds). | 1+ calls |
| `auto` (default) | `d3-narrowing`. | 1+ calls |

Thresholds (confidence at or above): 2 options 0.90, 3+ options 0.75. They are Laya English's numbers and have not been measured for d3. `min_confidence` replaces them; `thresholds` (`{"2": {"d3": 0.9}, "3+": {"d3": 0.7}}`) replaces one; `prune_ratio` (0.25) tunes the narrowing. `answer_confidence` of a `noul` answer is the distance from a coin flip, because d3 gives none for yes/no.

Laya's two-model strategies (`english-only`, `typed-only`, `english-then-typed`, `typed-then-english`, `most-confident`, `average`, `consensus`) do not exist here: d3 is one model, so they return 422. `project`, `max_len` and `pair_ratio` are 422 too. `/v1/combine` still takes `pair_ratio` for callers that add voters.

**Cache.** Scoring is deterministic, so every request is kept, per replica, with `X-Cache: hit|miss`. The key includes the model generation, state and questions. Download invalidates answers across replicas. There is no streaming and no `temperature`.

**Tokens.** With `SERVICE_ADMIN_PASSWORD` set, public calls require `X-Internal-Token`. Tokens support expiry, rate limits and revocation. Token project restrictions do not exist: lists return 422; omitted or null `projects` stays compatible with the shared editor.

**Versus Laya.** `/v1/ask`, `/v1/validate`, `/v1/combine` and `/v1/strategies` take Laya's request and answer shapes, so a caller can send one question set to both (set `strategy` per service: Laya's default `auto` also works here). What d3 lacks: the two-model strategies, project models, `max_len`, `X-Laya-Cache` and `sync_seconds`. d3 loads about 3.6 GB in Ollama (Laya's two checkpoints are about 1.7 GB of weights) and wants a GPU or a fast CPU.

## Admin page

Open `/admin/` with `SERVICE_ADMIN_PASSWORD`. Tokens, usage, replicas, Settings, Updates and the Model variant card (choose a d3 size this machine can run, with estimated memory use) share the standard model-service page. CPU thread values are saved and reported only; Ollama chooses its own threads. Settings retain update-check hours, in-flight cap, answer cache size and warm-up. The model name is the restart setting `ollama_model` (else `SERVICE_OLLAMA_MODEL`).

The Updates card compares the downloaded Ollama manifest digest with the online registry manifest at startup, every `update_check_hours` and on **Check now**. Checks fetch metadata only; only **Download** runs `ollama pull`. A successful download invalidates every replica through shared storage; each replica reloads on its next request.

## Run

Start Ollama and the API, then download from `/admin/` → **Updates** → **Download**:

```bash
curl -X POST http://127.0.0.1:29221/admin/updates/weights -H "X-Internal-Token: <ADMIN_PASSWORD>"
```

```bash
docker compose --env-file .env.compose up -d --build
```

Docker defaults to `http://host.docker.internal:11434`. A local run defaults to `http://127.0.0.1:11434` when `SERVICE_OLLAMA_URL` is unset. An explicitly empty URL is an error. Without Docker use `run.bat run` with a running Ollama; copy `.env.example` to `.env.local` and set the local URL there; `.env.compose` (Docker Desktop here) and `.env` (the deployment device) keep the Docker one.

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

## Admin pages and navigation

Open `/admin/` and sign in. Overview opens first; each sidebar destination has its own URL fragment and supports browser Back/Forward and direct links:

- `#overview`: health, model readiness, replicas, active tokens and request counts.
- `#playground`: service-specific request forms and streamed or media responses.
- `#tokens`: create, edit, expire and revoke scoped access tokens.
- `#runtime`: CPU threads where supported, replica information and live/restart settings.
- `#updates`: check versions and explicitly download weights.
- `#connections`: caller addresses and public endpoints.
- `#models` and `#voices`: project models and cloned voices, shown only when supported.

Collapse/Expand switches between labeled sidebar and icon rail on desktop and mobile. The non-secret sidebar preference is saved per service in this browser. Navigation preserves form drafts and active Playground requests; sign-out still clears credentials and cancels work. Page names appear once in the main header. No preview data or cross-service selector ships in the real admin UI. Existing authentication, quotas, settings precedence and explicit-download policy remain in effect.
