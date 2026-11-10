# Changelog

## Unreleased

### Playground: no token to paste

The Try it panel signs in with a temporary public token made from the admin session (named `playground (temporary)`, valid 1 day, revoked on sign-out or when the tab closes), so the service token field is hidden and choices such as voices load by themselves. The token shows in the tokens list while it lives.


- Admin page: new **Model variant** card. It lists the d3 sizes (`2b`, `4b`, `9b`, `27b`) with an estimate of the memory each takes (download size x 1.1 + 400 MB; not measured), the download size, whether Ollama has it, and whether this machine can run it on the GPU or only the CPU. A larger size is a larger model, not a smaller copy of the same one. Endpoints `GET /admin/variants` and `PUT /admin/variant` (409 when it does not fit). The Ollama model name is now the restart setting `ollama_model` (saved value, else `SERVICE_OLLAMA_MODEL`, else `d3:2b`). New `src/hardware.py` and `src/variants.py`.
- Env files: `run.bat` (`--dotenv`) now loads `.env.local`, then `.env`; the first file to set a name wins and a real environment variable still beats both, so a host run can keep `SERVICE_OLLAMA_URL=http://127.0.0.1:11434` in `.env.local` while Docker uses `http://host.docker.internal:11434`. `.env.compose` is for Docker Desktop on this device (`docker compose --env-file .env.compose up -d --build`). `.env` stays the file Deploy option 2 pushes to the deployment device, where a plain `docker compose up -d --build` reads it. `.env.local` and `.env.compose` are git-ignored and Docker-ignored (`.env.*`). New shared `settings.load_env_files`.
- Admin page and shared styles: polish layer in `web/index.html` (focus ring, hover and pressed states, layered shadow on the sign-in card and dialogs, tabular numbers, one entrance motion per page, themed scrollbars, `prefers-reduced-motion`). No behaviour change.

- Admin page: a closeable warning on the Overview appears while no weights are downloaded, with a button to the Updates page. While **Download new weights** runs, the Updates card shows a progress bar with megabytes downloaded (`progress` in `GET /admin/updates`; new shared `src/progress.py`). Replicas share the numbers through a small file next to the weights marker; where that folder is not shared (Ollama services) the bar shows on the replica doing the download.

- Admin UI: separate hash-routed pages in a collapsible sidebar; health-first Overview, one main-header page title, capability-specific navigation, draft-preserving navigation and responsive desktop/mobile layout. Existing admin and public API actions are preserved.

- First version: the d3 decision model (`d3:2b`, vLLM Semantic Router, Apache 2.0) through Ollama, built from `CNV_LFM3_Model`'s Ollama adapter and admin page.
- Laya-shaped routes: `POST /v1/ask` (strategies `d3-only`, `d3-narrowing`, `auto`; `min_confidence`, `thresholds`, `prune_ratio`, `trace`, `stream`), `POST /v1/validate`, `POST /v1/combine` and `GET /v1/strategies`. `src/answers.py` holds Laya's question rules. Laya's two-model strategies, `project`, `max_len` and `pair_ratio` on `/v1/ask` return 422.
- `POST /v1/decide` takes `state` and 1 to 16 typed questions (`choice`, `noul`, `score`) and returns Ollama's `/v1/systemone` answers: probabilities and confidence, no prose. Text only; `images` and unknown fields return 422.
- Answer cache keeps every request (scoring is deterministic). No streaming, `temperature`, thinking or project models.
- Readiness check uses `/api/show` because d3 refuses `/api/generate`; the model loads into memory on the first decision.
- Verified against Ollama 0.40.15 with `d3:2b`. Not yet compared with `DSC_Laya_Model`; Docker not built.
