# DSC_D3_Model

The d3 decision model (`d3:2b`, vLLM Semantic Router) served over HTTP through Ollama, behind the shared model-service admin page, tokens and settings. Port 29221, env prefix `SERVICE_`, token prefix `d3_`.

## Agent instructions
This file is the one instruction file for every coding agent; `CLAUDE.md` only imports it (`@AGENTS.md`). Edit this file, never `CLAUDE.md`.

## Commands (from this folder, venv `.venv`, managed by uv)
- Install: `uv sync` (`winget install astral-sh.uv` if uv is missing)
- Test: `uv run pytest -q`. `tests/test_admin_ui.py` needs Playwright and its browser and is skipped without it. On Windows, if uv fails with "cannot move the file to a different disk drive", set `UV_CACHE_DIR` to a folder on the same drive and `UV_LINK_MODE=copy`.
- Run: `run.bat run`, or `uv run python src/app.py` (add `--dotenv` to load `.env.local` and `.env`), or `docker compose --env-file .env.compose up -d --build`

## Rules
- Load downloaded weights only; no startup, warm-up or request downloads. Only the admin Updates Download callback downloads, then replica markers invalidate old weights. Missing files keep health/admin available and inference returns 503 with an admin Download hint.
- Ollama is the only backend (`src/ollama_backend.py`); local URL defaults there, and explicitly empty URLs fail. Run Ollama; use the admin Updates Download action before using `/v1/decide`.
- d3 refuses `/api/generate`: the loader checks `/api/show`, and Ollama loads the model on the first `/v1/systemone` call. Questions use `criteria` (an object for `choice` and `noul`, a list for `score`).
- `num_thread` is saved/reported only; Ollama chooses its threads. Keep injected fake loaders for tests, replica marker invalidation and recoverable 503 errors.
- Follow the `model-service-parity` skill and `AI/ModelServices/_docs/SHARED-FEATURES.md` (what must match in every model service) and `AI/ModelServices/_docs/SERVICE-SPECIFIC.md` (what differs on purpose). A change to a shared feature goes to every service; a new difference goes into `SERVICE-SPECIFIC.md` in the same change.
- `web/index.html`, `src/tokens.py`, `src/settings.py`, `src/updates.py` and `.dockerignore` are copies shared with the other model services. Change one, copy it to the others, run `python tools/check_sync.py` in each repo.
- Variants: `src/variants.py` lists the d3 sizes from the Ollama library (real manifest sizes; the estimate is arithmetic and labelled so; `-q8_0` is the same files, `-mxfp8` and `-nvfp4` are not offered), `src/hardware.py` reads RAM and the GPU (the same file as in STT_Whisper_Model). The Ollama model name is the restart setting `ollama_model`, which the `/admin/variant` endpoint saves; `engine_from_env` reads it from the settings. When Ollama is not on this machine (`engine.same_machine` is False) nothing is checked or blocked.
- Behaviour changes update `README.md`, `deployment.md`, `docs/guide-web-ui.md`, `.env.example` and `CHANGELOG.md` in the same commit.
- Never read or print `.env`, `.env.local`, `.env.compose`, `tokens.json` or `settings.json`.
- Commit with the noreply author email `214214011+scutiii2@users.noreply.github.com` (another address is rejected on push). Push `main`, then run `python tools/publish_live.py`. Build Docker only when the Dockerfile or the dependencies changed.
