# Changelog

## Unreleased

- Admin page: a closeable warning on the Overview appears while no weights are downloaded, with a button to the Updates page. While **Download new weights** runs, the Updates card shows a progress bar with megabytes downloaded (`progress` in `GET /admin/updates`; new shared `src/progress.py`). Replicas share the numbers through a small file next to the weights marker; where that folder is not shared (Ollama services) the bar shows on the replica doing the download.

- Admin UI: separate hash-routed pages in a collapsible sidebar; health-first Overview, one main-header page title, capability-specific navigation, draft-preserving navigation and responsive desktop/mobile layout. Existing admin and public API actions are preserved.

- First version: the d3 decision model (`d3:2b`, vLLM Semantic Router, Apache 2.0) through Ollama, built from `CNV_LFM3_Model`'s Ollama adapter and admin page.
- Laya-shaped routes: `POST /v1/ask` (strategies `d3-only`, `d3-narrowing`, `auto`; `min_confidence`, `thresholds`, `prune_ratio`, `trace`, `stream`), `POST /v1/validate`, `POST /v1/combine` and `GET /v1/strategies`. `src/answers.py` holds Laya's question rules. Laya's two-model strategies, `project`, `max_len` and `pair_ratio` on `/v1/ask` return 422.
- `POST /v1/decide` takes `state` and 1 to 16 typed questions (`choice`, `noul`, `score`) and returns Ollama's `/v1/systemone` answers: probabilities and confidence, no prose. Text only; `images` and unknown fields return 422.
- Answer cache keeps every request (scoring is deterministic). No streaming, `temperature`, thinking or project models.
- Readiness check uses `/api/show` because d3 refuses `/api/generate`; the model loads into memory on the first decision.
- Verified against Ollama 0.40.15 with `d3:2b`. Not yet compared with `DSC_Laya_Model`; Docker not built.
