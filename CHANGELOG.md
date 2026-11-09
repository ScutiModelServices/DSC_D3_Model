# Changelog

## Unreleased

- First version: the d3 decision model (`d3:2b`, vLLM Semantic Router, Apache 2.0) through Ollama, built from `CNV_LFM3_Model`'s Ollama adapter and admin page.
- `POST /v1/decide` takes `state` and 1 to 16 typed questions (`choice`, `noul`, `score`) and returns Ollama's `/v1/systemone` answers: probabilities and confidence, no prose. Text only; `images` and unknown fields return 422.
- Answer cache keeps every request (scoring is deterministic). No streaming, `temperature`, thinking or project models.
- Readiness check uses `/api/show` because d3 refuses `/api/generate`; the model loads into memory on the first decision.
- Verified against Ollama 0.40.15 with `d3:2b`. Not yet compared with `DSC_Laya_Model`; Docker not built.
