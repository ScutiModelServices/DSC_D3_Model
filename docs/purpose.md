# Purpose

**Kind:** text + typed questions → scored answers

**Group:** Decision

**Tags:** `cpu`, `host-threads`

| | |
|---|---|
| Input | `POST /v1/decide`: a `state` text and 1 to 16 named typed questions (`choice`, `noul`, `score`) with instructions and criteria. Text only here; the model also reads still images, which this service does not pass on. |
| Output | JSON with, per question, probabilities and a confidence (`choice`: chosen option; `noul`: probability true; `score`: expected level). No prose. |
| Runs on | CPU API; inference runs on the external Ollama host with `d3:2b` (vLLM Semantic Router, Apache 2.0, about 3.6 GB loaded). API memory has not been measured. |

**Use for:** a pretrained decision model to compare against `DSC_Laya_Model` on the same questions.
**Not for:** chat, free text, project checkpoints, images or video (422), or a Laya `/v1/ask` drop-in: the request shape differs.
