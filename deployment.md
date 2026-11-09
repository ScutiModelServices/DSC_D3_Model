# Deploying DSC_D3_Model

DSC_D3_Model is one HTTP service (port 29221), with inference in a separately running Ollama. It runs as `dsc_d3_model` replicas behind an nginx load balancer,
`dsc_d3_model_lb`. Callers use `http://<host>:29221`; they never reach a replica directly.

## Ways to run it

Each path has its own guide. Pick one. First start needs an explicit admin Updates Download; see the web UI guide below.

| Path | Guide |
|---|---|
| Docker Desktop on this PC | [guide-docker-same-device.md](docs/guide-docker-same-device.md) |
| Docker on another always-on machine, published with `run.bat` (Deploy) | [guide-docker-deployment-device.md](docs/guide-docker-deployment-device.md) |
| Directly on Windows, no Docker | [guide-no-docker-same-device.md](docs/guide-no-docker-same-device.md) |

Once it runs, the admin page is covered by [guide-web-ui.md](docs/guide-web-ui.md).
