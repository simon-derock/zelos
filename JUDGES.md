# Judges' map: rubric → evidence

| Criterion | Where to look | Command |
|---|---|---|
| Problem importance & impact (30%) | README "What it does"; docs/WRITEUP.md §2 and §9: image QC as the certification gate before OoC data becomes a dataset or digital-twin input | — |
| Technical approach & innovation (30%) | `src/zelos/vision/` (physics metrics, robust reference, cause → action), `src/zelos/grounding/` (claim ledger gate), `src/zelos/agent/loop.py` (≤2 LLM calls, router, repair), `src/zelos/rag/` (Cohere + Qdrant + BM25 + RRF + rerank), `src/zelos/mcp_server.py`, `skills/zelos-qc/SKILL.md` | `make mcp`, `zelos explain <img> --provider workers_ai` |
| Results & validation (20%) | `results/results.md` (E1, E2, E4, E5, E9 with baselines, CIs, n, failures) | `make eval` |
| Reproducibility & quality (10%) | offline replay of every API response, pinned deps, Dockerfile, CI (lint, types, security, tests, smoke + provenance check, web, Docker), release workflow | `make judge` |
| Presentation (10%) | demo video; live console https://zelos.philipsimonderock.com (renders only real pipeline outputs from `web/public/data/site.json`) | `make site` |
| Interpretability & reliability | every claim typed and linked to a tool id or chunk id; fault injection: 100/100 fabrications blocked | `scripts/eval_grounding.py --replay` |
