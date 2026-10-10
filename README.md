<div align="center">

# Zelos

**Every organ-on-a-chip image, certified.**
An evidence-gated QC agent that tells an OoC lab *whether* a brightfield image can be trusted, *why* not, *what to do*, and which open-access paper says so, and that refuses to state anything its tools or sources cannot support.

[![CI](https://github.com/simon-derock/zelos/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-derock/zelos/actions)
![License](https://img.shields.io/badge/license-MIT-green) ![Training](https://img.shields.io/badge/gradient%20updates-0-blue) ![Offline](https://img.shields.io/badge/runs-offline%20%7C%20CPU-black)

AI4S Open Innovation: AI for Life Science (5th Pazhou Algorithm Competition) · **Category: End-to-End System** · Built by Philip Simon Derock

**[Live console](https://zelos.philipsimonderock.com)** · [Results](results/results.md) · [Judges' map](JUDGES.md) · [Limitations](docs/LIMITATIONS.md)

</div>

> Research tool. Not a medical device. No clinical or diagnostic claims.

## Judge quickstart (CPU, no API keys, no paid services)

```bash
git clone https://github.com/simon-derock/zelos && cd zelos
make setup            # venv + pinned deps
make judge            # tests + retrieval (E4) + grounding (E5) replayed offline, results rebuilt
make data && make eval  # optional: fetch the real 480-image subset from Zenodo and regenerate E1/E2/E9 too
```

Cohere and Cloudflare Workers AI responses used in the published runs are committed (`corpus/`, `eval/llm_cache/`), so every number reproduces without keys. With keys in `.env` (see `.env.example`; both services have free tiers) the same commands call the live APIs.

## What it does

```
image ─► QC metrics (17, physics-based) ─► robust z vs reference ─► cause + lab action ─► evidence retrieval ─► LLM explainer ─► grounding gate ─► certificate + claim ledger
          0 LLM calls                       0 LLM calls               0 LLM calls          Cohere + Qdrant + BM25   Workers AI       0 LLM calls
```

**The LLM proposes. Tools compute. The gate disposes.** Every number in an answer must come from a tool output (`qc-…` id), every literature sentence must cite a retrieved chunk that supports it, inferences must rest on supported claims, and anything else is repaired once or blocked, with the reason recorded in the claim ledger.

| Specialist | Job | LLM calls |
|---|---|---|
| Vision QC | focus ×3, contrast ×2, exposure/clipping ×3, illumination, noise, debris ×2, bubble, confluence, blob density, texture, compression | 0 |
| Reference comparator | median/MAD robust z per metric, percentile bands, density state; dev-session or batch reference | 0 |
| Cause attribution + action | fixed a-priori rule table (`configs/artifact_rules.yaml`, `configs/actions.yaml`) | 0 |
| Evidence retriever | Cohere `embed-v4.0` vectors in Qdrant + BM25, reciprocal rank fusion, Cohere `rerank-v3.5`; open-access CC BY / CC0 papers only | 0 |
| Explainer | Cloudflare Workers AI `llama-3.1-8b-instruct-fast`, strict JSON claims with evidence ids | 1 |
| Grounding gate + repair | numeric binding, citation existence + support, inference chaining; ≤1 repair | ≤1 |

Interfaces: CLI (`zelos inspect|explain|triage|ask`), FastAPI (`make api`), **MCP server** (`make mcp`, tools `inspect_image`, `explain_image`, `retrieve_evidence`, `check_claims`, `triage_folder`) with an agent skill in [`skills/zelos-qc/SKILL.md`](skills/zelos-qc/SKILL.md), and the web console in `web/`.

## Results (real OoC data; every number from [`results/results.md`](results/results.md))

Data: the expert-labelled **Organ-on-a-Chip (OOC) Image Dataset** (Movčana et al., *Data* 2024; Zenodo <!--np-->10.5281/zenodo.10203721<!--/np-->), a stratified 480-image subset across six cell lines, split by experiment session so no session is in both dev and test.

| Experiment | Headline |
|---|---|
| **E1 — injected corruptions, known ground truth** (11 kinds × 5 severities) | flags 97.1% of severity≥3 corruptions; names the injected cause for 31.0% of flagged images (top-2: 49.0%) vs chance 11.1%. Batch-relative reference: attribution 40.6% (top-2 57.0%). |
| E1 honest failure | the pre-registered dev-session reference also flags 76.7% of clean test images: **cross-session acquisition shift** (contrast and compression statistics differ by session). A batch-relative reference (unsupervised, added after this result) lowers that to 46.7% at AUROC 0.728. |
| **E2 — expert good/bad agreement** (n_test = 301) | the artifact QC score does *not* predict the expert label (AUROC 0.393, below the random baseline 0.518); kNN label transfer on the same physics features reaches **AUROC 0.710**. The expert label appears to track culture state, not image artifacts, so Zelos reports the two separately. |
| **E5 — grounding gate** (Workers AI 8B, 38 cases + 14 adversarial) | unsupported claims in the output: 0.067 ungated → **0.000** gated; citation validity 0.895 → **1.000**; unsafe adversarial outputs 0.214 → **0.000**; fault injection: **100/100** fabrications caught, 0/98 correct claims blocked. |
| E4 — retrieval | BM25 / dense / hybrid / hybrid+rerank on a 61-query gold set + unanswerable set: see [`results/e4.md`](results/e4.md) |
| E9 — efficiency | full single-image inspect in about 168 ms on CPU (p50); see `results/e9.json` |

Negative and null results are reported as measured; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Reproduce everything

| Command | Produces |
|---|---|
| `make data` | `data/raw/` stratified subset via CRC-verified HTTP range requests from the Zenodo zip (no full <!--np-->6.7<!--/np--> GB download) |
| `make corpus` | `corpus/manifest.csv`, `corpus/chunks.jsonl` from Europe PMC (CC BY / CC0 only) |
| `make eval` | `results/e1,e2,e4,e5,e9.json`, `results/results.json`, `results/results.md` with git SHA, config hash, data hash |
| `make test` / `make quality` | pytest with coverage / ruff, mypy, bandit, pip-audit |
| `make site` | `web/public/data/site.json`: real pipeline outputs rendered by the console |

CI (GitHub Actions) runs lint, types, security, tests on Python <!--np-->3.12<!--/np-->, the offline reviewer smoke run with a provenance check, the web build and a Docker build on every push; tags publish a Docker image to GHCR and a GitHub Release with `results/`.

## Data, models and licenses

| Resource | Use | License |
|---|---|---|
| OOC Image Dataset (Movčana et al. 2024, Zenodo <!--np-->10.5281/zenodo.10203721<!--/np-->) | evaluation; thumbnails on the console | CC BY 4.0 (Zenodo record); the article page lists CC BY-SA, so derived thumbnails are shared under CC BY-SA with attribution. Raw images are not redistributed; `data/download.py` fetches them. |
| Europe PMC open-access papers | retrieval corpus | CC BY / CC0 per paper, recorded in `corpus/manifest.csv` |
| Cohere `embed-v4.0`, `rerank-v3.5` | text embeddings, reranking | Cohere API terms (free trial key); offline hashed-vector fallback |
| Cloudflare Workers AI `@cf/meta/llama-3.1-8b-instruct-fast` | explainer | Workers AI terms (free daily allocation); deterministic template fallback |
| Qdrant (`qdrant-client`, local mode), rank-bm25, OpenCV, NumPy, SciPy, scikit-learn, FastAPI, FastMCP | engine | Apache-2.0 / BSD / MIT |

No personal, clinical or restricted data is used. No model is trained or fine-tuned; reference statistics are medians/MADs and kNN is non-parametric.

## License

Code: [MIT](LICENSE). Dataset and papers keep their own licenses (above).
