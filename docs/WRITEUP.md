# Zelos: Evidence-Gated QC Agent that Certifies Organ-on-a-Chip Images

*Subtitle: Flags organ-on-a-chip image problems, names the cause and the fix, cites open literature, and blocks any claim its data cannot support.*

## Category Declaration
**End-to-End System**

## Demo Video
[VIDEO LINK]

## Code Repository Link
https://github.com/simon-derock/zelos (tag `v1.0-submission`)

## Project Summary
Organ-on-a-chip (OoC) systems are in vitro human tissue models (lung, gut, vessel) grown in microfluidic channels for drug development. OoC labs image these cultures every day, and those images become the data behind tissue-model decisions, drug-response readouts and, increasingly, OoC digital twins. A defocused, saturated, vignetted or bubble-filled image silently corrupts that data. An LLM assistant asked to explain it may invent numbers or citations.

**Zelos** is an end-to-end, training-free agent that turns each brightfield image into a **certificate**. The certificate holds:
- a verdict (PASS, REVIEW or FAIL);
- the measured cause, with the recommended lab action;
- passages from open-access OoC literature, cited;
- a **claim ledger** that records, for every sentence, the tool output or passage that supports it.

The design rule is *the LLM proposes, tools compute, the gate disposes*:
- 17 physics-based metrics and a fixed rule table compute the verdict and the cause, with 0 LLM calls.
- Cohere embeddings in Qdrant, combined with BM25 and Cohere reranking, retrieve the evidence.
- A Cloudflare Workers AI model writes the explanation as typed claims.
- A deterministic gate checks every number against the tool outputs and every citation against the retrieved passages. It repairs a claim once, then blocks it.

On the expert-labelled public **OOC Image Dataset** (480 real images, six cell lines, split by experiment session):
- Zelos flags 97.1% of injected corruptions at severity ≥3 and names the injected cause 31.0% of the time (chance: 11.1%).
- Against the expert good/bad label, non-parametric kNN on the same features reaches AUROC 0.710, while the artifact score does not predict it. The expert label is a culture-state signal, not an image-artifact signal.
- The gate cuts unsupported claims from 0.067 to 0.000 and unsafe adversarial outputs from 0.214 to 0.000. It caught 100 of 100 injected fabrications.

Everything runs on CPU, and every published number regenerates offline with one command. Research use only.

## Technical Report

### 1. Team
Philip Simon Derock: team leader, AI and software engineering. No team member claims biology or clinical expertise, so the cross-disciplinary bonus is not claimed.

### 2. Problem and application scenario
- **The workflow.** Automated brightfield microscopy produces hundreds of OoC images per experiment, and expert review cannot keep pace. Bad images create two separate risks: wrong conclusions from artifacts, and silent contamination of shared datasets and digital-twin training data.
- **What the challenge asks for.** The challenge sponsor stresses the *standardization and assetization* of OoC data. Zelos is the gate in front of that asset: every image gets a machine-readable certificate (SHA-256 of the image, metrics, verdict, cause, action, evidence, ledger) before it enters a dataset.
- **Who uses it, and how:**
  - a technician running `zelos triage <folder>` after an imaging run;
  - a data steward certifying images before they are deposited;
  - any agent that can use MCP tools, through the Zelos MCP server and its skill file.

### 3. Data, licenses and compliance
- **Images:** the Organ-on-a-Chip (OOC) Image Dataset (Movčana et al., *Data* 9(2):28, 2024), Zenodo 10.5281/zenodo.10203721.
  - 3,072 brightfield PNGs (1542×2056, 8-bit) from six cell lines: A549, Caco-2, HPMEC, HSAEC, HUVEC, NHBE.
  - Each image has an expert good/bad decision, the time after seeding, and for some images the seeding density and flow rate.
  - License: CC BY 4.0 on the Zenodo record; the article page lists CC BY-SA. We follow the stricter one: derived thumbnails are shared under CC BY-SA with attribution, and raw images are never redistributed.
- **Subset:** `data/download.py` reads the remote zip's central directory and fetches a seeded subset of 480 images, stratified by label and cell line, with CRC-checked byte-range requests. The full 6.7 GB archive is never downloaded.
- **Splits are by experiment session** (the date prefix of the image id): 30% of sessions form dev, the rest form test, and no session appears in both.
- **Literature corpus:** 48 open-access papers (2,379 chunks), all CC BY, from Europe PMC. Each paper's license and URL are stored in `corpus/manifest.csv`.
- **Compliance:** no personal, clinical or restricted data is used.

### 4. Method
**Vision QC (0 LLM calls).** 17 classical metrics are computed on a copy downscaled to a 1024-pixel long side:
- three focus measures, each divided by image variance so it does not depend on contrast: Laplacian variance, Tenengrad gradient energy, and the high-frequency share of the FFT spectrum;
- RMS contrast and the p99–p1 intensity range;
- mean intensity and the fraction of clipped pixels at each end;
- illumination flatness, from a degree-2 polynomial fitted to the background;
- the Immerkær noise estimate;
- debris, from black-hat filtering and connected components;
- a bubble score, from round regions with no texture;
- a confluence proxy, blob density, LBP texture entropy, and 8×8 block artifacts.

**Scoring.** Each metric becomes a one-sided robust z-score (median and MAD) in the direction that indicates a problem, against a reference:
- the reference is either the dev-session "good" images, or the batch of images being triaged;
- the QC score is the largest z; REVIEW above 3 and FAIL above 5. These thresholds were set in advance and a sweep over them is reported.
- A fixed, human-readable rule table maps metrics to 9 cause labels, and each label to a lab action (re-focus, flat-field, degas or flush the channel, re-expose, and so on).

**Retrieval.**
- Each cause label maps to a query template.
- Retrieval combines Cohere `embed-v4.0` vectors in Qdrant (local embedded mode) with BM25 by reciprocal rank fusion (k = 60), then reranks with Cohere `rerank-v3.5`.
- It abstains when the best rerank relevance falls below a threshold set in advance.

**Explainer.** Cloudflare Workers AI (`@cf/meta/llama-3.1-8b-instruct-fast`, temperature 0) receives the QC tool output and the evidence, and must return JSON claims typed numeric, literature or inference, each with evidence ids.

**Grounding gate (0 LLM calls).** Every claim is checked in three ways:
- **Numeric claims:** each number must match a tool output within a relative tolerance of 10⁻³ or after rounding, and must be bound to the right metric.
- **Literature claims:** the cited chunk must be one that was actually retrieved, and it must support the sentence (lexical overlap above a threshold).
- **Inferences:** they must cite a claim that is already supported.

A claim that fails is sent back for one repair call; if it still fails, it is blocked and kept in the ledger with the reason. Requests for a diagnosis, and prompt injections, are refused by the router before any LLM call. Each run uses at most 2 LLM calls.

**Agent packaging.**
- A router dispatches requests to `inspect`, `triage`, `ask`, `certify` or `refuse`.
- The same tools are available through the CLI, FastAPI, an MCP server and a skill file (`skills/zelos-qc/SKILL.md`).
- The web console shows only real exported pipeline outputs.

### 5. Implementation
- **Stack:** Python 3.12 with OpenCV, NumPy, SciPy, scikit-learn (metrics and kNN only), rank-bm25, qdrant-client, Cohere, FastAPI and FastMCP. The web console is Vite with React and TypeScript, hosted on Netlify.
- **Hardware:** CPU only. A full inspection takes about 168 ms per image (p50, measured in E9).
- **Reproducibility:** responses from both API services are recorded and replayed, so `make judge` runs with no keys.
- **CI on every push:** ruff, mypy, bandit, pip-audit, pytest with coverage, an offline smoke run with a provenance check, the web build, and a Docker build. Tagged releases publish a Docker image to GHCR and attach `results/`.

### 6. Experiments and results
All values come from `results/results.md` (git SHA and config hash are inside).

**E1: injected corruptions, so the correct answer is known**
- Setup: 60 test-session images × 11 corruption kinds × 5 severities.
- Detection: 97.1% of severity ≥3 corruptions are flagged.
- Cause attribution among flagged images: 31.0% (top-2: 49.0%), against 11.1% by chance.
- AUROC: 0.778 for clean vs corrupted, against 0.629 for a focus-only baseline.
- **Honest failure:** the pre-registered dev-session reference also flags 76.7% of clean test images. This is a cross-session shift in acquisition (contrast and compression statistics differ between sessions).
- A batch-relative reference (unsupervised, added after seeing that result, and disclosed) cuts clean false flags to 46.7%, with AUROC 0.728 and attribution 40.6% (top-2: 57.0%).

**E2: agreement with the expert good/bad label**

Test set: 301 images from held-out sessions.

| Detector | AUROC [95% CI] |
|---|---|
| Artifact QC score | 0.393 [0.332, 0.459] |
| Same, session reference | 0.477 [0.413, 0.541] |
| Focus only | 0.524 |
| Random | 0.518 |
| kNN label transfer on the same features (k = 15, non-parametric) | **0.710 [0.653, 0.767]** |

Conclusion: the expert decision tracks culture state, not image artifacts. Zelos therefore reports image trust and culture state separately.

**E4: retrieval** (48 papers; 61 gold queries plus 12 unanswerable)

| Mode | Paper-level MRR |
|---|---|
| BM25 | 0.944 |
| Cohere dense | 0.973 |
| Hybrid | 0.951 |
| Hybrid + rerank | 1.000 (Hit@1 CI [0.941, 1.000]; chunk-level Hit@1 0.852) |

- Hybrid + rerank abstained on all 12 unanswerable questions and wrongly abstained on 0 of 61 answerable ones.
- The gold set is optimistic, because its questions were written from the corpus itself (see Limitations).

**E5: the grounding gate** (38 cases plus 14 adversarial prompts; template vs LLM without gate vs LLM with gate)

| Measure | LLM without gate | LLM with gate | Template |
|---|---|---|---|
| Unsupported claims in the output | 0.067 | **0.000** | 0.000 (by construction) |
| Valid citations | 0.895 | **1.000** | — |
| Unsafe adversarial outputs | 0.214 | **0.000** | — |
| Abstention recall | 0.625 | **1.000** | — |

- Fault injection: 100 of 100 fabricated numbers, fake citations, wrong support, uncited literature and orphan inferences were caught; 0 of 98 correct claims were blocked.
- Cost of the gated LLM: 1,280 tokens and 0.87 LLM calls per answer, 1.9 s p50.
- With the gate, the LLM yields more supported claims per case than the template (6.15 vs 4.90 for QC explanations), so the LLM earns its place only behind the gate.

### 7. Reliability analysis and limitations
See `docs/LIMITATIONS.md`, which ties every limitation to a measurement:
- the shift between imaging sessions;
- imperfect cause attribution (blur is often read as illumination or contrast, bubbles as debris);
- the expert label is not an artifact label;
- data from a single lab and a single public dataset;
- an optimistic retrieval gold set;
- heuristic support checks (lexical; the vector arm is off in E5);
- a small 8B LLM and a small E5;
- certificates are content hashes, not signatures.

QC "confidence" is expressed as reference percentiles and the margin to the threshold. It is not a calibrated probability.

### 8. Impact
- **Who benefits:** OoC labs and data stewards. Zelos gives an auditable, explainable quality gate that runs on a lab PC with no GPU, no training data and no paid service.
- **Why it extends easily:** every component is training-free and configured, not learned. Adapting it to a new chip or microscope means a new reference batch, with no retraining.
- **Roadmap:**
  - longitudinal monitoring of each well;
  - neural OoC and calcium-imaging modalities;
  - certificates that feed digital-twin data pipelines;
  - an entailment model inside the gate;
  - the expert culture-state signal (E2) as a second tool.

### 9. Reproduction
```bash
git clone https://github.com/simon-derock/zelos && cd zelos
make setup && make judge          # offline, no keys: tests + E4 + E5 replay + provenance check
make data && make eval            # real 480-image subset from Zenodo + E1/E2/E9
make site && cd web && npm ci && npm run dev   # console with real outputs
```

### 10. External models, libraries, datasets and tools
- **Dataset:** the OOC Image Dataset (CC BY 4.0 / CC BY-SA).
- **Literature:** Europe PMC open-access papers (CC BY).
- **Models and services:** Cohere `embed-v4.0` and `rerank-v3.5` (Cohere API terms, free trial key); Cloudflare Workers AI `@cf/meta/llama-3.1-8b-instruct-fast` (Workers AI terms, free daily allowance).
- **Libraries (Apache-2.0, BSD or MIT):** Qdrant client, rank-bm25, OpenCV, NumPy, SciPy, pandas, scikit-learn, FastAPI, FastMCP, React, Vite.
- **Fallbacks:** each external service has an offline one (hashed vectors, BM25 order, a deterministic template).
- **Prior work:** none reused; all code was written for this challenge.

## Optional Demo Link
https://zelos.philipsimonderock.com: a static console showing real pipeline outputs (no login). If it is unreachable, follow the local run instructions above; the video shows the same console.
