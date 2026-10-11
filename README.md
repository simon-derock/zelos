<div align="center">

# Zelos

### Every organ-on-a-chip image, certified.

An evidence-gated QC agent for in vitro organ-on-a-chip cultures. It finds the imaging fault, names the lab fix and the next step, cites open organ-on-a-chip research, and **blocks every claim the evidence cannot support.**

[![CI](https://github.com/simon-derock/zelos/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/simon-derock/zelos/actions)
![License](https://img.shields.io/badge/license-MIT-green)
![Runs](https://img.shields.io/badge/runs-CPU_·_offline-black)

**[Live console](https://zelos.philipsimonderock.com)** · **[Demo video](https://youtu.be/bhoM2jl5zM4)** · [Results](results/results.md) · [What changed in v1.1](docs/V1_1_CHANGES.md) · [Limitations](docs/LIMITATIONS.md)

<sub>AI4S Open Innovation: AI for Life Science · 5th Pazhou Algorithm Competition · <b>End-to-End System</b> · Built by Philip Simon Derock</sub>

</div>

> **Which version is this?** The submitted and evaluated version is tag [`v1.0-submission`](https://github.com/simon-derock/zelos/tree/v1.0-submission) on `main`. This `main` branch is **v1.1, built after the submission deadline** for the final round. Corrections to v1.0 are listed in [docs/V1_1_CHANGES.md](docs/V1_1_CHANGES.md).

---

## Measured, not claimed (v1.1, every number from [`results/results.md`](results/results.md))

| | |
|---|---|
| **100/100** | injected fabrications (fake numbers, fake citations, unsupported inferences) blocked by the grounding gate; 0 of 98 correct claims blocked |
| **0.067 → 0.000** | unsupported claims in LLM answers, without vs with the gate; valid citations 0.895 → 1.000 |
| **66.4% / 63.3%** | injected imaging faults (severity ≥3) caught on real chip images from unseen sessions, single image / session reference, at 1.7% / 3.3% false flags |
| **81.2% / 85.4%** | of caught faults, the correct cause is named (chance 11.1%) |
| **0.882 / 0.913 AUROC** | clean vs faulty, single image / session reference; a label-trained DINOv2 head on the same faults reaches 0.710 and cannot name a cause |
| **0.767 AUROC** | culture-state signal (frozen DINOv2 + kNN over expert-labelled neighbours, no model trained) against the expert good/bad label on 2,110 held-out images from 39 unseen sessions |
| **4 of 9** | cause → action rules have a verified literature sentence in the open corpus; the other 5 are reported as unsupported |

**What the evaluation also says, plainly**
- The image-fault score does **not** predict the expert good/bad label (AUROC 0.571 with a session reference; confidence intervals include 0.5). That label tracks culture state, which is why Zelos reports image trust and culture state as two separate outputs.
- Zelos does not claim state-of-the-art expert-label accuracy; its culture-state signal is a lookup against labelled neighbours, not a tuned classifier.
- Absolute thresholds do not transfer across imaging sessions (leave-one-session-out false flags 21.3% at the most conservative cutoffs); scoring relative to the session does (9.7% leave-one-session-out, 8.8% on test).
- General vision LLMs asked "good or bad?" flag 40.0% and 65.0% of clean images (n = 20 per group, one prompt).
- Illumination, debris and motion-blur faults are mostly missed at severity ≤3 (per-kind table in the results).

## How it works

**The LLM proposes. Tools compute. The gate disposes.**

```
brightfield image ─► 17 physics QC metrics ─► robust z vs reference ─► calibrated cutoffs ─► cause (signature centroids) + action + route ─► cited basis
                     0 LLM calls               session-relative or stored   0 LLM calls         0 LLM calls
        └► (optional) frozen DINOv2 kNN culture-state ─► evidence retrieval ─► LLM explainer ─► grounding gate ─► certificate + claim ledger
```

- **Verdict:** PASS / REVIEW / FAIL from per-metric calibrated cutoffs (false-alarm rate chosen leave-one-session-out). **Route:** USE, RE-IMAGE (fix at the microscope), INSPECT-CHIP (channel or optics), RE-EXPORT, REVIEW.
- **Cause:** nearest signature centroid in z-space, fitted on faults injected into development-session images, with the rule table as fallback. **Action** and **literature basis** come from a fixed table whose citations are checked sentence by sentence.
- **Grounding gate:** every number must match a tool output, every citation must be a retrieved passage that supports the sentence, every inference must rest on supported claims. Failing claims get one repair, then are blocked and logged.
- **Interfaces:** CLI (`zelos inspect | explain | triage | ask`), FastAPI with upload and session triage, an **MCP server** and [`SKILL.md`](skills/zelos-qc/SKILL.md), a web console, and a Docker packaging for a free Hugging Face Space ([docs/SPACE.md](docs/SPACE.md)).

## Quickstart

```bash
git clone https://github.com/simon-derock/zelos && cd zelos
make setup        # pinned deps
make judge        # tests + retrieval + grounding replayed offline: no API keys needed
make features     # optional: stream all 3,072 real images from Zenodo (about 45 min), keep only features
make eval         # regenerate every v1.1 result
```

Live Cohere and Cloudflare Workers AI calls are optional (`.env.example`). Every response used in the published results is committed and replayed.

## Data and licenses

| Resource | Use | License |
|---|---|---|
| Organ-on-a-Chip (OOC) Image Dataset, Movčana et al., *Data* 2024, Zenodo <!--np-->10.5281/zenodo.10203721<!--/np--> | evaluation on all 3,072 in vitro images (lung, intestinal, endothelial cell lines on chip), expert good/bad labels | CC BY <!--np-->4.0<!--/np--> (the article lists CC BY-SA; derived thumbnails shared under CC BY-SA). Images are streamed and discarded; only features are kept. |
| 48 Europe PMC open-access papers | retrieval corpus and rule citations (`corpus/manifest.csv`) | CC BY |
| DINOv2-small (Meta AI) | frozen feature extractor | Apache-2.0 |
| Cohere `embed-v4.0`, `rerank-v3.5` · Workers AI `llama-3.1-8b-instruct-fast` | embeddings, reranking, explanations | provider terms; offline fallbacks included |

No patient, clinical or personal data. No model is trained or fine-tuned; the only learned baseline (a linear head on DINOv2 features) is evaluated for comparison and is not part of Zelos.

> Research tool. Not a medical device. No clinical or diagnostic claims.

Code: [MIT](LICENSE)
