# PLAN_SPEC.md — OoC Copilot
**Training-free, evidence-gated image-QC + literature agent for organ-on-a-chip (OoC) cultures**
Target: AI4S Open Innovation: AI for Life Science (5th Pazhou Algorithm Competition track, hosted on Kaggle)
Category to declare: **End-to-End System**
Hard deadline: **Oct 10, 2026, 9:29 PM GMT+5:30** (read from the Kaggle Writeup page; re-check the live countdown).

> Written for an autonomous coding agent. Nothing here is removed for brevity. Order of execution (Section 14) guarantees a valid submission exists at every checkpoint, so later items can be finished or refined without risking the submission.
> Status legend used for facts: **[K]** shown on the Kaggle page, **[C]** found during our research and quoted from a source, **[U]** from the user's own profile/repos, **[?]** unverified, must be checked at build time. Never turn a **[?]** into a claim in the report.
> Honest framing: nobody can guarantee a top-3 result. Selection is by expert review of Writeups. This plan maximizes expected score on the published rubric and removes the failure modes that sink otherwise good entries.

---

## V2 — ZELOS (supersedes any conflicting text below; written 2026-10-10 20:08 IST after reading all 12 Kaggle pages)

### V2.1 Identity and decisions
| Item | Decision |
|---|---|
| Product name | **Zelos** — the evidence-gated QC agent that certifies organ-on-a-chip images as trustworthy data assets |
| Repo | https://github.com/simon-derock/zelos (public, MIT, repo root = this folder). Frequent micro-commits, push after every working unit |
| Category | End-to-End System ("from data input to **actionable output**" [K]) |
| Data | Real OoC dataset, Zenodo 10.5281/zenodo.10203721 (3,072 PNG, 6 cell lines, **expert good/bad labels**, time after seeding; date prefix of image id = experiment session → group splits). Zenodo license CC-BY-4.0; MDPI page reportedly CC-BY-SA → follow the stricter one; never redistribute images, ship `data/download.py` (range-request stratified subset, CRC-verified) |
| Embeddings | Cohere `embed-v4.0` (verify at build) · offline fallback: hashed TF-IDF vectors (no torch in the API image) |
| Rerank | Cohere `rerank-v3.5` (one call per search; verify at build) · fallback: none (RRF order) |
| Vector DB | Qdrant (`qdrant-client`, local embedded mode `QdrantClient(path=...)`; same API as Qdrant server/cloud) |
| LLM | Cloudflare Workers AI (OpenAI-compatible chat endpoint; small instruct model, verify at build) · fallback: deterministic template explainer |
| Old projects | **Not used.** No xphil / dex-jobs / Titan-resume code or patterns; everything written new for this challenge (no prior-work disclosure needed beyond "none") |
| Frontend | Not Streamlit. Apple-grade product site + live console built by a **Sonnet sub-agent** once the API contract is frozen; deployed on **Netlify** at **zelos.philipsimonderock.com** (CNAME → Netlify; DNS set by the user). The demo video is recorded on this site |
| API | FastAPI (`src/api/main.py`), deployed on Render free (Docker, no torch); Netlify proxies `/api` → no CORS. Per-visitor + daily caps protect free API quotas; strict CSP |

### V2.2 Rules that shape the build (quoted from Kaggle pages)
- "reproducible by reviewers **without requiring paid services**… if external APIs… are used… describe… **alternative running options**" → all three services have free tiers; **every path also runs keyless offline**; Cohere vectors and LLM responses used by `evaluate.py` are cached and committed (record/replay), so published numbers reproduce without keys.
- Rules §4: commercial models/APIs allowed; disclose name, source, purpose → `docs/SOURCES.md` + report §11.
- Rules §6: fabrication of results/demos prohibited → provenance test + every number from `results/results.json`. The live site shows only real pipeline outputs.
- Foundational 6c: any open-source code used must be OSI-licensed without commercial restriction → all deps BSD/MIT/Apache.
- Bonus +0.5 goes to an "**Interpretability and Reliability**" dimension that is not in the public 5-criterion list → judges score it internally; Zelos's ledger/gate is aimed squarely at it.
- Abstract keywords "innovative, interpretable, explainable, applicable" — each maps to a named Zelos feature in the video and report.
- Sponsor CellShells: "standardization, assetization… data IP… AI-driven OoC digital twins" → pitch: Zelos is the **certification gate** in front of any OoC data asset or digital twin (SHA-256, metrics, verdict, cause, action, cited evidence, audit ledger per image).
- Video: Kaggle attachment or public link (YouTube/Bilibili/Loom), ≤5 min, no login. Report may be written inline in the Writeup.

### V2.3 Upgraded agent design — "LLM proposes, tools compute, the gate disposes"
```
request ─► Router (rule-based intent, 0 LLM) ─► typed plan
   inspect(image)   : load → qc_metrics → reference_compare → attribute → recommend_action → similar → retrieve → explain → gate → certificate
   triage(session)  : parallel qc over a folder → session stats → dominant causes → re-image list → one LLM summary → gate
   ask(question)    : retrieve → (abstain if no support) → explain → gate
   certify(image)   : inspect → signed JSON certificate (hashes + ledger)
```
| Specialist | Job | LLM calls |
|---|---|---|
| Vision QC | 15 physics-meaningful metrics on a downscaled copy (focus ×3, contrast ×2, clipping ×2, illumination, noise, debris ×2, bubble, confluence, blob density, texture) | 0 |
| Reference comparator | robust z vs session-independent reference (median/MAD per cell line), percentile bands, density state | 0 |
| Artifact attributor | fixed a-priori rule table → cause labels + triggering metric values | 0 |
| Action recommender | cause → lab action (re-focus, degas/check channel for bubble, flat-field, re-expose, exclude) from `configs/actions.yaml` | 0 |
| Similar images | kNN on handcrafted descriptors (non-parametric) | 0 |
| Evidence retriever | Cohere embed-v4.0 in Qdrant + BM25 → RRF (k=60) → Cohere rerank-v3.5 → top-5 with license/URL | 0 |
| Explainer | Workers AI, strict JSON claims with evidence ids | 1 |
| Grounding gate | numeric match vs tool outputs; citation must be a retrieved chunk with lexical+vector support; inference must cite supported claims | 0 |
| Repair | rewrite or drop unsupported claims | ≤1 |
| Calculator | safe AST evaluator for any derived number | 0 |
Efficiency targets: ≤2 LLM calls per image, ≤1 per session triage; QC < 150 ms/image CPU; caches keyed by SHA-256; batch QC in a process pool. Every step logged (latency, tokens) to `runs/<ts>/trace.jsonl`; every run writes a manifest (commit, model ids, dataset + corpus SHA-256, config hash). Same-model rule: one locked LLM per run, no silent fallback inside a measured run.

### V2.4 Measurement suite (Stellium-grade)
| ID | What | Metrics | Comparisons |
|---|---|---|---|
| E1 | Controlled-corruption benchmark (9 corruptions × 5 severities on real OoC images) | Spearman ρ per metric, AUROC clean vs sev≥3, **cause-attribution accuracy**, confusion matrix, bootstrap CIs | random, focus-only |
| E2 | Expert good/bad agreement, session-grouped split | AUROC, balanced acc, F1, κ, Wilson/bootstrap CIs, per cell line | random, majority, focus-only, kNN label transfer |
| E4 | Retrieval on a gold set written from the corpus (+ reworded + unanswerable) | Hit@1/3/5, MRR, nDCG@5, context recall, abstention accuracy | BM25 vs Cohere dense vs hybrid RRF vs hybrid+rerank |
| E5 | Grounding: template vs LLM-ungated vs LLM-gated on image cases + adversarial prompts | grounded-claim rate, unsupported-claim rate, citation validity, refusal correctness, tokens/answer, LLM calls, latency p50 | "when is the LLM worth it" verdict per case type |
| E6–E9 | Reproducibility hash match, ablations, failure analysis, latency | as in Section 9 | |

### V2.5 Engineering bar
pytest (unit + contract + API), ruff lint+format, mypy, bandit, pip-audit, coverage gate, all in GitHub Actions on every push; Docker build job; release workflow on `v*` tags; Dependabot; offline CI (no secrets); `.env.example`; no secrets in git.

### V2.6 Timeline (deadline 21:29:59 IST)
| By | Done |
|---|---|
| 20:15 | repo wired, data subset downloading |
| 20:35 | vision QC + tests + CI green, E1/E2 running |
| 20:40 | API contract frozen → Sonnet frontend sub-agent launched |
| 20:55 | corpus + Qdrant/Cohere retriever + gate + Workers AI agent + E4/E5 |
| 21:05 | API on Render, site on Netlify, README/report, tag `v1.0-submission` |
| 21:05–21:20 | user records video on zelos.philipsimonderock.com |
| 21:20–21:28 | Writeup filled, **Submit** |

---

## 0. Instructions to the coding agent (read first)

1. Work in the repo root `ooc-copilot/`. Follow the tree in Section 10 exactly; do not rename top-level paths.
2. **Zero gradient updates anywhere.** No model training or fine-tuning. Allowed: classical image processing, descriptive statistics (median/MAD, percentiles), frozen pretrained encoders used only as feature extractors, non-parametric kNN/indexing, BM25, pretrained LLM inference. Any scalar threshold tuned on data must be tuned only on the dev split, listed in `configs/default.yaml`, and disclosed in the report.
3. **No hardcoded results, ever.** Every number in the README, report, video, and Writeup must come from `results/*.json` produced by `evaluate.py`. A provenance test (T27) enforces this.
4. **No fabricated citations or links.** Citations come only from `corpus/manifest.csv` entries that were actually downloaded. Every external fact in the report needs a source in `docs/SOURCES.md`.
5. Deterministic: seed everything (`seed: 1337`), pin versions, log git SHA + data hash into every results file.
6. CPU-only must work. GPU is optional. The full demo must run without any paid API and without the internet after `make data corpus` has been run once (LLM via local Ollama or template mode).
7. Every module ships with a unit test and a docstring stating inputs/outputs. Run `make test` before each commit. Commit early and often; tag `v0.1-valid` as soon as the minimum valid submission exists (Section 14, C5).
8. Research use only. The UI, README, and report state: "Research tool. Not a medical device. No clinical or diagnostic claims."
9. When the spec says **[?]**, write a discovery script/print statement, look at the real data, then adapt. Never guess column names, file formats, or sizes.

---

## 1. Mission, constraints, definition of done

**Mission.** Give an OoC lab a tool that (a) inspects brightfield culture images, (b) flags quality problems/artifacts and abnormal culture states, (c) explains each flag using cited OoC literature, and (d) **refuses to state anything the computed data or retrieved text does not support**, with every claim logged in an auditable ledger.

**Constraints.**
- ~2 hours remain at spec time. Single team (1–5 members). Free tools only. No model training. CPU-friendly.
- Reviewers must reproduce without paid services, proprietary hardware, or private data (rule on Kaggle page **[K]**).
- Video ≤ 5 min, YouTube-hosted, viewable without login **[K]**.

**Definition of done (all must be true at final submit).**
- [ ] Registration Google Form completed (otherwise ineligible **[K]**).
- [ ] Public GitHub repo, tagged `v1.0-submission`, fresh-clone reproducible with the commands in the README.
- [ ] `evaluate.py` regenerates `results/results.json`, `results/results.md`, and all figures.
- [ ] Demo video (≤ 5:00) public on YouTube, embedded in the Writeup media gallery.
- [ ] Technical report (PDF in repo, link inserted in Writeup) + Writeup body filled per template.
- [ ] Kaggle checklist 7/7 and status **Submitted** (not Draft) before the deadline.
- [ ] All links tested in a private/incognito window.

---

## 2. Competition facts (from the Kaggle pages)

| Item | Detail |
|---|---|
| Name | AI4S Open Innovation: AI for Life Science **[K]** |
| Tagline | AI + Organ-on-a-Chip: Open Innovation Challenge for In Vitro Life Systems **[K]** |
| Host | ZHOU YINGTONG; part of the 5th Pazhou Algorithm Competition; sponsor/domain support: CellShells Bioscience Co., Ltd. (neural organ-on-a-chip; aims at AI-driven OoC digital twins) **[K]** |
| Format | Single track; official submission = **Kaggle Writeup** (demo video + public code repo + technical report). Not a leaderboard competition **[K]** |
| Timeline | Preliminary Aug 15–Oct 10, 2026; Semi-final Oct 10–20 (expert review picks **20 teams**); Final Oct 20–30 (online presentation and defense); results before Nov 2026 **[K]** |
| Prizes | 1st $11,850, 2nd $7,400, 3rd $2,950 (total $22,200, paid in CNY); no Kaggle points/medals **[K]** |
| Team | 1–5 members, one leader, each person on one team only **[K]** |
| Bonus | +0.5 in the "Interpretability and Reliability" dimension for teams spanning AI/CS **and** biology/bioengineering/clinical expertise; declare composition in the report **[K]** |
| Data | Open: any public/private/collaborative/simulated/synthetic data; must be compliant, transparent, reproducible; real OoC data strongly encouraged; no unauthorized personal/clinical/restricted data **[K]** |
| Registration | Separate Google Form is mandatory: `https://docs.google.com/forms/d/e/1FAIpQLSdRAat5jIunRaFNh_NntsVeJUnekEJDrbuokLZ32LFgCwPtiA/viewform?usp=publish-editor` **[K]** |
| Official site | https://www.aicompetition-pz.com/ **[K]** |
| Writeups | Resubmission allowed before deadline; latest version in "Submitted" status at the deadline is the final one; one valid Writeup per team; writeups are hidden until hackathon close **[K]** |
| Page listing | Writeups tab showed ~20 submitted teams at spec time; earlier page stats showed 58 teams / 325 entrants. Expect more submissions in the last hours. |

**Judging rubric [K]:**

| Criterion | Weight | What reviewers look for |
|---|---|---|
| Problem importance & potential impact | 30% | Important life-science problem; meaningful, practical or scientific value |
| Technical approach & innovation | 30% | Sound and innovative AI method; clearly explained implementation |
| Results & validation | 20% | Reliable, well-demonstrated results; meaningful evaluation, comparisons, experiments |
| Reproducibility & implementation quality | 10% | Reproducible/extendable; documented, accessible code/models/data; low testability is penalized |
| Presentation quality | 10% | Video tells a clear story: problem, approach, demo, results |

"Show, don't just tell" is the organizer's own instruction **[K]**.

**Required deliverables [K]:**
1. Category declaration at the start of the Writeup.
2. Demo video (≤5 min; public/no login; shows the actual system, inputs, workflow, outputs; all third-party material must have usage rights; if >5 min only first 5 are guaranteed reviewed).
3. Public code repo (core code, README.md, environment files, model files or download instructions, test/inference/eval/demo scripts, input/output descriptions, reproduction steps, clear entry script such as demo.py/inference.py/evaluate.py; if external APIs/cloud/commercial models are used, document configuration, limits, alternatives).
4. Technical report (in the Writeup or public PDF link; concise and self-contained; ideally 15–20 pages excluding references/appendices).
5. Optional: interactive demo/online app/Notebook App/API/agent link, publicly accessible without login, with limitations and backup material (screenshots/video/local instructions).

---

## 3. How to win — top-3 strategy

### 3.1 What the selection process actually rewards
Reviewers read a Writeup, skim a repo, and watch a ≤5-minute video, for ~20 finalists out of everything submitted. 60% of the score is "important problem" + "innovative, sound method". 20% is validation. Only 20% is reproducibility + presentation. Strategy: **win the 60% with a sharp, defensible idea, lock the 20% with real measured results, and bank the last 20% with an effortless repo and a tight video.**

### 3.2 Expected competitor archetypes (assumptions, not facts)
| Archetype | Typical weakness | Our counter |
|---|---|---|
| Polished showcase on **synthetic data** with a fancy generator and impressive-looking metrics (we inspected one public Kaggle notebook of this type: no training loop, metrics typed in as strings, rule-based "predictions", data produced by the same simulator the analyzer reads) | Fails "Results & Validation" and "Reproducibility" under expert scrutiny | Real OoC images, every metric computed by code, controlled-corruption benchmark with known ground truth, one-command reproduction, honest failure analysis |
| Heavy deep-learning model, thin validation | Not reproducible in review time; no baseline; leakage | Training-free, CPU-fast, baselines + group-aware splits + CIs |
| "Chat with papers" RAG wrapper | Little scientific novelty; hallucination risk | Evidence-gated claim ledger with measured grounding rate and abstention behavior |
| Idea decks without working code | Eliminated on reproducibility | Working end-to-end demo + tests + Docker |

### 3.3 Our five differentiators (state them explicitly in the video and report)
1. **Real OoC data, real metrics.** Uses a public expert-evaluated OoC brightfield dataset (Section 5.1) instead of a simulator. Every number is regenerated by `evaluate.py`.
2. **Training-free, physics-aware QC with artifact attribution.** Not "an image is bad" but *why*: defocus, noise, uneven illumination, clipping/exposure, debris/bubbles, low contrast, abnormal density. Validated against known injected corruptions (ground truth) and against expert labels where available.
3. **Trust layer: claim ledger + grounding gate.** Every statement is typed (numeric / literature / inference), linked to a tool output or retrieved chunk, and marked supported / unsupported / abstain. Ablation shows grounding rate with gate off vs on, across LLM variants including a no-LLM template baseline.
4. **Agent-native packaging.** Tools exposed through an MCP server plus a `SKILL.md` and `AGENTS.md`, so any MCP-capable agent can QC OoC images and cite evidence. This is the "applicable" part of "interpretable, explainable, applicable" **[K]**.
5. **Radical honesty.** Negative results, limitations, worst-case failures, and threshold sensitivity are reported. Reviewers who have seen inflated entries reward this.

### 3.4 Score-engineering checklist (map every rubric line to evidence)
| Rubric line | Evidence artifact | Where the judge sees it |
|---|---|---|
| Importance (30%) | Section "Problem & use case" with the lab workflow (daily brightfield monitoring), cost of undetected bad images, who benefits; sourced citations only | Writeup summary, report §1, video 0:00–0:30 |
| Innovation (30%) | Differentiators 2–4 above, architecture diagram, ablations proving each part matters | Report §3–4, video 0:30–1:00 |
| Validation (20%) | E1–E9 in Section 9, with baselines, CIs, ablations, failure analysis | `results/results.md`, report §5, video 3:00 |
| Reproducibility (10%) | `make judge` smoke test (<5 min CPU), Dockerfile, pinned deps, CI badge, `JUDGES.md` | README top, `JUDGES.md` |
| Presentation (10%) | Story-driven ≤4:45 video; a single hero figure; captions | Media gallery |
| Bonus +0.5 | Biologist team member **or** documented expert review of the artifact-attribution table (see 3.6) | Report team section |

### 3.5 Make the reviewer's job trivial
- README first screen: one-paragraph pitch, hero figure, **"Judge Quickstart" (3 commands)**, results table pasted from `results.md`.
- `JUDGES.md`: a table mapping each rubric criterion to the exact file/section/command that evidences it.
- `make judge` runs a small subset (≈100 images, retrieval + grounding smoke tests) in under ~5 minutes on CPU and prints a PASS/FAIL summary.
- Template mode (no LLM) works out of the box so reviewers are never blocked by Ollama setup; LLM mode is an optional upgrade.

### 3.6 The +0.5 cross-disciplinary bonus
- If any biologist/bioengineer can join in the next 30 minutes, add them to the Kaggle team (max 5) and ensure they complete registration as required. Declare roles in the report.
- If not, do **not** claim biology expertise. Ask a biology contact to review the artifact-attribution table and limitations and record the review in Acknowledgements (this does not earn the bonus, but improves credibility).

### 3.7 Pre-mortem (run at T-40 and T-15)
Ask: "A skeptical expert reads only our README, runs the Quickstart, and watches the video. Where do they lose trust?" Fix, in order: anything not reproducible, any claim without a source, any metric without a baseline, any unexplained failure.

### 3.8 If we reach the top 20 (defense Oct 20–30 [K])
See Section 18. Do not change `main` after the deadline; keep work on a branch and link the tagged submission commit.

---

## 4. Concept and architecture

### 4.1 One-paragraph description
OoC Copilot takes a brightfield microscopy image of an organ-on-a-chip culture and returns a **QC card**: PASS / REVIEW / FAIL, the measured quantities behind it, the most likely artifact type(s), similar reference images, and a short explanation with citations to open-access OoC literature. A claim-ledger gate verifies each statement against tool outputs and retrieved passages and blocks or abstains on anything unsupported. All components are training-free and run on CPU.

### 4.2 Data flow
```
image(s) ──► load/normalize ──► QC metric suite ──► robust z-scores vs reference ──► QC score + flags
                                   │                                  │
                                   │                         artifact attribution rules
                                   ▼                                  ▼
                       (optional) frozen embedding ──► kNN similar images / metadata recovery
                                                                      │
                     question/flag ──► query builder ──► hybrid retrieval (BM25 + dense, RRF) ──► chunks + IDs
                                                                      │
              structured JSON (tool results) + chunks ──► explainer (LLM or template) ──► draft report
                                                                      │
                                       claim extractor ──► grounding gate ──► claim ledger ──► final QC card
```

### 4.3 Components
| ID | Component | Training-free? | Notes |
|---|---|---|---|
| M1 | Data layer | n/a | download, hash, schema discovery, group-aware splits |
| M2 | QC metric suite | yes | classical image statistics |
| M3 | Reference scoring + attribution | yes | robust statistics; fixed a-priori rule table |
| M4 | Embeddings + kNN (optional) | yes | frozen pretrained encoder, non-parametric |
| M5 | Corpus + hybrid retriever | yes | BM25 + sentence embeddings + RRF |
| M6 | Agent loop + tools | yes | ReAct, max 8 steps |
| M7 | Explainer | yes | local LLM (Ollama/llama.cpp) or deterministic template |
| M8 | Grounding gate + claim ledger | yes | regex/numeric match + lexical/embedding support checks |
| M9 | MCP server + SKILL.md | yes | FastMCP |
| M10 | App/CLI | yes | Streamlit + Typer |
| M11 | Evaluation harness | yes | E1–E9 |

---

## 5. Data sources — complete link inventory

### 5.1 PRIMARY image dataset (real OoC data)
**Organ-On-A-Chip (OOC) Image Dataset for Machine Learning and Tissue Model Evaluation** (Movčana et al., *Data* 9(2):28, 2024)
- DOI: https://doi.org/10.3390/data9020028 **[C]**
- Article page: https://www.mdpi.com/2306-5729/9/2/28/htm **[C]**
- PDF: https://www.mdpi.com/2306-5729/9/2/28/pdf?version=1706774288 **[C]**
- OpenAlex record: https://api.openalex.org/works/doi:10.3390%2FDATA9020028 (content PDF: https://content.openalex.org/works/W4391431032.pdf) **[C]**
- Content **[C]**: 3,072 brightfield microscopy images from an automated OoC setup; six cell lines; one metadata spreadsheet; each image evaluated by a cell-biology expert; for some images parameters such as cell type, seeding density, time after seeding, and flow rate are provided; intended for image-quality assessment and artifact identification.
- License **[C]**: the MDPI page states "Dataset License: CC-BY-SA"; the article itself is CC-BY. Cite the paper; share derived data under CC-BY-SA; do not redistribute the images in our repo unless the license allows it with attribution (prefer a download script).
- **Download location: [?]** not confirmed in research. Open the article's "Data Availability Statement" (typically near the end of the HTML page) to find the repository link (e.g., Zenodo/figshare/institutional). Record it in `data/SOURCES.md` with access date and file checksums.
- **Schema: [?]** do not assume column names. Task T03 prints the real schema. Decision tree in Section 5.5.

### 5.2 Fallback / supplementary image datasets (listed on the Kaggle page [K])
| Resource | Link | Use here |
|---|---|---|
| BBBC (Broad Bioimage Benchmark Collection) | https://bbbc.broadinstitute.org/ | Fallback if primary data is unavailable; sets with ground truth (e.g., BBBC039 nuclei masks, BBBC021 compound/MoA) are named from memory — **[?] verify on the site**; use for classical segmentation sanity checks and corruption-benchmark base images |
| IDR | https://idr.openmicroscopy.org/ | Optional additional real images/phenotype metadata |
| RxRx1 | https://www.rxrx.ai/rxrx1 | Optional; fluorescence + genetic perturbations (not brightfield) — only for context |
| CytoData | https://www.cytodata.org/ | Context/community resource; cite if used |
| JUMP Hub | https://broadinstitute.github.io/jump_hub/ | Context/docs for Cell Painting; cite if used |
| JUMP-Cell Painting | https://jump-cellpainting.broadinstitute.org/ | Optional; large; not needed for the 2-hour build |
| CellNet | https://cellnet.hms.harvard.edu/ | Transcriptomics/cell identity; not used (mention as out of scope) |

### 5.3 RAG / literature sources
| Source | Link | Status/Use |
|---|---|---|
| OoC review: design, sensing, applications (Biosensors 2024) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11430660/ | **[C]** seed paper; check license on the page before including |
| OoC review: microscopic imaging methods | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8879989/ | **[C]** seed paper (brightfield imaging with OoC); check license |
| The dataset paper itself | https://www.mdpi.com/2306-5729/9/2/28/htm | **[C]** CC-BY; include (explains criteria for quality/artifacts) |
| MedRAG corpora (HF) | https://huggingface.co/MedRAG ; PubMed: https://huggingface.co/datasets/MedRAG/pubmed (~23.9M snippets; repo ≈70 GB) ; Textbooks: https://huggingface.co/datasets/MedRAG/textbooks (125,847 snippets) | **[C]** too large/off-topic for the 2-hour build. Documented as an optional extension and as a negative design decision (broad corpus dilutes OoC relevance) |
| StatPearls | https://huggingface.co/datasets/MedRAG/statpearls | **[C]** HF repo is empty (redistribution not allowed); raw files at NCBI Bookshelf (https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz); not used |
| MedRAG toolkit | https://github.com/Teddy-XiongGZ/MedRAG | **[C]** reference only |
| Europe PMC REST API | https://www.ebi.ac.uk/europepmc/webservices/rest/ | **[?]** verify at build time; preferred way to discover and fetch open-access OoC full texts and licenses programmatically |
| PMC Open Access subset | via NCBI/Europe PMC | **[?]** licenses vary per article; filter to CC-BY/CC0 and record per-paper license |

**Corpus rule:** only include papers whose license permits reuse for text indexing and short excerpt display; store `license` and `url` per paper in `corpus/manifest.csv`; show only short excerpts (≤ 25 words) in the UI and always link the source. Do not commit full texts of papers with restrictive licenses; commit only the manifest + fetch script.

### 5.4 Project and tool links
| Item | Link | Source |
|---|---|---|
| Competition overview | https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview | [K] |
| Writeup URL pattern | `https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/<slug>` | [K] |
| User GitHub | https://github.com/simon-derock | [U] |
| User portfolio | https://philipsimonderock.com | [U] |
| xphil model (HF) | https://huggingface.co/philip11/xphil-gemma-4-E4B (GGUF Q3_K_M 4.85 GB, Q4_K_M 5.34 GB; Ollama: `ollama run hf.co/philip11/xphil-gemma-4-E4B:Q3_K_M`; Apache-2.0 adapter, base under Gemma terms; do NOT add `<|think|>`; replace its clinical system prompt) | [U]/[C] |
| User repos to harvest code patterns | `dex-jobs` (async ReAct engine, tool registry, multi-provider failover, memory), `Agentic-RAG-Chatbot` (multi-format ingestion, ChromaDB, Streamlit/HF Spaces), `Titan-resume` (claim provenance, deterministic gates, bounded repair loops), `xphil` (3-layer grounded pipeline, G1–G5 gates, calibrated confidence format), `Lunarbit` (evidence IDs idea only; no Neo4j) — all under https://github.com/simon-derock | [U] |
| Competitor reference | Public Kaggle notebook "SynapTwin-OoC" and a writeup/notebook titled "BioFluidNet-OoC" appear in the user's Kaggle "Viewed" list | [K]; used only for gap analysis, never copied |

Pretrained components to use by name (verify exact HF IDs/licenses at build time **[?]**): `rank_bm25` (BM25), `sentence-transformers/all-MiniLM-L6-v2` (dense text embeddings), a small frozen image encoder such as `facebook/dinov2-small` (optional), `scikit-image`, `scipy`, `opencv-python-headless`, `numpy`, `pandas`, `scikit-learn` (kNN/statistics only), `streamlit`, `typer`, `fastmcp`, `ollama` client (optional), `pytest`.

### 5.5 Dataset discovery decision tree (task T03, runs first)
```
Find download link in article "Data Availability" ──► download ──► checksum ──► list files
 ├─ Images + metadata spreadsheet present ──► print columns, dtypes, value counts, missing rates
 │    ├─ Has per-image quality/artifact/expert labels? ──► E2 (expert-agreement) enabled
 │    ├─ Has cell line / seeding density / time / flow rate? ──► E3 (metadata recovery) enabled
 │    └─ Has well/plate/experiment IDs? ──► use GROUP splits (prevents near-duplicate leakage)
 └─ Not downloadable in 10 minutes OR unreadable
      ──► FALLBACK: BBBC subset (verify download size first) as base images
          E1 (controlled corruptions) + E4 + E5 + E6 still run; E2/E3 reported as "not available"
          and the report states the dataset limitation plainly.
```
E1 (corruption benchmark), E4–E9 do not depend on expert labels, so the project remains valid under every branch.

---

## 6. Vision module (training-free) — M1–M4

### 6.1 Preprocessing
- Load grayscale float32 in [0,1]; record original dtype/bit depth/size; if RGB brightfield, convert with luminance weights and keep the original for display.
- Optional: downscale long side to ≤1024 px for metrics that are scale-sensitive; keep parameters in config.
- Compute `image_id`, SHA-256 of file bytes, and size; cache results in `cache/qc/<hash>.json`.

### 6.2 QC metric suite (`src/vision/metrics.py`)
All metrics return a float plus metadata (kernel size, units "relative" unless pixel size known). Document the formula in the docstring.

| Metric | Definition (implementation hint) | Detects |
|---|---|---|
| `focus_lapvar` | variance of Laplacian of normalized image | defocus/blur |
| `focus_tenengrad` | mean(gx²+gy²), Sobel | defocus/blur |
| `focus_high_freq_ratio` | energy above a fixed frequency fraction / total (FFT) | blur, over-smoothing |
| `contrast_rms` | std of intensities | low contrast |
| `contrast_p99_p1` | p99 − p1 of intensities | low dynamic range |
| `clip_low_frac`, `clip_high_frac` | fraction of pixels at min/max after normalization of the raw bit depth | under/over-exposure |
| `illum_flatness` | fit low-order 2D polynomial (degree 2) to a heavily blurred image; report (max−min of fitted background)/mean | uneven illumination/vignetting |
| `noise_sigma` | Immerkær fast noise estimator / robust MAD of Laplacian response | sensor/shot noise |
| `debris_count`, `debris_area_frac` | black-hat/top-hat + threshold + connected components filtered by area and solidity | debris, dust, particulates |
| `bubble_score` | large near-circular objects with strong edge ring (region props: area, circularity/eccentricity, edge contrast) | bubbles |
| `foreground_frac` (confluence proxy) | local standard-deviation map → Otsu threshold → foreground fraction | too sparse / too confluent |
| `blob_density_per_mpx` | Laplacian-of-Gaussian blob count per megapixel (units "per Mpx" unless pixel size is documented **[?]**) | density |
| `texture_lbp_entropy`, `glcm_contrast`, `glcm_homogeneity` | LBP histogram entropy; GLCM stats | texture shifts |
| `jpeg_blockiness` (optional) | 8×8 block boundary discontinuity | compression artifacts |

### 6.3 Reference statistics and scoring (`src/vision/scoring.py`)
- **Reference set:** a fixed subset of images designated "reference" in the dev split (if expert labels exist, use images labeled acceptable; otherwise use all dev images). Compute per-metric **median and MAD** (optionally per cell line if metadata exists and group sizes ≥ 30). These are descriptive statistics, not learned weights.
- **Robust z-score:** `z = 0.6745 * (x − median) / MAD` (guard MAD=0).
- **QC score:** `max_i |z_i|` over metrics relevant to quality (focus, contrast, clipping, illumination, noise, debris/bubbles), plus an aggregated RMS for information.
- **Pre-registered thresholds (a priori, in config):** REVIEW if score > 3.0, FAIL if > 5.0. Run a **threshold sensitivity sweep** (2.0–6.0) and report curves; if thresholds are tuned on dev, say so and report test-only results.
- **Density state:** separate from quality: `SPARSE | NORMAL | CONFLUENT` from `foreground_frac` and `blob_density_per_mpx` z-scores, per cell line where available.

### 6.4 Artifact attribution rule table (`src/vision/attribution.py`)
A fixed, human-readable table in `configs/artifact_rules.yaml`, reviewed by a biologist if possible:
| Condition (on z-scores/values) | Label | Plain-language note |
|---|---|---|
| focus_* strongly below reference, high_freq_ratio low | `DEFOCUS_OR_BLUR` | image appears out of focus |
| noise_sigma strongly above reference | `NOISE` | elevated sensor noise |
| illum_flatness strongly above reference | `UNEVEN_ILLUMINATION` | background gradient/vignetting |
| clip_low_frac or clip_high_frac above limit | `UNDER_OR_OVER_EXPOSED` | clipped pixels |
| contrast_* strongly below reference | `LOW_CONTRAST` | little dynamic range |
| debris_count/area high | `DEBRIS` | particulates/dust |
| bubble_score high | `BUBBLE` | possible bubble |
| foreground_frac extreme | `ABNORMAL_DENSITY` | sparse or over-confluent culture |
Multiple labels may apply. Each label carries the metric values that triggered it (needed by the ledger).

### 6.5 Frozen embeddings + kNN (optional, parallelizable) — `src/vision/embed.py`
- Extract features with a frozen small encoder (or a handcrafted fallback: concatenated LBP/GLCM/intensity statistics) and cache to `cache/embeddings/*.npy`. No fine-tuning.
- `find_similar(image_id, k)` via cosine kNN. Used for (a) "similar reference images" in the UI and (b) metadata recovery experiment E3.
- CPU time guard: if embedding all images would exceed ~10 minutes, embed a stratified subset and say so in the report.

---

## 7. RAG module — M5

### 7.1 Corpus build (`scripts/build_corpus.py`)
1. Seed with PMC11430660, PMC8879989, and the dataset paper (Section 5.3).
2. Discover additional open-access papers via Europe PMC **[?]** with queries such as: `"organ-on-a-chip" brightfield imaging`, `"organ-on-a-chip" image quality`, `microfluidic cell culture monitoring machine learning`, `"organ-on-a-chip" neurotoxicity`, `"organ-on-a-chip" image analysis segmentation`, `cell seeding density flow rate microfluidic`, `brightfield microscopy artifact bubble debris`, `automated imaging microphysiological system`.
3. Keep only CC-BY/CC0 (or equivalent) papers; target **60–150 papers** (minimum 30). Save `corpus/manifest.csv` with columns: `pmcid, title, year, license, url, retrieved_at, sha256`.
4. Parse into sections/paragraphs; chunk to ~150–250 words with 30-word overlap; chunk ID format `PMCID::section::index`; store `corpus/chunks.jsonl`.
5. Do not commit full texts for restrictive licenses; commit manifest + script. Cache full texts under `cache/` (git-ignored).

### 7.2 Retrieval (`src/rag/retriever.py`)
- BM25 (`rank_bm25`) over chunks.
- Dense embeddings with `all-MiniLM-L6-v2` (CPU), cached.
- **Hybrid with Reciprocal Rank Fusion** (k=60), return top-k (default 5) with scores.
- Query builder converts QC flags into queries (e.g., `DEFOCUS_OR_BLUR` → "out-of-focus brightfield microscopy image quality organ-on-a-chip"). Keep the mapping in `configs/query_templates.yaml`.
- Always return chunk text, chunk ID, paper title, URL, license.

### 7.3 Retrieval gold set (for E4)
- `eval/retrieval_gold.jsonl` with **≥ 40 questions**, each mapped to ≥1 relevant chunk/paper, written from the actual corpus by a human/agent and spot-checked. Include **≥ 10 out-of-corpus questions** whose correct behavior is abstention.

---

## 8. Agent, grounding gate, MCP, skills — M6–M9

### 8.1 Tool registry (shared by agent and MCP)
| Tool | Input | Output (JSON) |
|---|---|---|
| `load_image` | path or dataset image_id | image stats, id, hash |
| `compute_qc` | image_id | all metrics + z-scores + QC score + level + artifact labels |
| `compare_to_reference` | image_id, cell_line (optional) | per-metric percentile vs reference, density state |
| `find_similar_images` | image_id, k | list of (image_id, similarity, metadata) |
| `retrieve_evidence` | query, k | chunks with IDs, titles, URLs, licenses, scores |
| `calculate` | expression | numeric result (safe evaluator, no `eval`) — LLM must use this for any derived number |
| `check_claims` | draft text, tool_trace | claim ledger JSON |
| `write_qc_card` | tool_trace, ledger | final markdown/JSON card |
All tool outputs carry a unique `tool_call_id`.

### 8.2 Agent loop (`src/agent/loop.py`)
- ReAct style, **max 8 steps**, JSON tool-call protocol, strict schema validation, retry once on malformed output, then fall back to template mode.
- Provider abstraction `src/agent/llm.py`: `ollama`, `openai_compatible` (llama.cpp/vLLM server on localhost), and `template` (no LLM). Default = `template` so reviewers are never blocked.
- Logs every step to `runs/<timestamp>/trace.jsonl` (tool call, args, result hash, latency).
- Harvest patterns from the user's `dex-jobs` ReAct engine (zero-abstraction async loop, failover) but keep dependencies minimal.

### 8.3 Explainer contract
System prompt (core rules): "You are a research assistant for OoC image QC. Use ONLY values present in TOOL_RESULTS and passages present in EVIDENCE. Cite passages as [chunk_id]. Never state a number that is not in TOOL_RESULTS or computed with `calculate`. If evidence is insufficient, reply `ABSTAIN: <reason>`. Never give medical, diagnostic, or clinical advice. This is a research tool." Replace any clinical system prompt (e.g., xphil's) with this one.

Template mode builds the explanation deterministically from the JSON (this is the 100%-grounded-by-construction baseline in E5).

### 8.4 Grounding gate and claim ledger (`src/grounding/`)
1. **Claim extraction:** split the draft into sentences; classify each as `numeric`, `literature`, or `inference`.
2. **Numeric check:** extract numbers/units; match against flattened tool outputs and `calculate` results within relative tolerance (default 1e-3, plus rounding tolerance); unmatched numbers → `unsupported`.
3. **Literature check:** each sentence with a citation must cite an existing retrieved chunk ID, and the sentence's key terms must be supported by the chunk (lexical overlap ≥ θ_lex **and** embedding cosine ≥ θ_emb; thresholds set a priori in config, sensitivity reported). Citation without support → `unsupported`.
4. **Inference check:** inferences must reference at least one supported claim ID; otherwise `unsupported`.
5. **Repair loop (bounded, 2 rounds):** unsupported claims are fed back for rewrite or removal; remaining ones are dropped from the final card and listed in the ledger as blocked.
6. **Abstention:** if the evidence set is empty or below a score threshold, output `ABSTAIN` with reason.
Ledger schema (`results/ledger_schema.json`): `{claim_id, text, type, evidence:[tool_call_id|chunk_id], status: supported|unsupported|abstain, reason}`.

**Confidence wording:** QC "confidence" is expressed as a reference-percentile band and score margin from the threshold — **not** a calibrated probability. Do not claim calibration unless ECE is actually computed in E5/E2.

### 8.5 MCP server (`src/mcp_server.py`)
- Built with FastMCP; exposes the tools in 8.1 (except internal-only helpers).
- `mcp_manifest.json` lists tool names, schemas, example calls.
- Test: start the server, list tools, call `compute_qc` on a fixture image, assert schema validity.
- Harvest the user's legal-research FastMCP experience for structure **[U]**.

### 8.6 Skills and agent docs
- `skills/ooc-qc/SKILL.md` with YAML frontmatter (`name`, `description`) and instructions: when to use the skill, the required tool order (`load_image → compute_qc → compare_to_reference → retrieve_evidence → check_claims → write_qc_card`), how to handle ABSTAIN, forbidden behaviors (no clinical claims, no uncited literature claims), output format.
- `AGENTS.md` (root): the rules from Section 0, adapted for any coding/automation agent working in the repo.
- `JUDGES.md`: the rubric-to-evidence map (Section 3.4).

---

## 9. Evaluation protocol — everything runs from `evaluate.py`

General rules: fixed seeds; dev/test split (default 30/70) **group-aware** if well/plate/experiment IDs exist; bootstrap 95% CIs (1,000 resamples); every table reports n; baselines are always shown; failures are shown, not hidden.

| ID | Experiment | Ground truth | Metrics | Baselines |
|---|---|---|---|---|
| **E1** | **Controlled-corruption benchmark** on ≥200 clean-ish images stratified by cell line (or BBBC fallback). Apply corruptions at 5 severities each: Gaussian blur, defocus (disk), motion blur, additive Gaussian noise, illumination gradient/vignette, contrast reduction, under/over-exposure with clipping, synthetic debris/bubbles, JPEG compression | Known by construction | Spearman ρ between severity and each metric (monotonicity); AUROC clean vs severity ≥3; flag precision/recall at pre-registered thresholds; **artifact attribution accuracy** (does the predicted label match the injected corruption); confusion matrix | Random; single-metric focus-only detector |
| **E2** | **Expert-label agreement** (only if quality/artifact labels exist **[?]**) | Expert evaluation in the dataset | AUROC, F1, Cohen's κ, precision/recall, per-class; threshold sensitivity curve | Random, majority class, focus-only |
| **E3** | **Metadata recovery by kNN on frozen/handcrafted features** (cell line; others if available **[?]**) | Dataset metadata | accuracy, macro-F1, top-k; **group-aware split** | Majority class, random, handcrafted-only vs embedding |
| **E4** | **Retrieval quality** on the gold set | Hand-labeled chunks/papers | hit@1/3/5, MRR, nDCG@5; abstention accuracy on out-of-corpus questions | BM25-only, dense-only vs hybrid |
| **E5** | **Grounding & abstention** on ≥60 (image, question) cases, including ≥10 unanswerable and ≥10 adversarial prompts (asks for diagnosis, invented numbers, fake citations) | Computed tool outputs + gold evidence | numeric grounding rate, citation validity rate, unsupported-claim rate, abstention precision/recall, refusal correctness on adversarial prompts | Template-only (grounded by construction); LLM without gate; LLM with gate; compare LLM variants (e.g., xphil GGUF vs a general small instruct model) |
| **E6** | **Reproducibility test**: fresh clone in clean env/Docker; `make judge` and `make eval` | n/a | pass/fail, wall-clock, identical results hash across 2 runs | n/a |
| **E7** | **Ablations**: remove artifact attribution; remove retrieval; remove gate; metrics subsets; thresholds sweep | as above | metric deltas | full system |
| **E8** | **Failure analysis**: 5–10 worst-case images/queries shown with explanations | n/a | qualitative table in report | n/a |
| **E9** | **Efficiency**: per-image latency (CPU), embedding cache size, memory | n/a | median/p95 latency | n/a |

Outputs: `results/results.json` (all numbers + provenance: git SHA, timestamp, data hash, config hash, library versions), `results/results.md` (tables pasted into README/report), `results/figures/*.png` (hero figure, corruption monotonicity plots, confusion matrices, PR/ROC, retrieval bars, grounding ablation bars, threshold sweeps).

Honesty rules: report any experiment that could not run and why; never fill a table cell with a placeholder; state when a result is not statistically distinguishable from a baseline.

---

## 10. Repository structure
```
ooc-copilot/
├── README.md                 # pitch, hero fig, Judge Quickstart, results table, limits, licenses
├── JUDGES.md                 # rubric → evidence map
├── AGENTS.md                 # rules for coding/automation agents
├── PLAN_SPEC.md              # this file
├── LICENSE                   # code license (MIT/Apache-2.0); data/corpus licenses documented separately
├── Makefile                  # setup data corpus qc eval judge app mcp test report all
├── Dockerfile
├── requirements.txt          # pinned
├── environment.yml           # optional
├── pyproject.toml
├── configs/
│   ├── default.yaml          # seeds, thresholds, k, paths, split ratios
│   ├── artifact_rules.yaml
│   └── query_templates.yaml
├── data/
│   ├── README.md             # how to obtain data, license, checksums
│   ├── SOURCES.md            # links + access dates
│   └── download.py
├── corpus/
│   ├── manifest.csv          # pmcid, title, license, url, retrieved_at, sha256
│   └── chunks.jsonl          # only if licenses allow; else built by script
├── src/
│   ├── common/               # io, hashing, seeding, config, logging
│   ├── vision/               # metrics.py, scoring.py, attribution.py, embed.py, corrupt.py
│   ├── rag/                  # corpus.py, retriever.py, query_builder.py
│   ├── agent/                # loop.py, llm.py, tools.py, prompts.py, template.py
│   ├── grounding/            # extract.py, numeric.py, literature.py, ledger.py, gate.py
│   ├── mcp_server.py
│   └── app/                  # streamlit_app.py, cli.py
├── skills/ooc-qc/SKILL.md
├── eval/
│   ├── retrieval_gold.jsonl
│   ├── grounding_cases.jsonl
│   └── adversarial_prompts.jsonl
├── scripts/                  # build_corpus.py, make_figures.py, record_demo_assets.py
├── tests/                    # pytest suite (Section 13)
├── results/                  # generated only; never hand-edited
├── report/                   # report.md, report.pdf, figures/
├── docs/                     # SOURCES.md, LIMITATIONS.md, DEFENSE_QA.md, VIDEO_SCRIPT.md
└── .github/workflows/ci.yml  # lint + tests + smoke eval
```

---

## 11. Key interfaces (signatures the agent should implement)
```python
# src/vision/metrics.py
def compute_metrics(img: np.ndarray) -> dict[str, float]: ...
# src/vision/scoring.py
def fit_reference(metrics_df: pd.DataFrame, by: str | None) -> dict: ...        # medians + MAD only
def score_image(metrics: dict, ref: dict, cell_line: str | None) -> dict: ...   # z, score, level
# src/vision/attribution.py
def attribute(metrics: dict, z: dict, rules: dict) -> list[dict]: ...           # labels + triggering metrics
# src/vision/corrupt.py
def corrupt(img: np.ndarray, kind: str, severity: int, rng) -> np.ndarray: ...
# src/vision/embed.py
def embed_images(paths: list[str], backend: str) -> np.ndarray: ...
def knn(index: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...
# src/rag/retriever.py
class HybridRetriever:
    def __init__(self, chunks_path: str, cfg: dict): ...
    def search(self, query: str, k: int = 5) -> list[dict]: ...                 # chunk_id,text,title,url,license,score
# src/agent/loop.py
def run_agent(task: dict, provider: str, max_steps: int = 8) -> dict: ...       # returns card + ledger + trace
# src/grounding/gate.py
def extract_claims(text: str) -> list[dict]: ...
def verify(claims: list[dict], tool_trace: list[dict], evidence: list[dict], cfg: dict) -> dict: ...  # ledger
def repair(draft: str, ledger: dict, llm, rounds: int = 2) -> tuple[str, dict]: ...
```

---

## 12. UI, CLI, packaging

### 12.1 Streamlit app (`src/app/streamlit_app.py`)
- Upload image or pick a dataset example.
- **QC card:** PASS/REVIEW/FAIL badge, score, density state, artifact labels with the metric values that triggered them, per-metric percentile bars.
- **Similar reference images** panel (kNN).
- **Evidence panel:** retrieved passages (short excerpts ≤ 25 words, with source link + license).
- **Claim ledger view:** each claim green (supported) / red (blocked) / grey (abstain), with evidence IDs. Include a toggle "Show what the ungated LLM said" to demonstrate the gate (used in the video).
- Disclaimer banner: research use only; not diagnostic.
- Works in template mode with no LLM; LLM provider selectable in sidebar.

### 12.2 CLI (`src/app/cli.py`, Typer)
`ooc qc <image>` · `ooc explain <image> --provider template|ollama` · `ooc eval --subset 100` · `ooc corpus build` · `ooc mcp serve`.

### 12.3 Packaging
- `Dockerfile` (python:3.11-slim, pinned requirements, entrypoint `make judge`).
- `make judge` ≈ 100-image smoke eval + retrieval/gate tests, PASS/FAIL summary, target < 5 min CPU.
- `make all` = setup → data → corpus → eval → report assets.
- CI: lint (ruff), tests, smoke eval on fixtures; badge in README.
- Optional public demo: Hugging Face Space or Streamlit Community Cloud running template mode (must be reachable without login; provide backup screenshots/video). Mark as optional in the Writeup.

---

## 13. Work breakdown with acceptance tests

| ID | Task | Acceptance test |
|---|---|---|
| T00 | Registration form submitted; confirm time zone and deadline countdown; verify the Writeups tab to see whether a draft/submission already exists under your account | Screenshot of form confirmation saved to private notes |
| T01 | Repo skeleton, tooling, Makefile, config, license, `AGENTS.md`; **freeze interfaces** (Section 11) so parallel streams don't collide | `make test` runs (even if empty); tree matches Section 10 |
| T02 | Locate dataset download link in the article's Data Availability section; download; checksum; write `data/SOURCES.md` | Files present; SHA-256 recorded |
| T03 | Schema discovery script `scripts/discover_data.py` | Prints columns/dtypes/value counts/missing; decision-tree branch chosen and logged to `results/data_report.json` |
| T04 | Group-aware dev/test split utility | Unit test: no group appears in both splits |
| T05 | QC metric suite | Unit tests: blurred image → lower focus; noisier → higher noise_sigma; vignetted → higher illum_flatness |
| T06 | Reference stats + scoring + levels | Unit test: reference images score mostly PASS; heavy corruption → FAIL |
| T07 | Artifact attribution rules | Unit test: each injected corruption type is attributed to the right label on fixtures |
| T08 | Corruption generator | Determinism test with fixed seed; severity ordering monotone in a proxy metric |
| T09 | E1 corruption benchmark | `results/e1.json` exists with CIs, confusion matrix, figures |
| T10 | E2 expert agreement (if labels exist) | `results/e2.json` or documented "not available" |
| T11 | Embeddings (+ fallback features) + kNN + E3 | `results/e3.json` with baselines and group-aware split |
| T12 | Corpus build (manifest + chunks) | ≥30 papers (target 60–150), licenses recorded, chunk IDs unique |
| T13 | Hybrid retriever + query templates | Unit test: known question returns its gold chunk in top-5 |
| T14 | Retrieval gold set + E4 | `results/e4.json`; includes abstention accuracy |
| T15 | Tool registry + `calculate` safe evaluator | Unit tests incl. rejecting code injection in `calculate` |
| T16 | Template-mode explainer | Output is 100% grounded by construction; ledger all supported |
| T17 | LLM provider abstraction (Ollama / OpenAI-compatible / template) + prompt contract | Works with at least one local model; falls back to template on failure |
| T18 | Claim extractor + numeric/literature/inference checks + ledger | Unit tests: fabricated number blocked; fake citation blocked; correct claim passes |
| T19 | Repair loop + abstention | Test: unsupported claim is removed or rewritten within 2 rounds; empty evidence → ABSTAIN |
| T20 | Agent loop (max 8 steps) + traces | Trace file written; schema-valid |
| T21 | E5 grounding/abstention/adversarial cases | `results/e5.json` with gate on/off and LLM variants |
| T22 | MCP server + manifest | Test lists tools and calls `compute_qc` |
| T23 | `SKILL.md`, `JUDGES.md`, README skeleton | Files exist; README Quickstart verified from a clean clone |
| T24 | Streamlit app + CLI | Manual smoke test; screenshots saved for report |
| T25 | `evaluate.py` orchestrating E1–E9 + figures + `results.md` | One command regenerates everything; provenance fields present |
| T26 | Dockerfile + CI + `make judge` | `make judge` PASS in clean container |
| T27 | Provenance test | Test fails if README/report numbers are not found in `results/results.json` |
| T28 | Technical report (md → pdf) | Contains all required sections (15.1); renders; links valid |
| T29 | Demo video recorded, uploaded to YouTube (public), captions added | Opens in incognito; ≤ 5:00 |
| T30 | Writeup filled, checklist 7/7, **Submit** | Status shows **Submitted**; all links tested in incognito |
| T31 | Tag `v1.0-submission`; verify fresh clone; freeze `main` | Tag exists; Writeup links point to the tag/commit |

---

## 14. Timeline (T-minus, assuming ~2h; T-0 = 9:29 PM IST deadline)

Parallel streams (assign to separate agents if available; otherwise run top-to-bottom):
- **Stream A (vision/eval-E1–E3):** T02–T11
- **Stream B (RAG):** T12–T14
- **Stream C (agent/grounding/MCP):** T15–T22
- **Stream D (packaging/docs/video/writeup):** T00–T01, T23–T31

| When | Checkpoint | Must be true |
|---|---|---|
| T-120 → T-110 | **C0** | Registration done (T00); repo created; interfaces frozen (T01); data link found (T02) |
| T-110 → T-85 | **C1** | Data loaded; schema known (T03); metrics + scoring working (T05–T06); corruption generator (T08) |
| T-85 → T-60 | **C2** | E1 results (T09); corpus + retriever working (T12–T13); template explainer (T16); gate unit tests (T18) |
| T-60 → T-45 | **C3** | Agent loop + MCP (T20, T22); app runs (T24); `evaluate.py` runs end-to-end (T25) |
| T-45 | **C4 — first valid submission** | README + results table + short report + video draft; **submit Writeup v1** (resubmission is allowed **[K]**) |
| T-45 → T-30 | | E3/E4/E5 finalized; LLM mode tested; figures; report completed; **video recorded and uploaded to YouTube** (allow processing time) |
| T-30 → T-15 | **C5** | Docker/CI/`make judge` verified from a clean clone (T26); provenance test (T27); tag `v0.1-valid` |
| T-15 → T-5 | **C6 — final polish** | Pre-mortem (3.7); update Writeup with final numbers; test all links in incognito |
| T-5 → T-0 | **C7 — final submit** | Status **Submitted**; screenshot of confirmation; tag `v1.0-submission` |

Rule: if a checkpoint is late, **do not skip downstream packaging** (video, Writeup, README). Finish later experiments and re-submit; the Writeup can be updated until the deadline. If something slips, the order of sacrifice is: E7 breadth → E9 → extra corpus papers → optional embeddings; never README, results provenance, video, or Writeup.

---

## 15. Submission package

### 15.1 Technical report outline (`report/report.md` → PDF)
Concise and self-contained; aim ~10–15 pages main text (limit stated: ideally 15–20 excluding references/appendices **[K]**).
1. **Title, team, summary** (declare category: End-to-End System; declare team composition and expertise honestly)
2. **Problem & application scenario** (lab workflow, why QC matters; every external claim sourced)
3. **Related work and gap** (OoC imaging reviews; the OoC image dataset; why training-free; why evidence-gating)
4. **Data**: sources, licenses (CC-BY-SA for the dataset), splits, preprocessing, compliance statement (no personal/clinical data), corpus manifest and licenses
5. **Method**: QC metric suite, reference scoring, attribution rules, embeddings/kNN, retrieval, agent, grounding gate, claim ledger, MCP/skill packaging; architecture diagram
6. **Implementation details**: libraries, versions, config, hardware, runtimes
7. **Experiments & results**: E1–E9 with tables/figures, baselines, CIs, ablations
8. **Reliability analysis & limitations**: single-lab data, small corpus, heuristic grounding checks, thresholds disclosure, not calibrated probabilities, no clinical use, failure cases
9. **Impact**: who benefits, how to deploy, how to extend (e.g., other chips, other imaging modalities, longitudinal monitoring, integration with lab systems)
10. **Reproduction instructions**: exact commands, expected outputs, runtimes
11. **Sources and licenses** of all external models, libraries, datasets, AI tools used (including any AI coding assistants and LLMs)
12. **Acknowledgements and ethics**
Appendices: configuration, full tables, claim ledger examples, MCP schemas.

### 15.2 Demo video (≤ 4:45; YouTube public; captions)
| Time | Content |
|---|---|
| 0:00–0:20 | Hook: a lab technician faces hundreds of OoC images daily; bad images silently corrupt results. One-sentence promise. |
| 0:20–0:50 | Approach in one diagram: training-free QC + cited explanation + trust gate. |
| 0:50–2:20 | **Live demo:** (1) a good real image → PASS card; (2) a problematic real or corrupted image → REVIEW/FAIL with artifact labels and metrics; (3) similar images; (4) evidence panel with citations. |
| 2:20–2:50 | **The trust moment:** show the ungated LLM inventing a number/citation, then the gate blocking it and the ledger turning it red. |
| 2:50–3:40 | Results from `results.md`: corruption benchmark, expert agreement or metadata recovery, retrieval, grounding ablation; call out baselines and CIs. |
| 3:40–4:15 | Reproducibility: fresh clone → `make judge` PASS; MCP tool call from an external agent client. |
| 4:15–4:45 | Limits (honest), impact, next steps. |
Recording: 1080p, large fonts, no copyrighted music, only own or licensed footage, show inputs/outputs/real runs, burn in captions, upload by T-30 to allow YouTube processing.

### 15.3 Kaggle Writeup form (from the Writeup page)
| Field | Limit/Requirement | Content |
|---|---|---|
| Title | ≤ 80 chars | `OoC Copilot: Grounded Image QC and Evidence Agent for Organ-on-a-Chip` |
| Subtitle | ≤ 140 chars | `Flags organ-on-a-chip image problems and explains each flag with cited literature, blocking claims the data cannot support.` |
| Card/thumbnail | 560 × 280 | hero figure (QC card + ledger snippet) |
| Submission track | auto | Artificial Intelligence for Life Science |
| Media gallery | YouTube video required | demo video + 2–3 screenshots |
| Project description | template with sections | Category Declaration, Demo Video, Code Repository Link, Project Summary (200–300 words), Technical Report (inline or public PDF link), Optional Demo Link; keep the registration note at the top |
| Project links | add links | GitHub repo (tagged release), report PDF, optional demo |
| Files | ≤ 100 MB | optional: `results.md`, report PDF |
| DOI citation | optional opt-in | team decision |
| Checklist | 7 items | all complete, then **Submit** (not Save Draft) |

Description body skeleton (fill every [..] from `results.json`):
```
## Category Declaration
End-to-End System

## Demo Video
[YouTube URL]

## Code Repository Link
[GitHub URL @ tag v1.0-submission]

## Project Summary
OoC Copilot helps organ-on-a-chip labs monitor brightfield cultures. It flags image
quality problems and abnormal culture states with a training-free, physics-aware QC
suite, attributes them to likely artifacts, retrieves cited open-access OoC literature,
and verifies every claim through a grounding gate and an auditable claim ledger.
On [dataset, n images] ... [E1 headline] ... [E2/E3 headline] ... [E4 headline] ...
[E5 headline: grounding rate X→Y with gate; abstention on out-of-corpus Z%].
Everything runs on CPU with free tools; one command regenerates all numbers.
Limitations: [single-lab data, small corpus, heuristic checks, research use only].
(200–300 words total)

## Technical Report
[PDF link + short inline abstract]

## Optional Demo Link
[URL or "see local run instructions"]
```

### 15.4 Final submission checklist
- [ ] Registration form done (and teammates, if any)
- [ ] Only one Writeup for the team; verify the Writeups tab for stray drafts/duplicates
- [ ] Category declared at the top of the description
- [ ] Video plays in incognito, ≤ 5:00, YouTube-hosted
- [ ] Repo public; tag `v1.0-submission`; README Quickstart verified from a clean clone
- [ ] Report PDF link opens without login
- [ ] All numbers in Writeup match `results/results.json`
- [ ] Data/corpus/model licenses stated; CC-BY-SA attribution for the dataset included
- [ ] Limitations section present; no clinical/diagnostic claims
- [ ] Status = **Submitted** before 9:29 PM IST; confirmation screenshot saved

---

## 16. Risk register and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Dataset download location not found or slow | Medium | 10-minute cap (T02); fall back to BBBC subset; E1/E4–E9 still valid; state the limitation |
| Metadata schema lacks quality labels | Medium | E2 reported "not available"; E1 provides ground-truth validation; E3 if metadata exists |
| Near-duplicate leakage inflates E3 | Medium | group-aware splits; report split method; add caveat |
| Embedding step too slow on CPU | Medium | stratified subset or handcrafted-feature fallback; cache |
| Corpus too small/off-topic | Medium | minimum 30 papers; tighten queries; include dataset paper; report retrieval metrics honestly |
| Licenses restrict redistribution | Medium | commit only manifest + fetch script; short excerpts only; cite sources |
| LLM unavailable/slow (Ollama not installed on reviewer machine) | High | default template mode; LLM mode optional; document |
| Grounding gate false positives/negatives | Medium | report both; thresholds sensitivity; show examples in failure analysis |
| YouTube processing delay | Medium | upload by T-30; keep unlisted→public flip ready; backup MP4 in repo release |
| Kaggle form issues (checklist stuck, image dimensions) | Low–Med | prepare 560×280 image early; test Save Draft early; submit v1 at T-45 |
| Registration forgotten | Low but fatal | T00 first; screenshot confirmation |
| Duplicate/ambiguous existing Writeup under the account | Low | check Writeups tab before creating; only one valid Writeup per team **[K]** |
| Overclaiming | Medium | provenance test; `docs/LIMITATIONS.md`; pre-mortem |
| Reusing prior work (xphil, dex-jobs, Titan-resume patterns) | Certain | disclose as methodology/code reuse in the report; present new OoC-specific contribution clearly; do not resubmit a prior hackathon project; observe licenses (xphil adapter Apache-2.0; base model Gemma terms) |
| Time overrun | High | follow Section 14 order; C4 submit v1 at T-45 |

---

## 17. Honesty, ethics, and compliance rules
- Research use only; no diagnosis, treatment, or clinical advice; no patient data.
- Use only data/corpora with compatible licenses; cite all sources; honor CC-BY-SA share-alike for derivatives of the image dataset.
- Third-party content in the video (images, music, model outputs) must have usage/redistribution rights **[K]**.
- Disclose AI assistance used to write code or text (coding agents, LLMs) in the report's sources section.
- State thresholds that were tuned on dev data; report test results only for headline claims.
- Report negative and null results; show failure cases.
- Do not claim biology expertise, clinical validation, or generalization beyond the tested data.
- External numbers (e.g., costs of staining) may appear only with a cited source; otherwise omit.

---

## 18. If we reach the top 20 (Oct 20–30 defense **[K]**)
- Freeze `main`; continue on a branch; never overwrite the tagged submission; keep the Writeup links stable.
- Prepare a 10–15 minute presentation: problem → approach → live demo → validation → limitations → roadmap. Rehearse a 3-minute version as well.
- `docs/DEFENSE_QA.md` should answer likely expert questions:
  - Why training-free? How do thresholds generalize to other chips or microscopes?
  - What does the QC score mean biologically? Which artifacts matter most for downstream assays?
  - How were expert labels used (if any)? What is the leakage risk?
  - How reliable is the grounding gate? What are its failure modes?
  - How would this integrate with a lab's imaging pipeline? Data privacy? Regulatory scope (research use only)?
  - What would you do with real longitudinal OoC time-series or fluorescence data? Digital-twin roadmap relevant to the sponsor.
- Roadmap candidates (clearly labeled future work): longitudinal monitoring, multi-modal (bright-field + sensors), active-learning with lab feedback (would introduce training — keep separate), virtual-staining comparison as a baseline study, integration with experiment-automation tools.

---

## Appendix A — Default config (`configs/default.yaml`)
```yaml
seed: 1337
split: {dev: 0.30, test: 0.70, group_by: null}   # set group_by after T03
qc:
  review_z: 3.0
  fail_z: 5.0
  sweep_z: [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
corruptions: {severities: [1,2,3,4,5], n_images: 200}
retrieval: {k: 5, rrf_k: 60, chunk_words: 200, overlap_words: 30}
grounding: {num_rel_tol: 0.001, theta_lex: 0.35, theta_emb: 0.55, repair_rounds: 2}
agent: {max_steps: 8, provider: template}
bootstrap: {n: 1000, alpha: 0.05}
```
(All thresholds are a-priori defaults; any tuning must be on the dev split and reported.)

## Appendix B — README skeleton
1. Title + one-paragraph pitch + hero figure
2. **Judge Quickstart** (3 commands: `make setup`, `make data corpus`, `make judge`)
3. What it does (QC card screenshot, ledger screenshot)
4. Results (pasted from `results/results.md`)
5. How it works (architecture diagram, training-free statement)
6. Reproduce everything (`make all`), runtimes, hardware
7. LLM modes (template default; Ollama/llama.cpp optional), MCP usage, SKILL.md
8. Data & corpus licenses; attribution (CC-BY-SA dataset)
9. Limitations; research-use-only statement
10. Citation, team, acknowledgements, third-party sources

## Appendix C — Makefile targets
`setup` (venv + pinned deps) · `data` (download + checksum + schema report) · `corpus` (build manifest/chunks) · `qc` (compute metrics cache) · `eval` (E1–E9) · `judge` (smoke subset + PASS/FAIL) · `app` · `mcp` · `test` · `report` (md→pdf) · `all`.

## Appendix D — `JUDGES.md` skeleton
| Criterion | Evidence | Command / file |
|---|---|---|
| Importance (30%) | problem + use case | `report/report.md` §2 |
| Innovation (30%) | training-free QC + artifact attribution; claim ledger gate; MCP + SKILL.md | `src/vision/`, `src/grounding/`, `src/mcp_server.py`, `skills/ooc-qc/SKILL.md` |
| Validation (20%) | E1–E9 with baselines/CIs | `results/results.md`, `make eval` |
| Reproducibility (10%) | Docker, CI, `make judge`, provenance | `Dockerfile`, `.github/workflows/ci.yml`, `make judge` |
| Presentation (10%) | video + hero figure | Writeup media gallery |

## Appendix E — Known unknowns to resolve in the first 15 minutes
1. Where the OoC image dataset is hosted and its size/format. **[?]**
2. Columns in the metadata spreadsheet; whether quality/artifact labels and group IDs exist. **[?]**
3. Whether pixel size/magnification is documented (affects per-area densities). **[?]**
4. Which OoC papers are available under compatible licenses via Europe PMC/PMC. **[?]**
5. Whether Ollama and a small local model are available on the demo machine. **[?]**
6. Whether any teammate with biology/bioengineering expertise can join (bonus +0.5). **[?]**
7. Exact remaining time on the Kaggle countdown and time zone (deadline 9:29 PM GMT+5:30 **[K]**).
