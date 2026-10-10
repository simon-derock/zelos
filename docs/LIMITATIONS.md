# Limitations (measured, not hypothetical)

| Limitation | Evidence | Effect |
|---|---|---|
| Cross-session acquisition shift | E1: the pre-registered dev-session reference flags most clean test images (results/e1.json `clean_flag_rate`) | a single global reference does not transfer between imaging sessions; the batch-relative reference (added after that result, disclosed) reduces but does not remove false flags |
| Cause attribution is imperfect | E1 per-kind attribution (results/results.md): blur kinds are often attributed to illumination/contrast, bubbles to debris | the primary label is a hint; the top-2 list and the metric values are shown on every certificate |
| Expert label ≠ image artifacts | E2: artifact QC score is below random against the expert good/bad decision; kNN label transfer is the useful signal | Zelos reports image trust and culture state as separate outputs; no claim that QC predicts culture quality |
| Small, single-lab data | 480-image stratified subset of one public dataset (six cell lines, one automated microscope) | thresholds and references are not validated on other chips or microscopes |
| Retrieval gold set is optimistic | eval/README_retrieval.md: questions written from corpus chunks by an automated assistant, spot-checked; most papers hold a gold chunk; unanswerable questions are far off-domain | perfect paper-level scores leave no headroom; mode differences are within noise |
| Grounding checks are heuristic | numeric match by value and metric binding; literature support by lexical overlap (vector arm off in E5) | a claim can reuse a correct number with a wrong noun; semantic entailment is future work |
| Small LLM, small E5 | Workers AI 8B model; 38 cases + 14 adversarial prompts | rates have wide confidence intervals (reported) |
| Certificates are content hashes | SHA-256 of image bytes + ledger | not cryptographic signatures |
| No clinical use | research tool only | no diagnostic, treatment or clinical claims are made or allowed |
