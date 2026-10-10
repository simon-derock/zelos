# Demo video script (target 4:00, hard limit 5:00) — record on zelos.philipsimonderock.com at 1920×1080

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | Hero section | "Organ-on-a-chip labs image their cultures every day. One defocused or bubble-filled image can silently corrupt a dataset or a digital twin, and an AI assistant asked about it may invent numbers. This is Zelos: every organ-on-a-chip image, certified." |
| 0:20–0:45 | How it works | "The LLM proposes, tools compute, the gate disposes. Seventeen physics-based metrics and a fixed rule table decide the verdict, the cause and the lab action with zero LLM calls. Cohere and Qdrant retrieve open-access OoC papers. A Workers AI model writes the explanation, and a deterministic gate checks every number and every citation." |
| 0:45–1:30 | Console: a real PASS image, then a real REVIEW/FAIL image | "These are real images from the public expert-labelled OoC dataset. Here's a pass. Here's a flagged one: the score, the cause, the metric that triggered it, and the action, for example re-focus or flat-field correction." |
| 1:30–2:10 | Console: corrupted samples (bubble, overexposure, illumination) | "We also inject known corruptions so we know the right answer. Zelos fails them and attaches cited evidence. Each citation links to a CC BY paper." |
| 2:10–2:45 | Ledger + toggle "Show what the LLM said without the gate" | "This is the trust moment. Every claim is typed and linked to a tool id or a passage. Toggle the gate off and you see the raw LLM draft. Across our evaluation the gate cuts unsupported claims and unsafe adversarial outputs to zero." |
| 2:45–3:30 | Benchmarks section | Read the tiles: E1 detection and attribution vs chance; the honest cross-session shift; E2 kNN vs artifact score ("the expert label is about the culture, not the image, so we report them separately"); E4 retrieval; E5 gate vs no gate with fault injection 100 of 100. |
| 3:30–3:50 | Terminal: `make judge` (or `zelos explain <image>`), GitHub Actions green | "One command reproduces every number offline, with no keys and no GPU. CI checks that every README number comes from the results files." |
| 3:50–4:10 | Footer | "Zelos is training-free, runs on a lab PC, and exposes MCP tools so any agent can use it. Next: longitudinal monitoring and neural OoC. Research use only. Not a medical device." |
