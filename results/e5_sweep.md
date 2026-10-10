# E5 model sweep: which LLM behind the gate

git 3674e968 | LLM mode replay | explain-QC n=20

## Models and run status

| Model | Provider | Profile | Status | Cases | Responses in cache | Neurons recorded in cache |
|---|---|---|---|---|---|---|
| `@cf/meta/llama-3.1-8b-instruct-fast` | workers_ai | max_tokens=production (700) | ok | explain_qc=20 | 79 | 0 |
| `@cf/google/gemma-4-26b-a4b-it` | workers_ai | max_tokens=production (700), extra={"chat_template_kwargs": {"enable_thinking": false}} | ok | explain_qc=20 | 46 | 582 |
| `@cf/openai/gpt-oss-120b` | workers_ai | max_tokens=1500, extra={"reasoning_effort": "low"} | ok | explain_qc=20 | 48 | 2339 |
| `@cf/qwen/qwen3.8-27b` | workers_ai | max_tokens=production (700), extra={"chat_template_kwargs": {"enable_thinking": false}} | ok | explain_qc=20 | 27 | 4245 |
| `command-a-03-2025` | cohere | max_tokens=production (700) | no_cached_responses | 0 | 0 | 0 |

## Explain-QC cases

| Metric | llama-3.1-8b-fast (n=20) | gemma-4-26b-a4b (n=20) | gpt-oss-120b (n=20) | qwen3.8-27b (n=20) |
|---|---|---|---|---|
| Unsupported-claim rate BEFORE gate (draft) | 0.008 [0.002, 0.046] (1/119) | 0.000 [0.000, 0.041] (0/89) | 0.029 [0.010, 0.083] (3/102) | 0.000 [0.000, 0.033] (0/112) |
| Unsupported-claim rate after gate | 0.000 [0.000, 0.030] (0/123) | 0.000 [0.000, 0.041] (0/89) | 0.000 [0.000, 0.035] (0/106) | 0.000 [0.000, 0.033] (0/112) |
| Valid-citation rate, draft | 0.966 [0.828, 0.994] (28/29) | 1.000 [0.816, 1.000] (17/17) | 0.909 [0.722, 0.975] (20/22) | 1.000 [0.883, 1.000] (29/29) |
| Valid-citation rate, after gate | 1.000 [0.883, 1.000] (29/29) | 1.000 [0.816, 1.000] (17/17) | 1.000 [0.839, 1.000] (20/20) | 1.000 [0.883, 1.000] (29/29) |
| Numeric-hallucination rate, draft | 0.000 [0.000, 0.079] (0/45) | 0.000 [0.000, 0.129] (0/26) | 0.000 [0.000, 0.058] (0/63) | 0.000 [0.000, 0.040] (0/93) |
| Numeric-hallucination rate, after gate | 0.000 [0.000, 0.074] (0/48) | 0.000 [0.000, 0.129] (0/26) | 0.000 [0.000, 0.059] (0/61) | 0.000 [0.000, 0.040] (0/93) |
| Supported claims per case, draft | 5.90 | 4.45 | 4.95 | 5.60 |
| Supported claims per case, after gate (sd) | 6.15 (0.91) | 4.45 (0.80) | 5.30 (1.05) | 5.60 (0.58) |
| Claims blocked per case | 0.05 | 0.00 | 0.15 | 0.00 |
| JSON-format failure rate (per LLM call) | 0.000 [0.000, 0.155] (0/21) | 0.000 [0.000, 0.161] (0/20) | 0.000 [0.000, 0.149] (0/22) | 0.000 [0.000, 0.161] (0/20) |
| Explicit ABSTAIN rate (per LLM call) | 0.000 [0.000, 0.155] (0/21) | 0.000 [0.000, 0.161] (0/20) | 0.000 [0.000, 0.149] (0/22) | 0.000 [0.000, 0.161] (0/20) |
| Empty-claims reply rate (per LLM call) | 0.000 [0.000, 0.155] (0/21) | 0.000 [0.000, 0.161] (0/20) | 0.000 [0.000, 0.149] (0/22) | 0.000 [0.000, 0.161] (0/20) |
| Truncated / empty-text responses | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Tokens per answer (prompt + completion) | 1926 (1643 + 283) | 2026 (1753 + 273) | 2234 (1784 + 450) | 2095 (1685 + 410) |
| LLM calls per answer | 1.05 | 1.00 | 1.10 | 1.00 |
| Answers that reached the LLM | 20/20 | 20/20 | 20/20 | 20/20 |
| Latency p50 / p95 (ms), answers that called the LLM | 2117 / 3184 | 6541 / 8126 | 5119 / 12862 | 10111 / 44698 |
| Cost per 100 answers (USD) | $0.0182 | $0.0257 | $0.0962 | $0.2069 |
| Neurons per 100 answers | 1662 | 2338 | 8748 | 18809 |

## Verdict (computed)

- explain_qc: best supported claims per case at zero unsupported after the gate = llama-3.1-8b-fast (6.15/case, n=20). Cheapest within 5% of best = llama-3.1-8b-fast (6.15/case, $0.0182 per 100 answers).
  - gemma-4-26b-a4b minus best: -1.70 supported claims/case [-2.30, -1.15] (paired bootstrap, n=20)
  - gpt-oss-120b minus best: -0.85 supported claims/case [-1.50, -0.20] (paired bootstrap, n=20)
  - qwen3.8-27b minus best: -0.55 supported claims/case [-1.30, -0.10] (paired bootstrap, n=20)

## Prices and call ledger

- llama-3.1-8b-fast: $0.045/M in, $0.384/M out. Source: Workers AI pricing page, row @cf/meta/llama-3.1-8b-instruct-fp8-fast (the -fast id is not returned by the models search catalog; same price assumed).
- gemma-4-26b-a4b: $0.1/M in, $0.3/M out. Source: Workers AI models catalog.
- gpt-oss-120b: $0.35/M in, $0.75/M out. Source: Workers AI models catalog.
- qwen3.8-27b: $0.45/M in, $3.2/M out. Source: Workers AI models catalog.
- command-a-03-2025: trial key. Source: Cohere trial key (no per-token charge; production price not used).
- Catalog price check: "replay"
- Live responses in this invocation: 0; HTTP attempts by provider: {"workers_ai": 0, "cohere": 0}; Workers AI neurons billed this run: 0; Cohere cap: 120.
- command-a-03-2025: explain_qc: no usable responses (3 case(s) unavailable)

Notes:
- Same cases, prompts, retriever and gate for every model; only the model (and, for reasoning models, max_tokens and the reasoning-off/low switch) differs. 'draft' is the first model output before the gate; 'gated' is what the user sees after the gate and at most one repair call.
- Unsupported/citation/numeric rates are judged by independent re-verification of the claims (eg.judge), as in E5.
- JSON-format failure = a model reply that is neither strict JSON claims nor an ABSTAIN (it is parsed as free text).
- Latency is the recorded provider latency of the live call plus local step time; replayed cache entries keep their recorded latency.
