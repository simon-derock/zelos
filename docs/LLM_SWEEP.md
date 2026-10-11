# LLM sweep: which model behind the gate?

Because every claim is verified by the deterministic gate, the LLM is swappable. This sweep measures which
model is worth using. Full tables: `results/e5_sweep.md` / `results/e5_sweep.json`; reproduce offline with
`python scripts/eval_llm_sweep.py --replay --ask` (needs no keys; responses are in `eval/llm_cache`).

## Method

Same 20 explain-QC cases, prompts, offline BM25 retriever (cached queries replayed) and gate as E5
(`scripts/eval_grounding.py` helpers are imported, not copied). Only the model changes, via
`zelos.agent.llm` (`provider`, `model`, record/replay cache keyed on provider+model+messages+params).
"Draft" is the first model output before the gate; "after gate" is what the user sees after the gate and at
most one repair call. Claims are re-verified independently; CIs are Wilson 95%. Reasoning models cannot answer
the strict-JSON prompt at the production `max_tokens=700` with default reasoning (one-case probe: `content: null`,
`finish_reason: length` for gemma-4 and gpt-oss), so they run with thinking off (gemma-4, qwen3.8) or
`reasoning_effort: low` and `max_tokens=1500` (gpt-oss). Cost = measured tokens x catalog price.

## Results, explain-QC (n=20)

| Metric | llama-3.1-8b-fast | gemma-4-26b-a4b | gpt-oss-120b | qwen3.8-27b |
|---|---|---|---|---|
| Unsupported rate before gate | 0.008 [0.002, 0.046] (1/119) | 0.000 [0.000, 0.041] (0/89) | 0.029 [0.010, 0.083] (3/102) | 0.000 [0.000, 0.033] (0/112) |
| Unsupported rate after gate | 0.000 [0.000, 0.030] (0/123) | 0.000 [0.000, 0.041] (0/89) | 0.000 [0.000, 0.035] (0/106) | 0.000 [0.000, 0.033] (0/112) |
| Valid-citation rate before / after | 0.97 / 1.00 | 1.00 / 1.00 | 0.91 / 1.00 | 1.00 / 1.00 |
| Numeric hallucination (draft) | 0/45 | 0/26 | 0/63 | 0/93 |
| Supported claims per case | 6.15 | 4.45 | 5.30 | 5.60 |
| JSON-format failures per call | 0/21 | 0/20 | 0/22 | 0/20 |
| Tokens per answer | 1926 | 2026 | 2234 | 2095 |
| LLM calls per answer | 1.05 | 1.00 | 1.10 | 1.00 |
| Latency p50 / p95 (ms) | 2137 / 3210 | 6545 / 8129 | 5119 / 12884 | 10128 / 44713 |
| USD per 100 answers | $0.018 | $0.026 | $0.096 | $0.207 |

Computed verdict: best supported claims per case at zero unsupported after the gate is llama-3.1-8b-fast
(6.15/case); it is also the cheapest within 5% of best. Paired difference to it (95% bootstrap CI):
gemma-4 -1.70 [-2.30, -1.15], gpt-oss -0.85 [-1.50, -0.20], qwen3.8 -0.55 [-1.30, -0.10] claims/case.
On the 18 ask-literature cases (n=18; qwen3.8 only 3 cases, not compared) llama again leads (2.67 vs 0.33
gemma-4, 0.67 gpt-oss supported claims/case), but the larger models reply `{"claims": []}` to about half of the
calls (gemma-4 14/26, gpt-oss 13/26, llama 5/30) while llama's draft is 18% unsupported (11/60) before the gate.

## Caveats

- Cohere `command-a-03-2025` was not measured: the shared trial key returned HTTP 429 (1000 calls/month
  reached); recorded as `quota_exhausted`, no cell filled. The provider code is covered by mocked tests only.
- Only the gate makes the 0% unsupported rate true by construction; the differences are in how many supported
  claims survive. Claim count rewards verbosity; supported claims are not weighted by usefulness.
- 20 cases, single run at temperature 0, one network location, 3 parallel workers: latencies are indicative.
  qwen3.8 ask-literature stopped after 3 cases (neuron budget). Reasoning settings were chosen by one probe each.
- `@cf/meta/llama-3.1-8b-instruct-fast` is not in the models-search catalog; its price is the pricing-page row
  for `...-fp8-fast`. Other prices matched the catalog. Neurons used: 6,965 in the sweep plus about 935 in manual probes
  (daily free allowance 10,000).
