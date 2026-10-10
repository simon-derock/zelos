# Vision-LLM baseline vs Zelos

n = 20 per group, same images for every system.

| System | false flags on expert-good | flags on expert-bad | false flags on clean | sev-5 faults detected | unparsed |
|---|---:|---:|---:|---:|---:|
| Zelos v1.1 (no model trained) | 0.05 | 0.00 | 0.00 | 0.80 | 0 |
| @cf/mistralai/mistral-small-3.1-24b-instruct | 0.45 | 0.70 | 0.40 | 0.90 | 0 |
| @cf/meta/llama-4-scout-17b-16e-instruct | 0.65 | 0.95 | 0.65 | 1.00 | 0 |
| @cf/google/gemma-4-26b-a4b-it | n/a | n/a | n/a | n/a | 80 |
