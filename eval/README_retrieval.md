# Retrieval gold set (E4)

`retrieval_gold.jsonl` has one JSON object per line: `qid, question, type, gold_chunk_ids, gold_pmcids`
(reworded rows also carry `paraphrase_of`).

| type | meaning |
|---|---|
| `answerable` | a question whose answer is stated in one specific corpus chunk (gold chunk and its paper) |
| `reworded` | a paraphrase of an answerable question with the same gold chunk, written with different vocabulary |
| `unanswerable` | a question clearly outside the corpus (sports, finance, cooking, ...); the correct behaviour is to abstain |

Counts per type are printed in `results/e4.json` (`gold`), computed from this file.

## How it was made (and its limits)

- The questions were written by an automated assistant that read the text of the chosen chunks in
  `corpus/chunks.jsonl`, then spot-checked by reading a sample of question/chunk pairs. They were **not** written or
  independently verified by a domain expert and there was no second annotator, so no inter-annotator agreement exists.
- Chunks were picked from keyword-matched candidates (one chunk per paper where possible) around OoC imaging and
  culture topics: bubbles, autofocus, illumination correction, seeding density, flow and shear stress, segmentation.
  The pool is therefore biased towards chunks that contain distinctive facts, and is not a random sample of the corpus.
- Questions were written from the chunk text, so they share vocabulary with their gold chunk. This favours lexical
  retrieval (BM25). The `reworded` rows reduce, but do not remove, that bias.
- Only the source chunk is marked as gold. Another chunk, or another paper, may also answer a question; such hits
  count as misses, so absolute scores are conservative. Paper-level metrics are primary because they are less
  sensitive to this; chunk-level metrics are secondary.
- The unanswerable questions are far outside the domain, so abstention is easy. Hard near-domain unanswerable
  questions are not covered, and the reported abstention accuracy must not be read as robustness to adversarial queries.
- Sample sizes are small (see the `n` fields); the Wilson 95% intervals in `results/e4.json` are wide and differences
  between modes can be within noise.
- Abstention thresholds (`configs/rag.yaml`) were fixed before evaluation and not tuned on this set.

## Reproduce

```
python scripts/eval_retrieval.py            # uses the Cohere key if present, otherwise replays committed caches
python scripts/eval_retrieval.py --offline  # forces cache replay (no key); writes results/e4.json and results/e4.md
```
