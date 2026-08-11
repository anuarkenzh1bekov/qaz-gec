# QazGEC — Qazaq Grammatical Error Correction

Fine-tune a small seq2seq model (**google/umt5-base**) to correct spelling/grammar
errors in Kazakh text. Training must fit one GPU (RTX 5070, 12 GB) in ≤ 3 hours.

## Layout

```
data/
  raw/         corrected_small_kazakh_corpus_final.csv   # source clean corpus (gitignored)
  interim/     code_only.jsonl, gpt.jsonl, logs/         # build intermediates (gitignored)
  processed/   train.jsonl                               # final training set (gitignored, 90MB)
               val.jsonl                                 # synthetic held-out, for monitoring
               test_real.jsonl                           # held-out, GPT-noised — "before vs after"
               test_regression.jsonl                     # clean held-out — "what it cost"
scripts/       dataset builders (see pipeline below)
notebooks/     training notebook (train.ipynb) — created separately
```

## Data pipeline

Run from the repo root:

```bash
# 1. rule-based corruption of the clean corpus -> code_only.jsonl + val.jsonl
python scripts/make_dataset.py

# 2. add GPT-generated "human" noise, merge 70/30 -> processed/train.jsonl
python scripts/augment_gpt.py            # needs OPENAI_API_KEY (.env)

# 3. build the two evaluation sets (both held-out, not in train/val)
python scripts/make_real_test.py         # 120 GPT-noised pairs  -> test_real.jsonl
python scripts/make_regression.py        # 100 clean + 5 general -> test_regression.jsonl
```

## Evaluation design

Three sets, not a random 70/30 split — the task asks for two different things:

| Set | What it measures |
|-----|------------------|
| `val.jsonl` | training progress only (synthetic, in-distribution) |
| `test_real.jsonl` | **improvement** — did it learn to fix errors (before vs after) |
| `test_regression.jsonl` | **cost** — does it wrongly "fix" already-clean text; general skills lost |

`test_real` errors are GPT-synthesized (not human-collected) on held-out sentences,
with a different/harsher prompt than training — an honest but imperfect proxy.
