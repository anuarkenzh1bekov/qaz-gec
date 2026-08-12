# QazGEC — Qazaq Grammatical Error Correction

Fine-tune a small seq2seq model (**google/umt5-small**) to correct spelling/grammar
errors in Kazakh text. Training fits one GPU (RTX 5070, 12 GB) in ~2 hours.

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
notebooks/     train_qazqec.ipynb — training + inference notebook
models/        umt5-gec-small/best — trained model (gitignored)
```

## ⚠️ Requirement: transformers < 5

Use **transformers 4.57.x** (see `requirements.txt`). transformers **5.x ships a
broken UMT5 decoder causal mask** — future decoder tokens leak into past positions,
so the model trains to "cheat" (teacher-forced loss collapses to ~0.017) but emits
pure garbage at generation time. This is a library bug, not a modeling issue; on
4.57.6 the mask is correct (verified: Δ=0 when future tokens change).

## Train / infer

```bash
pip install -r requirements.txt
jupyter lab notebooks/train_qazqec.ipynb   # train + inference; saves models/umt5-gec-small/best
```

Config: umt5-small, MAX_LENGTH 128, eff. batch 32 (bs 16 × accum 2), Adafactor,
lr 3e-4, bf16, 1 epoch. Result: **eval_loss 0.207**, generation fixes errors and
mostly leaves already-clean text unchanged (~78% of clean inputs untouched).

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
