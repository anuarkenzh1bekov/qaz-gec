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
scripts/       dataset builders (see pipeline below) + train.py, infer.py
notebooks/     train_qazqec.ipynb — exploratory training notebook
models/        umt5-gec-fixed/best — trained model (gitignored)
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
python scripts/train.py          # ~2h on RTX 5070; saves models/umt5-gec-fixed/best
python scripts/infer.py "мәтінді осында жаз"
```

Config: umt5-base, MAX_LENGTH 128, eff. batch 32 (bs 8 × accum 4), Adafactor,
lr 3e-4, bf16, 1 epoch, `tie_word_embeddings=False` (umt5 is untied — keeps the
trained lm_head on reload). Result: **eval_loss 0.106**, generation fixes errors
and leaves already-clean text unchanged.

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
