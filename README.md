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

## Results & cost

### Training metrics (umt5-small, 1 epoch)

| step | train loss | val loss |
|------|-----------|----------|
| 1000 | 0.630 | 0.389 |
| 2000 | 0.488 | 0.284 |
| 3000 | 0.413 | 0.231 |
| 4000 | 0.389 | 0.213 |
| 5000 | 0.369 | **0.207** |

### Held-out results (`test_real` — did it learn to fix?)

| model | char-acc ↑ | exact-match ↑ | clean text kept ↑ |
|---|---|---|---|
| raw input (no model) | 0.468 | — | — |
| raw umt5-small (no fine-tune) | 0.016 | 0.000 | 0.00 |
| **fine-tuned umt5-small** | **0.522** | **0.083** | **0.78** |

Raw `umt5-small` can't correct at all — it's a span-fill pretrain and emits
garbage (`......`). The whole skill comes from fine-tuning.

### Edit-level metrics (`test_real`, 120 sentences, beam search)

Char-accuracy is coarse; for correction what matters is how much closer the output
gets to the target and how many sentences actually improve.

| metric | before (input) | after (model) |
|---|---|---|
| avg char edit-distance to target | 37.07 | **19.38** |
| word-level accuracy | 0.463 | **0.658** |
| **error reduction** | — | **47.7%** |

Per-sentence outcome: **improved 88.3%** (106/120), unchanged 7.5%, **worse 4.2%** (5/120).

The model roughly halves the distance to the correct sentence and improves 88% of
inputs while breaking only 4%. The low exact-match (0.083) is expected — one leftover
error in a long sentence zeroes it — so edit-distance and improvement rate are the
honest signals here.

**Before vs after, same input:**

| input (noisy) | raw | fine-tuned | target |
|---|---|---|---|
| `…iс-әрекети … техникалык кабілетти … кажет етет.` | `……` | `…iс-әрекеті … техникалық қабілетті … қажет етеді.` | `…іс-әрекеті … қабілетті … қажет етеді.` |

### What we paid for it

- **Over-correction** — the main cost: ~22% of already-clean sentences get
  needlessly changed (`test_regression`: 78% left untouched).
- **General ability** — the base model's non-correction skills (QA, arithmetic)
  are effectively gone; after single-task SFT it only knows "rewrite, fixed."

### Cost to train

- **Hardware**: 1× RTX 5070 (12 GB), bf16, Adafactor, batch 16×2, 1 epoch (~5.3k steps).
- **Time**: ~12 h · **VRAM**: ~9 GB peak (~11 GB reserved) · **Money**: $0 compute
  (local GPU), ~$3 on gpt-4o-mini for data (dataset augmentation + `test_real`).

### Where we stopped, and why

- First tried **umt5-base**, but transformers **5.x has a broken UMT5 decoder causal
  mask** (future tokens leak into past positions → loss collapses to ~0.017,
  generation is garbage). Fixed by pinning **transformers 4.57.x**.
- **umt5-base** then scored higher (char-acc 0.60 vs 0.52) but is 2× the size and
  ~3× slower. With the **deadline closing in**, we settled on **umt5-small** at
  1 epoch (`eval_loss 0.207`) — good enough, fast, small footprint.

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
