"""Build a regression test set: clean, held-out Kazakh sentences.

Goal: measure the *cost* of fine-tuning. We feed the model correct text and
check whether it wrongly "corrects" what was already fine. So input == target.

Every sentence here is guaranteed NOT to appear in train/val (held-out), using
the SAME sentence-splitting rules as make_dataset.py so the comparison is fair.

Output: data/test_regression.jsonl with rows
    {"input": <clean>, "target": <clean>, "type": "keep"}

Optionally appends a few hand-written general tasks (type "general") that are
NOT about correction, to see whether the base model's other skills degraded.

Usage:
    python make_regression.py --csv corrected_small_kazakh_corpus_final.csv \
        --out data/test_regression.jsonl --n 100
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# --- Must match make_dataset.py exactly, or "held-out" is meaningless. -------
MIN_LEN, MAX_LEN = 30, 400
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def sentences(text: str):
    for s in SENT_SPLIT.split(text.strip()):
        s = s.strip()
        if MIN_LEN <= len(s) <= MAX_LEN:
            yield s


def load_used(paths):
    """Every `clean` sentence already consumed by train/val."""
    used = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "clean" in obj:
                    used.add(obj["clean"])
    return used


# A few non-correction tasks: does the base model still do these after SFT?
GENERAL_TASKS = [
    {"input": "Қазақстанның астанасы қай қала?", "target": "Астана", "type": "general"},
    {"input": "Аптада неше күн бар?", "target": "Жеті", "type": "general"},
    {"input": "Мына сөйлемді аяқта: Күн шығыстан ...", "target": "шығады", "type": "general"},
    {"input": "2 + 2 нешеге тең?", "target": "4", "type": "general"},
    {"input": "«Кітап» сөзінің көпше түрі қандай?", "target": "кітаптар", "type": "general"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/corrected_small_kazakh_corpus_final.csv")
    ap.add_argument("--out", default="data/processed/test_regression.jsonl")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--n", type=int, default=100, help="clean held-out sentences")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-general", action="store_true",
                    help="skip the general (non-correction) probe tasks")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    used = load_used([args.train, args.val])
    print(f"held-out filter: {len(used)} sentences already in train/val")

    df = pd.read_csv(args.csv, usecols=["content"])
    print(f"loaded {len(df)} documents")

    # Collect candidate clean sentences that were NOT used in training.
    candidates, seen = [], set()
    for text in df["content"].dropna():
        for s in sentences(str(text)):
            if s in used or s in seen:
                continue
            seen.add(s)
            candidates.append(s)

    print(f"found {len(candidates)} held-out clean sentences")
    if len(candidates) < args.n:
        print(f"WARNING: only {len(candidates)} available, wanted {args.n}")

    rng.shuffle(candidates)
    picked = candidates[: args.n]

    rows = [{"input": s, "target": s, "type": "keep"} for s in picked]
    if not args.no_general:
        rows.extend(GENERAL_TASKS)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_keep = sum(r["type"] == "keep" for r in rows)
    n_gen = sum(r["type"] == "general" for r in rows)
    print(f"wrote {len(rows)} rows -> {out}  (keep={n_keep}, general={n_gen})")
    print("\n--- samples (keep) ---")
    for r in rows[:3]:
        print(r["input"][:100])


if __name__ == "__main__":
    main()
