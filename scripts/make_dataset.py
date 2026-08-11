"""Build a Kazakh spelling/typo error-correction dataset.

Input : a CSV with a `content` column of CLEAN Kazakh text.
Output: JSONL pairs {"corrupted": <noisy>, "clean": <original>} split into
        train/val, ready for supervised fine-tuning.

The clean corpus is the target. We synthesize the *input* by injecting
realistic Kazakh errors so the model learns clean -> from -> noisy.

Usage:
    python make_dataset.py --csv corrected_small_kazakh_corpus_final.csv --out data
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and choke on Cyrillic; force UTF-8 stdout.
sys.stdout.reconfigure(encoding="utf-8")

# --- Error model -----------------------------------------------------------

# The #1 real-world Kazakh error: typing without a Kazakh keyboard layout, so
# Kazakh-specific letters collapse to their nearest Russian/Latin lookalike.
DEACCENT = {
    "ә": "а", "і": "и", "ң": "н", "ғ": "г", "ү": "у",
    "ұ": "у", "қ": "к", "ө": "о", "һ": "х",
}
DEACCENT.update({k.upper(): v.upper() for k, v in DEACCENT.items()})
KAZ_LETTERS = "аәбвгғдеёжзиійклмнңоөпрстуұүфхһцчшщъыьэюяқ"

# Per-word probability that a word gets corrupted at all. Kept low so most of
# a sentence stays clean (mirrors real text: a few mistakes, not garbled).
P_WORD = 0.22
# Given a word is corrupted, how the corruption type is chosen.
OP_WEIGHTS = {
    "deaccent": 0.50,   # ә->а etc. — dominant, most realistic
    "delete": 0.14,
    "transpose": 0.14,
    "duplicate": 0.10,
    "substitute": 0.12,
}

# Sentence length filter (characters) — keep short, learnable units.
MIN_LEN, MAX_LEN = 30, 400
# Cap total examples so training fits the 3h budget.
MAX_EXAMPLES = 120_000
VAL_FRACTION = 0.02

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def deaccent_word(w: str, rng: random.Random) -> str:
    idxs = [i for i, c in enumerate(w) if c in DEACCENT]
    if not idxs:
        return w  # nothing to deaccent; caller retries another op
    n = 1 if len(idxs) == 1 else rng.randint(1, min(2, len(idxs)))
    chars = list(w)
    for i in rng.sample(idxs, n):
        chars[i] = DEACCENT[chars[i]]
    return "".join(chars)


def typo_word(w: str, op: str, rng: random.Random) -> str:
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    if op == "delete":
        return w[:i] + w[i + 1:]
    if op == "duplicate":
        return w[:i] + w[i] + w[i:]
    if op == "substitute":
        return w[:i] + rng.choice(KAZ_LETTERS) + w[i + 1:]
    if op == "transpose":
        j = min(i, len(w) - 2)
        return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return w


def corrupt_word(w: str, rng: random.Random) -> str:
    op = rng.choices(list(OP_WEIGHTS), weights=list(OP_WEIGHTS.values()))[0]
    if op == "deaccent":
        return deaccent_word(w, rng)
    return typo_word(w, op, rng)


def corrupt_sentence(s: str, rng: random.Random) -> str:
    words = s.split(" ")
    out = []
    changed = False
    for w in words:
        if w and rng.random() < P_WORD:
            nw = corrupt_word(w, rng)
            if nw != w:
                changed = True
            out.append(nw)
        else:
            out.append(w)
    result = " ".join(out)
    # Guarantee a non-trivial pair: if nothing changed, force one deaccent.
    if not changed:
        for i, w in enumerate(words):
            dw = deaccent_word(w, rng)
            if dw != w:
                out[i] = dw
                return " ".join(out)
        return None  # no Kazakh letters and no typo landed — drop it
    return result


def sentences(text: str):
    for s in SENT_SPLIT.split(text.strip()):
        s = s.strip()
        if MIN_LEN <= len(s) <= MAX_LEN:
            yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/corrected_small_kazakh_corpus_final.csv")
    # code-only (rule-based) corruption is an intermediate; it gets merged with
    # GPT noise in augment_gpt.py before training.
    ap.add_argument("--train-out", default="data/interim/code_only.jsonl")
    ap.add_argument("--val-out", default="data/processed/val.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for p in (args.train_out, args.val_out):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, usecols=["content"])
    print(f"loaded {len(df)} documents")

    pairs, seen = [], set()
    for text in df["content"].dropna():
        for clean in sentences(str(text)):
            if clean in seen:
                continue
            seen.add(clean)
            corrupted = corrupt_sentence(clean, rng)
            if corrupted and corrupted != clean:
                pairs.append({"corrupted": corrupted, "clean": clean})
        if len(pairs) >= MAX_EXAMPLES:
            break

    rng.shuffle(pairs)
    n_val = int(len(pairs) * VAL_FRACTION)
    val, train = pairs[:n_val], pairs[n_val:]

    for path, rows in [(args.train_out, train), (args.val_out, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>6} -> {path}")

    # Sanity: how much did we actually change, on average?
    def edit_ratio(p):
        c, k = p["corrupted"], p["clean"]
        diff = sum(a != b for a, b in zip(c, k)) + abs(len(c) - len(k))
        return diff / max(len(k), 1)

    avg = sum(edit_ratio(p) for p in pairs[:2000]) / min(2000, len(pairs))
    print(f"avg char-edit ratio (first 2k): {avg:.3f}")
    print("\n--- samples ---")
    for p in pairs[:5]:
        print("NOISY:", p["corrupted"])
        print("CLEAN:", p["clean"])
        print()


if __name__ == "__main__":
    main()
