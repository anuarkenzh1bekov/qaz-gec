"""Build a *held-out* error-correction test set with GPT-generated noise.

This is the set used for the "before vs after" table. To be a fair test (not a
re-measure of training) it differs from augment_gpt.py on TWO axes:

  1. Source sentences are HELD-OUT: clean sentences from the CSV that never
     appear in train/val (same splitting rules as make_dataset.py).
  2. A different, harsher error prompt: realistic *human* mistakes as a native
     Kazakh writer would actually make, denser and more varied than training.

Honest label: these are GPT-synthesized "realistic" errors, NOT errors
collected from real humans. Say so in the write-up.

Output: data/test_real.jsonl with rows
    {"corrupted": <noisy>, "clean": <held-out original>, "source": "gpt-test"}

Usage:
    python make_real_test.py --n 120 --model gpt-4o-mini
    python make_real_test.py --smoke 10        # cheap check first
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# --- Held-out sentence extraction: identical rules to make_dataset.py --------
MIN_LEN, MAX_LEN = 30, 400
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def sentences(text: str):
    for s in SENT_SPLIT.split(text.strip()):
        s = s.strip()
        if MIN_LEN <= len(s) <= MAX_LEN:
            yield s


def load_used(paths):
    used = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "clean" in obj:
                    used.add(obj["clean"])
    return used


# Deliberately different from augment_gpt.py: denser, more human, and it may
# touch grammar/word-choice the way a real writer slips — a harder test.
SYSTEM = (
    "You simulate how a real, careless native Kazakh writer types on a phone or "
    "keyboard without a proper Kazakh layout. For each numbered clean sentence, "
    "produce a realistic NOISY version containing 3-6 mistakes a real person would "
    "actually make: Kazakh-specific letters dropped to Russian/Latin lookalikes "
    "(ә і ң ғ ү ұ қ ө һ -> а и н г у у к о х), wrong or missing case/possessive/plural "
    "suffixes, phonetic spellings the way words sound in speech, doubled/dropped/swapped "
    "letters, wrongly joined or split words, missing or extra spaces, and misplaced "
    "punctuation. Make it look genuinely human and messy, not uniformly corrupted. "
    "Rules: keep the SAME meaning and word order; do NOT translate, rephrase, summarize, "
    "or add/remove whole ideas. Return ONLY a JSON object mapping each index as a string "
    'to its corrupted sentence, e.g. {"1": "...", "2": "..."}.'
)


def looks_valid(clean: str, noisy: str) -> bool:
    if not noisy or noisy == clean:
        return False
    return 0.7 <= len(noisy) / max(len(clean), 1) <= 1.4


def corrupt_batch(client, model, sents):
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    resp = client.chat.completions.create(
        model=model,
        temperature=1.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": numbered},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    out = []
    for i, clean in enumerate(sents):
        noisy = str(data.get(str(i + 1), "")).strip()
        if looks_valid(clean, noisy):
            out.append((noisy, clean))
    return out


def run_with_retry(client, model, sents, tries=4):
    for attempt in range(tries):
        try:
            return corrupt_batch(client, model, sents)
        except Exception as e:
            if attempt == tries - 1:
                print(f"  batch failed permanently: {e}", flush=True)
                return []
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/corrected_small_kazakh_corpus_final.csv")
    ap.add_argument("--out", default="data/processed/test_real.jsonl")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in the environment or a .env file first.")

    client = OpenAI()
    rng = random.Random(args.seed)
    n_target = args.smoke or args.n

    used = load_used([args.train, args.val])
    print(f"held-out filter: {len(used)} sentences already in train/val")

    df = pd.read_csv(args.csv, usecols=["content"])
    candidates, seen = [], set()
    for text in df["content"].dropna():
        for s in sentences(str(text)):
            if s not in used and s not in seen:
                seen.add(s)
                candidates.append(s)
    print(f"held-out clean pool: {len(candidates)}")

    rng.shuffle(candidates)
    # Over-sample source sentences a bit; GPT validation drops some.
    todo = candidates[: int(n_target * 1.5) + args.batch]

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    written = 0
    with out_path.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_with_retry, client, args.model, b) for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            for noisy, clean in fut.result():
                if written >= n_target:
                    break
                f.write(json.dumps({"corrupted": noisy, "clean": clean,
                                    "source": "gpt-test"}, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            if written >= n_target:
                break

    print(f"wrote {written} pairs -> {out_path}")
    print("\n--- samples ---")
    for line in list(out_path.open(encoding="utf-8"))[:5]:
        p = json.loads(line)
        print("NOISY:", p["corrupted"])
        print("CLEAN:", p["clean"], "\n")


if __name__ == "__main__":
    main()
