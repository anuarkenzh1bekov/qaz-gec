"""Augment the dataset with GPT-generated human-like noise.

Rule-based corruption (make_dataset.py) can't produce human mistakes: wrong
suffixes, phonetic spellings, joined/split words. gpt-4o-mini corrupts clean
sentences for the 30% slice, then merges code(70%) + gpt(30%) -> train.jsonl.

    export OPENAI_API_KEY=sk-...
    python augment_gpt.py                 # full run
    python augment_gpt.py --smoke 30      # cheap smoke test
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

SYSTEM = (
    "You corrupt clean Kazakh sentences to build training data for a spelling and "
    "grammar correction model. For each numbered sentence, produce a NOISY version "
    "that a real Kazakh writer might actually type by mistake. Introduce 2-5 realistic "
    "errors such as: missing Kazakh-specific letters (ә і ң ғ ү ұ қ ө һ typed as а и н "
    "г у у к о х), wrong or dropped case/possessive suffixes, phonetic misspellings, "
    "doubled or dropped letters, wrongly joined or split words, missing or extra spaces. "
    "Rules: keep the same meaning and word order; do NOT translate; do NOT rephrase; do "
    "NOT correct anything; do NOT add or delete whole words. Return ONLY a JSON object "
    'mapping each index as a string to its corrupted sentence, e.g. {"1": "...", "2": "..."}.'
)

# Per-batch error "personas", rotated across batches to span many error modes.
STYLES = [
    "For this batch, focus on missing Kazakh-specific letters replaced by Russian/Latin "
    "lookalikes (ә->а, қ->к, ғ->г, ұ/ү->у, ө->о, ң->н, і->i, һ->х).",
    "For this batch, focus on wrong or dropped grammatical suffixes: case endings, "
    "possessive, plural, and verb-tense agreement mistakes.",
    "For this batch, focus on phonetic misspellings — spell words the careless way they "
    "sound in casual speech.",
    "For this batch, focus on mobile fast-typing slips: adjacent-key hits and missing "
    "spaces so two words run together.",
    "For this batch, focus on doubled, dropped, or transposed letters from fast careless typing.",
    "For this batch, focus on wrongly split or merged words and misplaced or duplicated "
    "punctuation and spacing.",
    "For this batch, mix several different error types naturally, as an average careless writer would.",
]


def load_clean_pool(path: Path) -> list[str]:
    seen = set()
    for line in path.open(encoding="utf-8"):
        c = json.loads(line)["clean"]
        if c not in seen:
            seen.add(c)
    return list(seen)


def looks_valid(clean: str, noisy: str) -> bool:
    if not noisy or noisy == clean:
        return False
    # Reject rephrases/hallucinations: length must stay close to the original.
    return 0.7 <= len(noisy) / max(len(clean), 1) <= 1.4


def corrupt_batch(client: OpenAI, model: str, sents: list[str], style: str = "") -> list[tuple[str, str]]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    resp = client.chat.completions.create(
        model=model,
        temperature=1.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM + " " + style},
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


def run_with_retry(client, model, sents, style="", tries=4):
    for attempt in range(tries):
        try:
            return corrupt_batch(client, model, sents, style)
        except Exception as e:  # rate limit / transient / bad json
            if attempt == tries - 1:
                print(f"  batch failed permanently: {e}", flush=True)
                return []
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-file", default="data/interim/code_only.jsonl")
    ap.add_argument("--out", default="data/interim/gpt.jsonl")
    ap.add_argument("--merged", default="data/processed/train.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--frac", type=float, default=0.30, help="GPT share of final set")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gpt", type=int, default=0, help="hard cap; 0 = ratio-driven")
    ap.add_argument("--smoke", type=int, default=0, help="only N sentences, for testing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Fall back to a local .env for OPENAI_API_KEY.
    if not os.getenv("OPENAI_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in the environment or a .env file first.")

    client = OpenAI()
    rng = random.Random(args.seed)
    out_path, code_path = Path(args.out), Path(args.code_file)

    n_code = sum(1 for _ in code_path.open(encoding="utf-8"))
    pool = load_clean_pool(code_path)
    rng.shuffle(pool)

    # GPT pairs needed to hit the target ratio: gpt / (code+gpt) = frac.
    n_target = round(n_code * args.frac / (1 - args.frac))
    if args.max_gpt:
        n_target = min(n_target, args.max_gpt)
    if args.smoke:
        n_target = args.smoke

    # Resume: skip clean sentences already corrupted in a previous run.
    done = set()
    if out_path.exists() and not args.smoke:
        for line in out_path.open(encoding="utf-8"):
            done.add(json.loads(line)["clean"])
    todo = [s for s in pool if s not in done][:max(0, n_target - len(done))]
    print(f"code pairs={n_code}  target gpt={n_target}  already={len(done)}  to do={len(todo)}")
    if not todo:
        print("nothing to generate.")
    else:
        batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
        lock = threading.Lock()
        written = [0]
        with out_path.open("a", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_with_retry, client, args.model, b, STYLES[i % len(STYLES)]): b
                       for i, b in enumerate(batches)}
            for n, fut in enumerate(as_completed(futures), 1):
                pairs = fut.result()
                with lock:
                    for noisy, clean in pairs:
                        f.write(json.dumps({"corrupted": noisy, "clean": clean,
                                            "source": "gpt"}, ensure_ascii=False) + "\n")
                    written[0] += len(pairs)
                    f.flush()
                if n % 20 == 0 or n == len(batches):
                    print(f"  batch {n}/{len(batches)}  kept={written[0]}", flush=True)
        print(f"gpt pairs written total: {written[0]}")

    if args.smoke:
        print("\n--- smoke samples ---")
        for line in list(out_path.open(encoding="utf-8"))[-args.smoke:]:
            p = json.loads(line)
            print("NOISY:", p["corrupted"])
            print("CLEAN:", p["clean"], "\n")
        return

    # Merge code(70%) + gpt(30%) into one shuffled training file.
    merged = []
    for line in code_path.open(encoding="utf-8"):
        r = json.loads(line)
        r["source"] = "code"
        merged.append(r)
    for line in out_path.open(encoding="utf-8"):
        merged.append(json.loads(line))
    rng.shuffle(merged)
    with Path(args.merged).open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_gpt = sum(1 for r in merged if r["source"] == "gpt")
    print(f"\nmerged {len(merged)} -> {args.merged}  "
          f"(code={len(merged) - n_gpt}, gpt={n_gpt}, gpt share={n_gpt / len(merged):.1%})")


if __name__ == "__main__":
    main()
