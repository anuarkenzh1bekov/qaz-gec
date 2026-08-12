import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, torch, transformers
import pandas as pd
from transformers import (AutoTokenizer, UMT5Config, UMT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, EarlyStoppingCallback)
from datasets import Dataset, DatasetDict

print("transformers", transformers.__version__, flush=True)
assert transformers.__version__.startswith("4."), "MUST be 4.x — 5.x has UMT5 causal-mask bug"

MODEL_NAME = "google/umt5-base"
MAX_LENGTH = 128
OUT = "models/umt5-gec-fixed"

def load_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])

ds = DatasetDict({
    "train": Dataset.from_pandas(load_jsonl("data/processed/train.jsonl")),
    "validation": Dataset.from_pandas(load_jsonl("data/processed/val.jsonl")),
})

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess(ex):
    mi = tok(ex["corrupted"], max_length=MAX_LENGTH, truncation=True)
    lab = tok(text_target=ex["clean"], max_length=MAX_LENGTH, truncation=True)
    mi["labels"] = lab["input_ids"]
    return mi

tokenized = DatasetDict({s: d.map(preprocess, batched=True, remove_columns=d.column_names)
                         for s, d in ds.items()})

cfg = UMT5Config.from_pretrained(MODEL_NAME)
cfg.tie_word_embeddings = False           # umt5 is untied by design; keep trained lm_head
model = UMT5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=cfg)
model.config.use_cache = False

collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, label_pad_token_id=-100)

args = Seq2SeqTrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,        # eff batch 32
    bf16=True,
    optim="adafactor",
    num_train_epochs=1,
    learning_rate=3e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    group_by_length=True,
    eval_strategy="steps", eval_steps=1000,
    save_strategy="steps", save_steps=1000,
    save_total_limit=2, logging_steps=50,
    predict_with_generate=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
    data_collator=collator, processing_class=tok,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()

trainer.save_model(OUT + "/best")
tok.save_pretrained(OUT + "/best")
print("SAVED", OUT + "/best", flush=True)

# ---- verification: generation must produce clean Kazakh now ----
model.config.use_cache = True
model.eval()
rows = [json.loads(l) for l in open("data/processed/test_real.jsonl", encoding="utf-8")][:5]
print("\n===== GENERATION CHECK =====", flush=True)
for r in rows:
    x = tok(r["corrupted"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    with torch.no_grad():
        y = model.generate(x.input_ids, attention_mask=x.attention_mask,
                           max_new_tokens=160, num_beams=4,
                           no_repeat_ngram_size=3, early_stopping=True)
    print("SRC:", r["corrupted"])
    print("TGT:", r["clean"])
    print("OUT:", tok.decode(y[0], skip_special_tokens=True))
    print("-" * 50, flush=True)
print("DONE", flush=True)
