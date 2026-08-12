"""QazGEC inference — correct Kazakh spelling/grammar errors.

REQUIRES transformers < 5.  transformers 5.x ships a broken UMT5 decoder
causal mask (future tokens leak into past positions), which makes the model
train to "cheat" and produce garbage at generation time. Trained/served on
transformers 4.57.x.

    python scripts/infer.py "мәтінді осында жаз"
    python scripts/infer.py                       # runs a few built-in examples
"""
import sys, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CKPT = "models/umt5-gec-fixed/best"
MAX_LENGTH = 128

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(
    "cuda" if torch.cuda.is_available() else "cpu").eval()
model.config.use_cache = True


@torch.no_grad()
def correct(text: str) -> str:
    x = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    y = model.generate(
        x.input_ids, attention_mask=x.attention_mask,
        max_new_tokens=MAX_LENGTH, num_beams=4,
        no_repeat_ngram_size=3, early_stopping=True,
    )
    return tok.decode(y[0], skip_special_tokens=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(correct(" ".join(sys.argv[1:])))
    else:
        for t in [
            "Шамалыдан сон біз дүнйеге келдик.",
            "«Қазақ» газетінін редакторы болды.",
            "Зерттеулерге сәйкес, тез дамып келе жатқан кәсіптердін 75%-ы STEM салаларындагы жүзыреттілікті кажет етет.",
        ]:
            print("IN :", t)
            print("OUT:", correct(t))
            print("-" * 50)
