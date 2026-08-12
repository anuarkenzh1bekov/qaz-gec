# QazGEC — Qazaq Grammatical Error Correction

Fine-tune a small seq2seq model (**google/umt5-small**) to correct spelling/grammar
errors in Kazakh text. Training fits one GPU (RTX 5070, 12 GB) in ~2 hours.

## Как я это делал (для ревьюера)

Я выбрал 2-ое задание и решил дообучить модель для исправления ошибок в тексте.
Пару часов потратил на ресёрч и остановился на **google/umt5**.

### Почему выбрал google/umt5

- **Знает казахский.** UMT5 предобучен на mC4 (100+ языков, вкл. казахский).
  Хорошей одноязычной казахской seq2seq-модели нет (или не нашёл), так что
  мультиязычная база — реалистичный выбор.
- **Влезает в GPU.** ~9 ГБ VRAM при обучении, помещается в RTX 5070 (12 ГБ).
- **Архитектура под задачу.** GEC — это условная генерация «почти-копии»: выход
  почти совпадает с входом, отличаясь локальными правками. Под такую постановку
  архитектурно подходит encoder-decoder, и UMT5 выбран именно из этих соображений.
- **Двунаправленный энкодер.** Модель кодирует всё предложение целиком, с
  контекстом в обе стороны. Многие казахские ошибки нельзя распознать по левому
  контексту — например, восстановление специфичных букв (қ, ң, ғ, ә) или
  согласование падежных/притяжательных суффиксов требует взгляда на слово справа.
- **Cross-attention как механизм копирования.** Декодер на каждом шаге обращается
  к закодированному источнику. Поскольку в GEC подавляющее большинство токенов надо
  просто перенести без изменений, cross-attention работает встроенной опорой для
  точного копирования и удерживает модель от лишнего перефразирования.
- **Согласованность с предобучением.** UMT5 предобучен на задаче denoising —
  восстановление чистого текста из зашумлённого. Это по сути та же операция, что и
  коррекция ошибок, поэтому дообучение стартует не с нуля, а с уже релевантного
  индуктивного смещения.

### Датасет

Взял корпус **small_kazakh_corpus**, потому что нужен был чистый казахский текст без
ошибок. Такие корпуса вообще редкость для казахского — большинство открытых датасетов
либо маленькие, либо это параллельные переводы, а тут именно связный монолингвальный
текст, которого хватило с запасом.

Дальше встал вопрос: как получить пары ошибка/исправление. Первая мысль — накидать шум
кодом: переставить буквы, что-то удалить, продублировать. Быстро и бесплатно, но такие
ошибки не похожи на то, как реально ошибаются люди на казахском. У казахского своя
специфика: путают қ/к, ғ/г, неправильно клеят аффиксы, нарушают гармонию гласных.
Рандомная перестановка букв этого не воспроизведёт.

Поэтому замиксовал: часть данных нагенерировал скриптом (дёшево, для объёма), а часть —
через GPT-4o-mini с промптом «сделай реалистичную ошибку носителя языка». Ушло ~$3.

### Как поделили и зачем

Ключевая идея: не случайный 70/30-сплит, а 4 набора под разные вопросы.

1. **train (168k)** — обучение. Сам смешан code(70%) + gpt(30%):
   - code даёт объём и контролируемое распределение ошибок (деаккентизация, опечатки,
     суффиксы) — дёшево и масштабируемо.
   - gpt добавляет реалистичные человеческие ошибки, которые правилами не сгенерить
     (фонетика, слитно/раздельно, падежи). 30% — разбавить, но не удорожать датасет.
2. **val (2 400)** — только мониторинг обучения (eval_loss). Синтетика той же природы,
   что train; не финальная метрика, а датчик переобучения.
3. **test_real (120)** — «улучшение»: сработало ли исправление. Held-out (нет ни в
   train, ни в val), с более жёстким GPT-промптом, чем в train → это тест, а не
   переизмерение обучающего распределения.
4. **test_regression (105)** — «цена»: не портит ли модель уже чистый текст.
   input == target, held-out. Плюс 5 general-задач (QA, арифметика) — проба, не
   деградировали ли базовые навыки после SFT.

Почему так, а не рандом-сплит: случайный сплит из одного распределения ответил бы только
«переобучились ли мы на своей синтетике». А раздельные held-out наборы с другим
распределением ошибок (test_real) и с обратной задачей (test_regression) дают честную
оценку обобщения и побочного вреда.

### Первая попытка (umt5-base) и баг transformers

Сначала пробовал **umt5-base** — крупнее, точность выше, хотел выжать максимум. И тут
всё встало из-за **transformers 5.x**: у UMT5 в этой версии сломан decoder causal mask,
будущие токены протекают в прошлые позиции. На обучении это выглядело обманчиво
прекрасно — teacher-forced loss падал почти в ноль (~0.017), будто модель гений. А на
генерации вылезал чистый мусор, потому что модель училась «подсматривать» вперёд, чего
на инференсе нет.

Мучился долго, пока не понял, что дело не в данных и не в гиперпараметрах, а в самой
библиотеке. Проверил гипотезу прямо: менял будущие токены и смотрел, меняется ли выход
на прошлых позициях. Меняется — значит маска дырявая, баг подтверждён. Фикс — откат на
**transformers 4.57.x** (проверено, Δ=0 когда будущие токены меняются). После этого base
поехал нормально, но время уже съелось, и с учётом дедлайна финально остановился на
**umt5-small**.

### Обучение

Supervised fine-tuning seq2seq: вход = зашумлённое предложение, таргет = чистое.
Токенизация SentencePiece, MAX_LENGTH=128, обрезка по длине. Метки паддятся -100
(DataCollatorForSeq2Seq), чтобы паддинг не шёл в loss. Лосс — cross-entropy с teacher
forcing.

```python
Seq2SeqTrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    bf16=True,
    optim="adafactor",
    num_train_epochs=1,
    learning_rate=3e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    eval_steps=1000, save_steps=1000, save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False,
)
```

Почему так: **bf16** — вдвое меньше памяти, быстрее, без loss scaling (в отличие от
fp16). **Adafactor** — не хранит моменты как Adam, экономит VRAM, за счёт этого влезает
батч. **Батч 16×2** — 16 максимум для 12 ГБ при длине 128, аккумуляция добирает до
эффективного 32 без OOM. **lr 3e-4 + warmup 500 + линейный спад** — стандарт для T5,
warmup гасит ранние скачки градиента. **1 эпоха** — на 168k хватает, кривая val
(0.389 → 0.207) вышла на плато; EarlyStopping patience=3 как страховка. Лучший
чек-пойнт по eval_loss, а не последний. **use_cache=False** при обучении — KV-кэш нужен
только для генерации.

Ход обучения: **5 261 шаг** (168 326 / 32), 1 эпоха. Val loss монотонно падал
0.389 → 0.284 → 0.231 → 0.213 → 0.207 (шаги 1000→5000), переобучения нет. Лучший
чек-пойнт (eval_loss=0.207) сохранён в `models/umt5-gec-small/best`.

### Результаты

Мерял на двух held-out наборах, каждый под свой вопрос.

- **test_real («научилась ли чинить»).** Посимвольная близость к эталону: сырой вход
  0.468 → модель 0.522, exact-match 0.083. Прирост реальный. Чистый umt5-small без
  дообучения даёт 0.016 и генерит мусор — весь навык приходит с файнтюна, а не из
  претрейна.
- **test_regression («сколько стоило»).** Вход == таргет: 78% чистых предложений
  остаются нетронутыми, ~22% модель зря правит. Это главная цена дообучения.
- **Что забылось.** Базовые неспецифичные навыки (QA, арифметика) после SFT фактически
  пропали. Модель умеет ровно одно — переписывать предложение исправленным. Для
  однозадачного файнтюна это норма.
- **Потолок.** umt5-base на тех же данных давал ~0.60 против 0.52, но вдвое тяжелее и
  втрое медленнее. С учётом дедлайна остановился на small.

Коротко: модель уверенно чинит казахские ошибки, платит за это умеренной склонностью
трогать чистый текст и потерей общих навыков, а качество упирается в выбор маленькой
модели под ограничения по железу и времени.

> В написании скриптов, создании датасета, а также в измерении и подсчёте метрик я
> использовал Claude Code.

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
