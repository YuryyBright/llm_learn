# train.py
# Пояснення кожного кроку нижче

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# ─── 1. ЗАВАНТАЖЕННЯ ДАТАСЕТУ ─────────────────────────────────────────────────
LABEL2ID = {"noise": 0, "list": 1, "preference": 2, "knowledge": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Ваші 3 файли з даними
files = [
    "dataset_knowledge.jsonl", 
    "dataset_noise_list.jsonl", 
    "dataset_preference.jsonl"
]

all_rows = []
for file_path in files:
    # Читаємо кожен файл, обов'язково з utf-8 для кирилиці
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    all_rows.extend([json.loads(l) for l in lines if l])

texts  = [r["text"] for r in all_rows]
labels = [LABEL2ID[r["label"]] for r in all_rows]

# train/val/test split: 80% / 10% / 10%
X_train, X_tmp, y_train, y_tmp = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
X_val,   X_test, y_val,  y_test = train_test_split(X_tmp,  y_tmp,  test_size=0.5, random_state=42, stratify=y_tmp)

# ─── 2. ТОКЕНІЗАТОР ───────────────────────────────────────────────────────────
# Токенізатор розбиває текст на токени (підслова), додає [CLS] та [SEP],
# повертає input_ids + attention_mask (маска "де справжній текст, а де padding")
MODEL_NAME = "youscan/ukr-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",  # всі рядки однакової довжини
        truncation=True,       # обрізати якщо довше 512 токенів
        max_length=128,        # для речень 128 достатньо, швидше навчання
    )

# Перетворюємо у HuggingFace Dataset — зручний формат для Trainer
train_ds = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize, batched=True)
val_ds   = Dataset.from_dict({"text": X_val,   "label": y_val  }).map(tokenize, batched=True)
test_ds  = Dataset.from_dict({"text": X_test,  "label": y_test }).map(tokenize, batched=True)

# ─── 3. МОДЕЛЬ ────────────────────────────────────────────────────────────────
# AutoModelForSequenceClassification = тіло RoBERTa + лінійний шар (768 → 4)
# Він автоматично додає classification head поверх [CLS] токена
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ─── 4. МЕТРИКИ ───────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)  # logits → клас з найвищою вірогідністю
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),  # macro = однакова вага кожного класу
    }

# ─── 5. ГІПЕРПАРАМЕТРИ ────────────────────────────────────────────────────────
# lr=2e-5 — стандарт для fine-tuning BERT: великий lr "ламає" pretrained ваги
# weight_decay — регуляризація, запобігає перенавчанню
# warmup_steps — перші N кроків lr повільно зростає (стабільніше навчання)
args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",   # перевіряємо якість після кожної епохи
    save_strategy="epoch",
    load_best_model_at_end=True,   # залишити найкращу модель за F1
    metric_for_best_model="f1",
    logging_steps=10,
    fp16=torch.cuda.is_available(), # half-precision: вдвічі швидше на GPU
)

# ─── 6. НАВЧАННЯ ──────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

trainer.train()

# ─── 7. ФІНАЛЬНА ОЦІНКА ───────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

results = trainer.evaluate(test_ds)
print(f"Test Accuracy: {results['eval_accuracy']:.3f}")
print(f"Test F1 (macro): {results['eval_f1']:.3f}")

# Детальний звіт по кожному класу окремо
predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\n── Classification Report ──────────────────────")
print(classification_report(
    true_labels,
    preds,
    target_names=list(LABEL2ID.keys()),  # ["noise", "list", "preference", "knowledge"]
))
# Виведе для кожного класу:
# precision — скільки з передбачених цього класу справді цей клас
# recall    — скільки реальних прикладів класу модель знайшла
# f1-score  — середнє між precision і recall
# support   — скільки прикладів цього класу в тесті

# Матриця помилок — бачимо що з чим плутається
print("\n── Confusion Matrix ───────────────────────────")
cm = confusion_matrix(true_labels, preds)
labels_names = list(LABEL2ID.keys())

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_names,
    yticklabels=labels_names,
    ax=ax,
)
ax.set_xlabel("Передбачений клас")
ax.set_ylabel("Реальний клас")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Збережено: confusion_matrix.png")

# Аналіз помилок — показуємо конкретні приклади де модель помилилась
print("\n── Приклади помилок ───────────────────────────")
errors = [
    (X_test[i], ID2LABEL[true_labels[i]], ID2LABEL[preds[i]])
    for i in range(len(X_test))
    if true_labels[i] != preds[i]
]
for text, true, pred in errors[:10]:  # перші 10 помилок
    print(f"  ПРАВДА: {true:12} | МОДЕЛЬ: {pred:12} | {text[:80]}...")