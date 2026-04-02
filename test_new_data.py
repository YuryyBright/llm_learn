from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

# === НАЛАШТУВАННЯ ===
model_path = "./chunk-classifier-onnx"   # або "./chunk-classifier-onnx"

print(f"Завантажуємо ONNX модель з: {model_path}")

# Завантажуємо токенізатор і ONNX модель
tokenizer = AutoTokenizer.from_pretrained(model_path)
ort_model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    provider="CPUExecutionProvider",   # або "CUDAExecutionProvider" якщо є GPU
    file_name="model_quantized.onnx" if "int8" in model_path else "model.onnx"
)

# Створюємо pipeline
pipe = pipeline(
    "text-classification",
    model=ort_model,
    tokenizer=tokenizer,
    device=-1   # CPU
)

# Завантажуємо нові тестові дані
data = []
for line in Path("new_test_data.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        data.append(json.loads(line))

texts = [item["text"] for item in data]
true_labels = [item["label"] for item in data]

# Передбачення
print("Запуск передбачення на нових даних...")
results = pipe(texts, truncation=True, max_length=128)
pred_labels = [r["label"] for r in results]

# Результати
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Точність на нових даних: {accuracy:.4f} ({accuracy*100:.1f}%)\n")

print("Детальний звіт за класами:")
print(classification_report(true_labels, pred_labels, 
      target_names=["noise", "list", "preference", "knowledge"], digits=4))

# Показ помилок
print("\n❌ Приклади, де модель помилилася:")
errors = []
for text, true, pred in zip(texts, true_labels, pred_labels):
    if true != pred:
        errors.append((text, true, pred))
        print(f"Правда: {true:12} | Модель: {pred:12} | {text[:110]}...")

if not errors:
    print("Модель правильно класифікувала всі нові приклади!")