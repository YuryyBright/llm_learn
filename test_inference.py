import time
import json
from pathlib import Path
import numpy as np
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

CHECKPOINT_PATH = "./checkpoints/checkpoint-209"
ONNX_DIR = "./chunk-classifier-onnx"
QUANTIZED_DIR = "./chunk-classifier-onnx-int8"

# === НОВІ ТЕСТОВІ ДАНІ (більш реалістичні) ===
test_texts = [
    "Мені дуже подобається цей продукт, рекомендую!",
    "Підкажіть, будь ласка, як налаштувати інтеграцію?",
    "Купити пральну машину київ недорого",
    "Ось список необхідних речей: ручка, папір, зошит.",
    "Це просто жах, нічого не працює.",
    "Завтра о 10:00 зустріч з клієнтом, підготувати слайди.",
    "Я завжди віддаю перевагу темній темі в додатках.",
    "Фотосинтез — це процес перетворення світла на енергію в рослинах.",
    "Список покупок: молоко, хліб, яйця, сир, банани.",
    "Що таке нейропластичність і як її тренувати?"
] * 10   # 100 текстів

def benchmark_pipeline(pipe, data, name, iterations=3):
    print(f"\n--- Тестування: {name} ---")
    
    # Warmup
    _ = pipe(data[:5])
    
    start_time = time.time()
    for _ in range(iterations):
        results = pipe(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = len(data) * iterations
    ms_per_sample = (total_time / total_samples) * 1000
    fps = total_samples / total_time
    
    print(f"Час: {ms_per_sample:.2f} мс на один текст")
    print(f"Швидкість: {fps:.1f} текстів/сек")
    return results

print("Завантаження моделей...")

# 1. PyTorch модель (основна)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)

pt_model = ORTModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH, 
    local_files_only=True
)
pt_pipe = pipeline("text-classification", model=pt_model, tokenizer=tokenizer, device=-1)

# 2. ONNX модель (FP32)
onnx_model = ORTModelForSequenceClassification.from_pretrained(
    ONNX_DIR, 
    provider="CPUExecutionProvider",
    local_files_only=True
)
onnx_pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)

# 3. ONNX INT8 (квантована)
int8_model = ORTModelForSequenceClassification.from_pretrained(
    QUANTIZED_DIR, 
    provider="CPUExecutionProvider",
    file_name="model_quantized.onnx",
    local_files_only=True
)
int8_pipe = pipeline("text-classification", model=int8_model, tokenizer=tokenizer)

# Запуск бенчмарків
pt_res = benchmark_pipeline(pt_pipe, test_texts, "PyTorch (FP32)")
onnx_res = benchmark_pipeline(onnx_pipe, test_texts, "ONNX FP32")
int8_res = benchmark_pipeline(int8_pipe, test_texts, "ONNX INT8 (квантована)")

# Перевірка на розбіжності між PyTorch і INT8
print("\n--- Перевірка точності (PyTorch vs INT8) ---")
mismatches = 0
for i in range(len(test_texts)):
    if pt_res[i]['label'] != int8_res[i]['label']:
        mismatches += 1
        if mismatches <= 5:
            print(f"Розбіжність #{mismatches}: PyTorch={pt_res[i]['label']}, INT8={int8_res[i]['label']}")
            print(f"Текст: {test_texts[i][:80]}...\n")

if mismatches == 0:
    print("✅ Відмінно! Квантована INT8 модель дає 100% ідентичні результати.")
else:
    print(f"⚠️ Знайдено {mismatches} розбіжностей з {len(test_texts)} текстів.")

print("\nГотово!")