from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
import os

# Шлях до вашої найкращої моделі PyTorch (після train.py)
CHECKPOINT_PATH = "./checkpoints/checkpoint-209" # Замініть на свій
ONNX_DIR = "./chunk-classifier-onnx"
QUANTIZED_DIR = "./chunk-classifier-onnx-int8"

print("1. Завантаження та експорт моделі в ONNX...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = ORTModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH,
    export=True,
    provider="CPUExecutionProvider",
)

# Зберігаємо базову ONNX модель
model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
print(f"Базову ONNX модель збережено у {ONNX_DIR}")

print("\n2. Квантування моделі до INT8...")
# Створюємо квантизатор на основі базової ONNX моделі
quantizer = ORTQuantizer.from_pretrained(model)

# Налаштовуємо динамічне квантування (найкраще для NLP на CPU)
dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

# Застосовуємо квантування і зберігаємо
quantizer.quantize(
    save_dir=QUANTIZED_DIR,
    quantization_config=dqconfig,
)
tokenizer.save_pretrained(QUANTIZED_DIR)

# Розміри файлів для порівняння
pt_size = os.path.getsize(os.path.join(CHECKPOINT_PATH, "model.safetensors")) / (1024*1024)
onnx_size = os.path.getsize(os.path.join(ONNX_DIR, "model.onnx")) / (1024*1024)
int8_size = os.path.getsize(os.path.join(QUANTIZED_DIR, "model_quantized.onnx")) / (1024*1024)

print(f"\nКвантування завершено! Розміри моделей:")
print(f"PyTorch: ~{pt_size:.1f} MB")
print(f"ONNX: ~{onnx_size:.1f} MB")
print(f"ONNX INT8: ~{int8_size:.1f} MB")