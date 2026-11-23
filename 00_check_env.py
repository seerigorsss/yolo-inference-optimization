"""Шаг 0. Проверка окружения: версии библиотек и доступные ускорители."""

import platform
import sys


def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        return f"не установлен ({type(e).__name__})"


print(f"Python      {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
for mod in ["torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "cv2", "numpy", "openvino", "tensorrt",
            "coremltools"]:
    print(f"{mod:<12}{ver(mod)}")

print("\nУскорители:")
try:
    import torch

    print(f"  PyTorch CUDA : {torch.cuda.is_available()}"
          + (f"  -> {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}" if torch.cuda.is_available() else ""))
    print(f"  PyTorch MPS  : {torch.backends.mps.is_available()}  (Apple Silicon)")
    print(f"  PyTorch ROCm : {getattr(torch.version, 'hip', None) is not None}  (AMD Radeon, Linux)")
except ImportError:
    pass
try:
    import onnxruntime as ort

    print(f"  ONNX Runtime : {ort.get_available_providers()}")
except ImportError:
    pass
try:
    import cv2

    n = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
    print(f"  OpenCV CUDA  : {n > 0}  (устройств: {n}; pip-сборка OpenCV собрана без CUDA — это нормально)")
except ImportError:
    pass
