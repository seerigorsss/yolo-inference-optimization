# Часть на C++ (дополнительные баллы)

Проще всего — ноутбук [`lab1_cpp.ipynb`](../lab1_cpp.ipynb) в корне репозитория: он собирает программу,
запускает её на всех моделях и сравнивает с Python. Ниже — то же вручную из терминала.

`main.cpp` — распознавание объектов в видео на C++: чтение кадров, letterbox, инференс YOLO (ONNX),
разбор выхода сети, NMS, отрисовка, запись видео и замер времени по этапам (как в разделе 2 ноутбука —
удобно сравнить Python и C++).

| Бэкенд | Устройства (`--device`) | Что нужно |
|---|---|---|
| `--backend opencv` | `cpu`, `cuda`, `cuda-fp16` | OpenCV **≥ 4.10**. CUDA — только OpenCV 4.x, собранный с CUDA; в OpenCV 5.x новый DNN-движок пока только CPU |
| `--backend ort` | `cpu`, `cuda`, `trt` (TensorRT), `coreml`, `openvino` | + ONNX Runtime C++ (сборка с `-DWITH_ORT=ON`). Понимает INT8-модель из раздела 4 и модель «со сжатием внутри» из раздела 3. Для `cuda`/`trt` — GPU-сборка ONNX Runtime, CUDA/cuDNN (и TensorRT) в `PATH` |

Проверено: Linux, OpenCV 4.14, ONNX Runtime 1.32 (CPU). Код рассчитан и на OpenCV 5.x, но с ней не собирался.

## 1. Модели

Программа использует ONNX-модели из папки `models/` в корне репозитория. Их создаёт ноутбук `lab1_yolo.ipynb`
(разделы 3–4). Если нужна только C++-часть, базовую модель можно получить одной командой
(из корня репозитория, в окружении с `ultralytics`):

```bash
yolo export model=models/yolo26n.pt format=onnx      # скачает веса и создаст models/yolo26n.onnx
```

| Файл в `models/` | Откуда | Запуск |
|---|---|---|
| `yolo26n.onnx` | раздел 4 или команда выше | любой бэкенд |
| `yolo26n_fp16.onnx` | раздел 4 | `--backend ort` (для `--device trt` — FP16-движок) |
| `yolo26n_int8.onnx` | раздел 4 | `--backend ort` (CPU, TensorRT) |
| `yolo26n_compressed_interpolate_1280x720_to_320.onnx`, `yolo26n_compressed_conv_1280x720_to_320.onnx` | раздел 3 (сжатие кадра внутри модели: F.interpolate / слой свёртки) | `--backend ort`, видео того же разрешения |

Тестовое видео `data/demo.mp4` тоже создаёт ноутбук (раздел 2); подойдёт любое своё.

## 2. Зависимости

- **CMake ≥ 3.18** и компилятор C++17:
  Windows — Visual Studio 2022 с «Разработкой классических приложений на C++»;
  macOS — `xcode-select --install` и `brew install cmake`;
  Linux — `sudo apt install build-essential cmake`.
- **OpenCV ≥ 4.10**:
  - Windows: релиз с [opencv.org/releases](https://opencv.org/releases/), распаковать в `C:\opencv`,
    добавить `C:\opencv\build\x64\vc16\bin` (или `vc17` — зависит от версии) в `PATH`;
  - macOS: `brew install opencv`;
  - Linux: проверить `pkg-config --modversion opencv4`. В Ubuntu 24.04 из apt ставится 4.6 — она **не открывает**
    YOLO11/YOLO26 (ошибка в узле `Split`); нужна сборка OpenCV ≥ 4.10 из исходников или `--backend ort`.
- **ONNX Runtime** (для `--backend ort`): архив с
  [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases) —
  `onnxruntime-win-x64-*.zip`, `onnxruntime-osx-arm64-*.tgz`, `onnxruntime-linux-x64-*.tgz`
  (для NVIDIA — вариант с `-gpu-`). Распаковать; путь передаётся в `ONNXRUNTIME_DIR`.

## 3. Сборка

Linux / macOS (из папки `cpp/`):
```bash
cmake -S . -B build -DWITH_ORT=ON -DONNXRUNTIME_DIR=$HOME/onnxruntime-linux-x64-1.32.0
cmake --build build -j
```
Windows (PowerShell, из папки `cpp\`):
```powershell
cmake -S . -B build -DOpenCV_DIR=C:/opencv/build -DWITH_ORT=ON -DONNXRUNTIME_DIR=C:/onnxruntime-win-x64-1.32.0
cmake --build build --config Release
```
Без ONNX Runtime — просто не указывать `-DWITH_ORT=ON ...` (останется бэкенд OpenCV).

## 4. Запуск

```bash
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4                         # OpenCV DNN, CPU
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4 --device cuda-fp16      # OpenCV 4.x + CUDA
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4 --backend ort           # ONNX Runtime, CPU
./build/yolo_video ../models/yolo26n_int8.onnx ../data/demo.mp4 --backend ort      # INT8 из раздела 4
./build/yolo_video ../models/yolo26n_compressed_conv_1280x720_to_320.onnx ../data/demo.mp4 --backend ort  # раздел 3
./build/yolo_video ../models/yolo26n_fp16.onnx ../data/demo.mp4 --backend ort --device trt           # TensorRT
./build/yolo_video ../models/yolo26n.onnx 0 --show                                  # веб-камера, окно (q — выход)
```
Windows: `build\Release\yolo_video.exe ..\models\yolo26n.onnx ..\data\demo.mp4`.

Опции: `--imgsz 640`, `--conf 0.25`, `--iou 0.45`, `--max-frames N`, `--out out.mp4` (`--out ""` — не сохранять).
Результат — `out_cpp.mp4` и время по этапам:

```
Бэкенд:  ONNX Runtime 1.32.0 cpu
Кадров: 300
Среднее время на кадр, мс:
  read         1.11
  preprocess   1.92
  inference    72.15
  postprocess  0.67
  draw+write   8.21
FPS модели (pre+inference+post): 13.4
FPS всего конвейера: 11.9
```

## Частые проблемы

| Симптом | Решение |
|---|---|
| `ERROR during processing node ... [Split]` | старая OpenCV (4.6 из apt) — нужна ≥ 4.10 или `--backend ort` |
| `[!] OpenCV собран без CUDA` | pip/apt/brew-сборки OpenCV без CUDA; нужна сборка 4.x с `WITH_CUDA=ON, WITH_CUDNN=ON, OPENCV_DNN_CUDA=ON` |
| `libonnxruntime.so.1: cannot open shared object` | `-DONNXRUNTIME_DIR` должен указывать на распакованный архив (папки `include/` и `lib/`) |
| `TensorRT execution provider is not enabled in this build` | нужен GPU-архив ONNX Runtime (`-gpu-`), CUDA и TensorRT |
| `модель экспортирована для кадра WxH` | модель «со сжатием внутри» привязана к разрешению видео — создайте её в разделе 3 для этого видео |
