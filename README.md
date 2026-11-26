# Лабораторная работа 1. YOLO (Ultralytics, PyTorch) и оптимизация распознавания видеопотока

Проект из двух частей:

| Часть | Что внутри | Как запускать |
|---|---|---|
| **Python** — `lab1_yolo.ipynb` | все пункты задания: YOLO на изображениях, распознавание видео, сжатие кадра внутри модели, квантование, сравнение ONNX Runtime / TensorRT / OpenCV DNN / OpenVINO / CoreML | один ноутбук, **Run All** |
| **C++** — `lab1_cpp.ipynb` + `cpp/` | то же распознавание видео на C++ (OpenCV DNN и ONNX Runtime: CPU / CUDA / TensorRT / CoreML); ноутбук собирает программу, запускает её и сравнивает с Python | после Python-ноутбука, **Run All**; подробности — [`cpp/README.md`](cpp/README.md) |
| Теория — `THEORY.md` | анализ и сравнение способов оптимизации, результаты замеров, выводы | читать |
| **Отчёт — [`report_lab1.pdf`](report_lab1.pdf)** | отчёт по лабораторной работе (титул, реферат, разделы, выводы, листинги) | читать |

```
├── lab1_yolo.ipynb      Python-часть целиком
├── lab1_cpp.ipynb       C++-часть: сборка, запуск, сравнение с Python
├── requirements.txt     пакеты для ноутбука
├── THEORY.md            теоретическая часть
├── report_lab1.pdf      отчёт по лабораторной работе
├── cpp/                 C++-часть: main.cpp, CMakeLists.txt, README.md
└── models/ data/ runs/  создаются ноутбуком: модели, тестовое видео, результаты
```

Оба ноутбука в репозитории сохранены **с результатами выполнения** (пример прогона на 2-ядерном Intel Xeon без видеокарты),
поэтому таблицы и графики видны прямо на GitHub. При своём запуске они заменятся результатами вашего компьютера.

Модель по умолчанию — **YOLO26n** (самая новая лёгкая модель Ultralytics); меняется в ячейке параметров
(`MODEL = "yolo11n.pt"` и т.п.).

## Запуск Python-части

**1. Python 3.10–3.13 и виртуальное окружение**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # если ругается на политику: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. PyTorch под своё железо**

| Железо | Команда |
|---|---|
| NVIDIA GPU | команда с [pytorch.org/get-started](https://pytorch.org/get-started/locally/) для вашей CUDA, например `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132` |
| Mac на Apple Silicon | `pip install torch torchvision` (ускорение через MPS) |
| AMD Radeon, Linux | сборка ROCm, например `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14` |
| Только CPU / AMD Radeon на Windows | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

**3. Jupyter и ускорители (по желанию)**
```bash
pip install jupyterlab
```
Ноутбук сам найдёт установленные ускорители и добавит их в сравнение:

| Железо | Что поставить |
|---|---|
| NVIDIA | `pip install onnxruntime-gpu tensorrt` (до этого `pip uninstall -y onnxruntime`) |
| Intel / AMD процессор | `pip install openvino` |
| Apple | `pip install coremltools` |
| AMD Radeon на Windows | `pip install onnxruntime-directml` (до этого `pip uninstall -y onnxruntime`) |

**4. Запуск**
```bash
jupyter lab lab1_yolo.ipynb
```
(или открыть в VS Code / PyCharm, выбрав интерпретатор из `.venv`) → **Run All**.

Первая ячейка доставит пакеты из `requirements.txt`. При первом запуске нужен интернет: скачаются веса YOLO (~5 МБ)
и датасет COCO128 (~7 МБ) для mAP. Без видеокарты весь ноутбук выполняется 10–15 минут.

Параметры — во второй ячейке: своё видео (`VIDEO = "my.mp4"`, `"0"` — веб-камера), устройство,
число кадров, степени сжатия, `VAL_DATA = None` — не считать mAP (быстрее, без интернета).

**Что получится:**

| Раздел ноутбука | Результат |
|---|---|
| 1. YOLO на изображениях | рамки на картинках, `runs/01_images/` |
| 2. Видео | `runs/02_video/*.mp4`, время по этапам, FPS; сравнение 640 / 320 / пропуск кадров |
| 3. Сжатие внутри модели | изменённая модель: кадр фиксированного размера сжимается внутри графа через `F.interpolate` или слой свёртки (работает на GPU); проверка экспорта в ONNX; модели `models/*_compressed_*.onnx`, сравнение скорости и качества |
| 4. Квантование | `models/yolo26n.onnx`, `_fp16.onnx`, `_int8.onnx` и др.; размер, задержка, mAP |
| 5. Сравнение бэкендов | таблица и график по всем доступным ускорителям |

Все таблицы сохраняются в `runs/*/results.md` — их удобно вставить в отчёт (`THEORY.md`, раздел 11).

## Запуск C++-части

Ноутбук **`lab1_cpp.ipynb`** — запускать после `lab1_yolo.ipynb` (он создаёт модели в `models/` и видео).
Нужны CMake ≥ 3.18, компилятор C++17, OpenCV ≥ 4.10 и (по желанию) ONNX Runtime C++. Пути задаются в первой
ячейке или переменными окружения до запуска Jupyter:

```bash
export OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4        # Windows: set OpenCV_DIR=C:/opencv/build
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.32.0
jupyter lab lab1_cpp.ipynb                                  # → Run All
```

Ноутбук собирает `cpp/yolo_video` через CMake, запускает его на FP32-, INT8- и «сжатых» моделях (и на GPU, если
есть NVIDIA), сравнивает с Python по этапам конвейера и показывает кадры размеченного видео. Сборка и запуск
вручную из терминала — [`cpp/README.md`](cpp/README.md).

## Частые проблемы

| Симптом | Решение |
|---|---|
| `CUDA: False`, хотя есть NVIDIA | стоит CPU-сборка PyTorch — переустановить командой с pytorch.org |
| ONNX Runtime не видит CUDA | стоят одновременно `onnxruntime` и `onnxruntime-gpu`: удалить оба, поставить только `onnxruntime-gpu` |
| Нет интернета | положите `yolo26n.pt` в `models/` вручную, `VAL_DATA = None` |
| OpenCV DNN CUDA в таблице нет | pip-пакеты OpenCV собраны без CUDA; нужна сборка 4.x из исходников с CUDA (подробнее — `THEORY.md`, раздел 5) |
| Ноутбук запускается не из своей папки | запускайте Jupyter из корня репозитория: пути `models/`, `data/`, `runs/` считаются от текущей папки |
