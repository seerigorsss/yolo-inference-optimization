# Лабораторная работа 1. YOLO (Ultralytics, PyTorch) и оптимизация распознавания видеопотока

| Пункт задания | Где сделано |
|---|---|
| Установить Ultralytics и запустить обученную YOLO | `01_detect_image.py` |
| Распознавание видеофайла (Python) | `02_detect_video.py` |
| Сжатие изображения внутри модели | `03_compress_in_model.py` |
| Квантование модели | `04_quantize.py` |
| Анализ и сравнение ONNX / TensorRT / OpenCV CUDA (теория) | `THEORY.md` + замеры `05_benchmark.py` |
| Реализация на C++ (доп. баллы) | `cpp/` — OpenCV DNN и ONNX Runtime (CPU / CUDA / TensorRT / CoreML) |

Модель по умолчанию — **YOLO26n** (самая новая лёгкая модель Ultralytics). Любой скрипт принимает
`--model yolo11n.pt` (или `yolo26s.pt` и т.д.), если преподаватель просит другую.

```
yolo_lab1/
├── 00_check_env.py         проверка окружения и ускорителей
├── 01_detect_image.py      шаг 1: YOLO на картинках
├── make_demo_video.py      тестовое видео data/demo.mp4 (если нет своего)
├── 02_detect_video.py      шаг 2: распознавание видеофайла
├── 03_compress_in_model.py шаг 3: сжатие кадра внутри модели
├── 04_quantize.py          шаг 4: FP16 / INT8-квантование
├── 05_benchmark.py         шаг 5: сравнение бэкендов
├── common.py               общие функции (pre/post-обработка, NMS, отрисовка)
├── cpp/                    C++: main.cpp + CMakeLists.txt
├── THEORY.md               теоретическая часть
├── models/  data/  runs/   создаются автоматически: модели, видео, результаты
```

---

## Шаг 0. Установка

**Python 3.10–3.13.** Проверить: `python --version` (в Linux/macOS может называться `python3`).

### 0.1. Виртуальное окружение

Windows (PowerShell):
```powershell
cd yolo_lab1
python -m venv .venv
.venv\Scripts\Activate.ps1        # если ругается на политику: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Linux / macOS:
```bash
cd yolo_lab1
python3 -m venv .venv
source .venv/bin/activate
```
Дальше все команды — в активированном окружении (в начале строки терминала видно `(.venv)`).

### 0.2. PyTorch — под своё железо

| Железо | Команда |
|---|---|
| NVIDIA GPU (Windows/Linux) | команда с сайта [pytorch.org/get-started](https://pytorch.org/get-started/locally/) для вашей версии CUDA, например `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132` |
| Mac на Apple Silicon (M1–M5) | `pip install torch torchvision` (ускорение через MPS включено) |
| AMD Radeon, Linux | сборка ROCm с pytorch.org, например `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14` |
| Только CPU / AMD Radeon на Windows | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

### 0.3. Остальные пакеты

```bash
pip install -r requirements.txt
```
Ускорители (по желанию — то, что подходит к вашему компьютеру):

| Железо | Что поставить | Что даёт |
|---|---|---|
| NVIDIA | `pip uninstall -y onnxruntime && pip install onnxruntime-gpu tensorrt` | ONNX Runtime CUDA/TensorRT EP, TensorRT-движки |
| Intel / AMD процессор | `pip install openvino` | OpenVINO FP32/INT8 (часто самый быстрый вариант на CPU) |
| Apple | `pip install coremltools` | экспорт в CoreML (Neural Engine); ONNX Runtime CoreML EP уже есть в обычном `onnxruntime` для macOS |
| AMD Radeon на Windows | `pip uninstall -y onnxruntime && pip install onnxruntime-directml` | ONNX Runtime DirectML EP (GPU через DirectX 12) |

### 0.4. Проверка
```bash
python 00_check_env.py
```
Покажет версии и что доступно: CUDA / MPS / ROCm, провайдеры ONNX Runtime, CUDA в OpenCV.

---

## Шаг 1. Запуск обученной YOLO
```bash
python 01_detect_image.py
```
- При первом запуске Ultralytics сам скачает веса `models/yolo26n.pt` (~5 МБ) — нужен интернет.
- В консоли: найденные объекты с уверенностью и координатами, время pre/inference/post.
- Картинки с рамками: `runs/01_images/` (bus.jpg — автобус и 4 человека, zidane.jpg — 2 человека и галстук).
- Своё фото: `python 01_detect_image.py --source путь/к/фото.jpg`

## Шаг 2. Распознавание видеофайла
Нужно видео. Либо своё (`--source my.mp4`, `--source 0` — веб-камера), либо тестовое:
```bash
python make_demo_video.py          # создаст data/demo.mp4 (1280x720, 12 с)
python 02_detect_video.py          # обработает data/demo.mp4
python 02_detect_video.py --source my.mp4 --show    # своё видео с окном просмотра (q — выход)
```
Результат: `runs/02_video/demo_yolo26n.mp4` (видео с рамками) и `.json` со статистикой.
В консоли — время по этапам (чтение, препроцессинг, инференс, NMS, отрисовка+запись) и два FPS:
модели и всего конвейера.

Полезные варианты для отчёта:
```bash
python 02_detect_video.py --imgsz 320                 # меньше вход — быстрее, но хуже мелкие объекты
python 02_detect_video.py --vid-stride 2              # пропуск кадров
python 02_detect_video.py --device 0 --fp16           # NVIDIA GPU в FP16
python 02_detect_video.py --model models/yolo26n.onnx # ONNX Runtime (файл появится после шага 4)
```

## Шаг 3. Сжатие изображения внутри модели
```bash
python 03_compress_in_model.py
```
Что происходит: модель оборачивается в `nn.Module`, который принимает **сырой кадр** `uint8 [1,H,W,3]` BGR
и сам внутри графа делает BGR→RGB, **сжатие** (AvgPool k×k + bilinear resize), поля до кратного 32,
нормировку, запускает YOLO и пересчитывает рамки в координаты исходного кадра. Это экспортируется в ONNX
(`models/yolo26n_compressed_1280x720_to_320.onnx` и т.п.) и прогоняется через ONNX Runtime —
в Python остаётся только NMS.

Сравниваются степени сжатия `0` (без сжатия), `640`, `480`, `320`, `256`:
время, FPS, число объектов, доля объектов, найденных эталоном (640 — родное разрешение YOLO),
и **mAP на COCO128** при той же степени сжатия (датасет ~7 МБ скачается сам).
Таблица: `runs/03_compress/results.md`, видео для каждой степени: `runs/03_compress/*.mp4`.

Модель экспортируется под разрешение конкретного видео. Своё видео: `--source my.mp4`
(для 1920x1080 будут созданы новые файлы `..._1920x1080_to_...onnx`).

## Шаг 4. Квантование
```bash
python 04_quantize.py
```
Создаёт в `models/`:

| Файл | Что это |
|---|---|
| `yolo26n.onnx` | FP32 — исходная точка |
| `yolo26n_fp16.onnx` | FP16-веса (вход/выход float32) |
| `yolo26n_int8.onnx` | **основной результат**: статическое INT8 (QDQ), калибровка по кадрам видео, декодер головы во float |
| `yolo26n_int8_convs.onnx`, `_int8_wholegraph.onnx`, `_int8_dynamic.onnx` | демонстрационные варианты для анализа (что будет, если квантовать иначе) |

Таблица `runs/04_quantize/results.md`: размер, задержка на CPU, ускорение, mAP50/mAP50-95 на COCO128
и совпадение детекций с FP32 на видео. Занимает 3–6 минут.

Опции:
```bash
python 04_quantize.py --calib my.mp4 --calib-n 200 --method entropy   # своя калибровка
python 04_quantize.py --extra openvino      # + OpenVINO FP32/INT8 (Intel/AMD CPU)
python 04_quantize.py --extra engine        # + TensorRT FP16/INT8 (NVIDIA; сборка движка — несколько минут)
python 04_quantize.py --extra coreml        # + CoreML FP16/INT8 (macOS)
```

## Шаг 5. Сравнение способов ускорения
```bash
python 05_benchmark.py
```
Сам находит всё, что есть на компьютере (PyTorch CPU/CUDA/MPS, ONNX Runtime с разными EP, OpenCV DNN,
TensorRT, OpenVINO, CoreML), при необходимости экспортирует модели и на одних и тех же кадрах меряет
инференс, p95, FPS и совпадение детекций с эталоном. Таблица: `runs/05_benchmark/results.md` —
её удобно вставить в отчёт к теоретической части (`THEORY.md`, раздел 6).

```bash
python 05_benchmark.py --only ort opencv     # только выбранные группы
python 05_benchmark.py --no-export           # не экспортировать (TensorRT-движок строится минуты)
```

---

## Шаг 6 (доп. баллы). C++

Программа `cpp/main.cpp` — то же распознавание видео на C++: чтение кадров, letterbox, инференс,
разбор выхода YOLO, NMS, отрисовка, запись видео и замеры по этапам.

| Бэкенд | Устройства | Нужно |
|---|---|---|
| `--backend opencv` | `cpu`, `cuda`, `cuda-fp16` | OpenCV **≥ 4.10** (C++). CUDA — только OpenCV 4.x, собранный с CUDA; в 5.x новый DNN-движок пока только CPU |
| `--backend ort` | `cpu`, `cuda`, `trt` (TensorRT), `coreml`, `openvino` | + ONNX Runtime C++ (собирать с `-DWITH_ORT=ON`). Понимает INT8-модель шага 4 и модель «со сжатием внутри» шага 3. Для `cuda`/`trt` — GPU-сборка ONNX Runtime, CUDA/cuDNN (и TensorRT) в `PATH` |

Проверено: Linux, OpenCV 4.14 и ONNX Runtime 1.32 (CPU). Код рассчитан и на OpenCV 5.x, но с ней не собирался.

### 6.1. Зависимости
- **CMake ≥ 3.18** и компилятор C++17 (Windows — Visual Studio 2022 с «Разработкой классических приложений на C++»;
  macOS — `xcode-select --install`; Linux — `sudo apt install build-essential cmake`).
- **OpenCV ≥ 4.10**:
  - Windows: скачать релиз с [opencv.org/releases](https://opencv.org/releases/), распаковать в `C:\opencv`,
    добавить папку `C:\opencv\build\x64\vc16\bin` (или `vc17` — зависит от версии) в `PATH`;
  - macOS: `brew install opencv`;
  - Linux: `pkg-config --modversion opencv4`. В Ubuntu 24.04 из apt ставится 4.6 — она **не открывает**
    YOLO11/YOLO26 (ошибка в узле Split). Нужна сборка OpenCV ≥ 4.10 из исходников либо `--backend ort`.
- **ONNX Runtime** (для `--backend ort`): архив с
  [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
  (`onnxruntime-win-x64-*.zip`, `onnxruntime-osx-arm64-*.tgz`, `onnxruntime-linux-x64-*.tgz`;
  для NVIDIA — вариант `-gpu-`), распаковать, путь передать в `ONNXRUNTIME_DIR`.

### 6.2. Сборка
Linux / macOS:
```bash
cd cpp
cmake -S . -B build -DWITH_ORT=ON -DONNXRUNTIME_DIR=$HOME/onnxruntime-linux-x64-1.32.0
cmake --build build -j
```
Windows (PowerShell, «x64 Native Tools» или обычный терминал с установленной VS):
```powershell
cd cpp
cmake -S . -B build -DOpenCV_DIR=C:/opencv/build -DWITH_ORT=ON -DONNXRUNTIME_DIR=C:/onnxruntime-win-x64-1.32.0
cmake --build build --config Release
# исполняемый файл: build\Release\yolo_video.exe
```
Без ONNX Runtime — просто не указывать `-DWITH_ORT=ON ...`.

### 6.3. Запуск
Модели — из шагов 3–4 (папка `models/`), видео — `data/demo.mp4` или своё.
```bash
cd cpp
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4                         # OpenCV DNN, CPU
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4 --device cuda-fp16      # OpenCV 4.x + CUDA
./build/yolo_video ../models/yolo26n.onnx ../data/demo.mp4 --backend ort           # ONNX Runtime, CPU
./build/yolo_video ../models/yolo26n_int8.onnx ../data/demo.mp4 --backend ort      # INT8 из шага 4
./build/yolo_video ../models/yolo26n_compressed_1280x720_to_320.onnx ../data/demo.mp4 --backend ort   # шаг 3
./build/yolo_video ../models/yolo26n_fp16.onnx ../data/demo.mp4 --backend ort --device trt           # TensorRT
./build/yolo_video ../models/yolo26n.onnx 0 --show                                  # веб-камера
```
На Windows: `build\Release\yolo_video.exe ..\models\yolo26n.onnx ..\data\demo.mp4`.
Результат — `out_cpp.mp4` и таблица времени по этапам (как в Python, шаг 2 — удобно сравнить).

---

## Что показать на сдаче
1. `runs/01_images/` — рамки на картинках.
2. `runs/02_video/*.mp4` + разбивка времени по этапам из консоли.
3. `runs/03_compress/results.md` — скорость и точность при разных степенях сжатия, объяснение, что препроцессинг
   «живёт» в графе модели (схема в `THEORY.md`, раздел 5).
4. `runs/04_quantize/results.md` — FP32 / FP16 / INT8: размер ×3 меньше, ускорение на CPU, падение mAP ~1–3 пункта;
   почему «весь граф» в INT8 даёт mAP = 0.
5. `runs/05_benchmark/results.md` + `THEORY.md` — сравнение ONNX Runtime / TensorRT / OpenCV DNN (CUDA) и аналогов.
6. C++: вывод `yolo_video` для OpenCV DNN и ONNX Runtime.

## Частые проблемы
| Симптом | Решение |
|---|---|
| `Не найдено тестовое видео data/demo.mp4` | `python make_demo_video.py` или `--source my.mp4` |
| Нет интернета при первом запуске | веса `yolo26n.pt` и COCO128 скачиваются автоматически; положите `yolo26n.pt` в `models/` вручную или запускайте шаги 3–4 с `--val-data none` |
| `CUDA: False`, хотя есть NVIDIA | стоит CPU-сборка PyTorch: переустановить командой с pytorch.org (шаг 0.2) |
| ONNX Runtime не видит CUDA | одновременно стоят `onnxruntime` и `onnxruntime-gpu`: удалить оба, поставить только `onnxruntime-gpu` |
| OpenCV DNN: `cuda` молча работает на CPU | pip-пакеты OpenCV собраны без CUDA; нужна сборка 4.x из исходников с `WITH_CUDA=ON, WITH_CUDNN=ON, OPENCV_DNN_CUDA=ON` |
| C++: `ERROR during processing node ... [Split]` | старая OpenCV (4.6 из apt) — нужна ≥ 4.10 или `--backend ort` |
| C++: `libonnxruntime.so.1: cannot open shared object` | путь в `-DONNXRUNTIME_DIR` должен указывать на распакованный архив (с папками `include/` и `lib/`) |
| Шаг 3: `модель экспортирована для кадра ...` | модель «со сжатием внутри» привязана к разрешению видео — запустите шаг 3 с `--source` этого видео |
