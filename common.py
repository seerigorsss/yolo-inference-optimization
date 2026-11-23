"""Общие функции для всех шагов лабораторной работы.

Здесь собрано то, что нужно сразу нескольким скриптам:
  * выбор устройства (CUDA / Apple MPS / CPU);
  * letterbox-препроцессинг кадра под вход сети;
  * постобработка выхода YOLO (порог уверенности + NMS);
  * отрисовка рамок, сравнение детекций, статистика времени;
  * открытие видео и запись результата.

Код pre/post-обработки написан на NumPy + OpenCV и не зависит от Ultralytics,
поэтому одинаково работает с PyTorch, ONNX Runtime, OpenCV DNN и т.д.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"  # веса и экспортированные модели
RUNS = ROOT / "runs"  # результаты запусков
DATA = ROOT / "data"  # тестовое видео

MODELS.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)

DEFAULT_MODEL = "yolo26n.pt"  # самая свежая и лёгкая модель Ultralytics (2026); можно yolo11n.pt

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# --------------------------------------------------------------------------- устройства
def pick_device(pref: str = "auto") -> str:
    """Вернуть устройство в формате Ultralytics: '0' (CUDA), 'mps' (Apple Silicon) или 'cpu'."""
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ort_providers(prefer_gpu: bool = True) -> list[str]:
    """Список Execution Provider'ов ONNX Runtime в порядке приоритета (что есть в системе)."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"] if prefer_gpu else []
    return [p for p in order if p in available] + ["CPUExecutionProvider"]


def make_ort_session(path, providers=None):
    """Сессия ONNX Runtime с максимальной оптимизацией графа; если ускоритель не запустился — откат на CPU."""
    import onnxruntime as ort

    providers = providers or ort_providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    if "DmlExecutionProvider" in names:  # требование DirectML EP
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(str(path), so, providers=providers)
    except Exception as e:  # noqa: BLE001
        if names == ["CPUExecutionProvider"]:
            raise
        print(f"[!] {names[0]} не запустился ({type(e).__name__}), использую CPU")
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def model_path(name: str) -> str:
    """Путь к весам: файлы *.pt по умолчанию лежат в models/ (Ultralytics скачает их туда сам)."""
    p = Path(name)
    if p.exists() or p.parent != Path("."):
        return str(p)
    return str(MODELS / p.name)


def onnx_names(session) -> list[str]:
    """Имена классов из метаданных ONNX, которые записывает Ultralytics при экспорте."""
    meta = session.get_modelmeta().custom_metadata_map
    if "names" in meta:
        names = ast.literal_eval(meta["names"])
        return [names[i] for i in sorted(names)]
    return COCO_NAMES


# --------------------------------------------------------------------------- препроцессинг
def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Масштабирует кадр с сохранением пропорций и дополняет серыми полями до new_shape (h, w).

    Возвращает (картинка, коэффициент масштаба r, сдвиг (pad_x, pad_y)).
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = round(h * r), round(w * r)
    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    img = cv2.copyMakeBorder(
        img, top, new_shape[0] - nh - top, left, new_shape[1] - nw - left, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (left, top)


def preprocess(frame: np.ndarray, imgsz: int = 640):
    """BGR-кадр -> тензор float32 [1,3,imgsz,imgsz] в RGB, значения 0..1."""
    img, r, pad = letterbox(frame, (imgsz, imgsz))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)  # HWC BGR -> NCHW RGB
    return blob, r, pad


# --------------------------------------------------------------------------- постобработка
def postprocess(out, conf: float = 0.25, iou: float = 0.45, r: float = 1.0, pad=(0, 0), orig_shape=None,
                max_det: int = 300):
    """Выход YOLO -> (boxes xyxy [K,4], scores [K], classes [K]) в координатах исходного кадра.

    Поддерживаются два формата выхода:
      * (1, 4+nc, N): [cx, cy, w, h, p_class0..p_classN] для каждого якоря — нужен NMS
        (YOLOv8/11/26 при экспорте по умолчанию);
      * (1, N, 6): [x1, y1, x2, y2, score, class] — end-to-end (NMS-free YOLO26 с nms=False
        или модель со встроенным NMS).
    """
    out = np.asarray(out, dtype=np.float32)[0]
    if out.shape[-1] == 6:  # уже готовые детекции
        out = out[out[:, 4] > conf]
        boxes, scores, classes = out[:, :4].copy(), out[:, 4], out[:, 5].astype(int)
    else:
        pred = out.T  # (N, 4+nc)
        cls_scores = pred[:, 4:]
        classes = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(pred)), classes]
        keep = scores > conf
        pred, classes, scores = pred[keep], classes[keep], scores[keep]
        boxes = np.empty((len(pred), 4), np.float32)
        boxes[:, :2] = pred[:, :2] - pred[:, 2:4] / 2  # cx,cy,w,h -> x1,y1
        boxes[:, 2:] = pred[:, :2] + pred[:, 2:4] / 2  # -> x2,y2
        if len(boxes):
            # NMS отдельно по каждому классу: сдвигаем рамки разных классов далеко друг от друга
            off = boxes + classes[:, None] * 7680.0
            xywh = np.concatenate([off[:, :2], off[:, 2:] - off[:, :2]], 1)
            idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou, top_k=max_det)
            idx = np.array(idx, dtype=int).reshape(-1)
            boxes, scores, classes = boxes[idx], scores[idx], classes[idx]

    # из координат входа сети обратно в координаты кадра
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r
    if orig_shape is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes, scores, classes


# --------------------------------------------------------------------------- отрисовка
def _color(i: int):
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
               (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
               (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203)]
    return palette[int(i) % len(palette)]


def draw(frame, boxes, scores, classes, names=COCO_NAMES, info: str | None = None):
    """Рисует рамки и подписи прямо на кадре (in-place) и возвращает его."""
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, classes):
        col = _color(c)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"{names[c] if c < len(names) else c} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), col, -1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    if info:
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- сравнение детекций
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU всех пар рамок a[N,4] и b[M,4] (xyxy) -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod((rb - lt).clip(0), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], 1)
    area_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def count_matches(ref, test, iou_thr: float = 0.5) -> int:
    """Сколько эталонных объектов найдено и в test (тот же класс, IoU >= iou_thr, жадное сопоставление)."""
    rb, _, rc = ref
    tb, _, tc = test
    if len(rb) == 0 or len(tb) == 0:
        return 0
    ious = box_iou(rb, tb) * (rc[:, None] == tc[None, :])
    matched = 0
    used = set()
    for i in np.argsort(-ious.max(1)):
        j = int(ious[i].argmax())
        if ious[i, j] >= iou_thr and j not in used:
            used.add(j)
            matched += 1
    return matched


# --------------------------------------------------------------------------- время
class Stat:
    """Накопитель времени (мс) для этапа конвейера."""

    def __init__(self):
        self.v: list[float] = []

    def add(self, ms: float):
        self.v.append(ms)

    @property
    def mean(self) -> float:
        return float(np.mean(self.v)) if self.v else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.v, 95)) if self.v else 0.0


class Timer:
    """with Timer() as t: ...  ->  t.ms"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000


# --------------------------------------------------------------------------- видео
def default_video() -> str:
    """Тестовое видео по умолчанию (создаётся скриптом make_demo_video.py)."""
    p = DATA / "demo.mp4"
    if not p.exists():
        raise SystemExit(
            "Не найдено тестовое видео data/demo.mp4.\n"
            "Создайте его: python make_demo_video.py  — или укажите своё: --source путь/к/видео.mp4"
        )
    return str(p)


def open_video(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))  # "0" = веб-камера
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {source}")
    return cap


def make_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25, (w, h))


def read_frames(source: str, n: int, stride: int | None = None) -> list[np.ndarray]:
    """Прочитать до n кадров, равномерно распределённых по видео (для калибровки/бенчмарка)."""
    cap = open_video(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    stride = stride or max(total // n, 1)
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def export_model(weights: str, fmt: str, dst: Path, quantize=None, **kw) -> Path:
    """Экспорт средствами Ultralytics в файл/папку dst.

    Экспорт идёт из временной копии весов: Ultralytics кладёт результат (и промежуточный ONNX
    для TensorRT/OpenVINO) рядом с весами под тем же именем — так варианты не затирают друг друга.
    """
    import shutil
    import tempfile

    from ultralytics import YOLO

    dst = Path(dst)
    if dst.exists():
        return dst
    with tempfile.TemporaryDirectory(dir=MODELS) as tmp:
        w = Path(tmp) / Path(weights).name
        shutil.copy(weights, w)
        out = Path(YOLO(str(w)).export(format=fmt, quantize=quantize, verbose=False, **kw))
        shutil.move(str(out), str(dst))
    return dst


def cpu_name() -> str:
    """Название процессора (для подписи таблиц с результатами)."""
    import platform
    import subprocess

    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def file_mb(path) -> float:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def md_table(rows: list[dict], cols: list[str]) -> str:
    """Список словарей -> таблица Markdown."""
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)
