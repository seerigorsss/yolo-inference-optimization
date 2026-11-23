"""Шаг 2. Распознавание объектов в видеофайле (Python + Ultralytics).

Кадры читаются через OpenCV, каждый кадр прогоняется через YOLO, рамки рисуются и
записываются в новое видео. В конце печатается разбивка времени по этапам и FPS.

Модель может быть в любом формате, который понимает Ultralytics:
    yolo26n.pt (PyTorch), yolo26n.onnx (ONNX Runtime), yolo26n_openvino_model/,
    yolo26n.engine (TensorRT), yolo26n.mlpackage (CoreML) ...

Примеры:
    python 02_detect_video.py                                   # data/demo.mp4, PyTorch
    python 02_detect_video.py --source my.mp4 --show            # своё видео с окном просмотра
    python 02_detect_video.py --source 0                        # веб-камера
    python 02_detect_video.py --model models/yolo26n.onnx       # то же через ONNX Runtime
    python 02_detect_video.py --fp16 --device 0                 # половинная точность на NVIDIA GPU
    python 02_detect_video.py --imgsz 320                       # «сжатие» входа средствами Ultralytics
    python 02_detect_video.py --vid-stride 2                    # обрабатывать каждый 2-й кадр
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import DEFAULT_MODEL, RUNS, Stat, default_video, make_writer, model_path, open_video, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--source", help="видеофайл, папка не поддерживается; 0 = веб-камера (по умолчанию data/demo.mp4)")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--imgsz", type=int, default=640, help="размер входа сети (для экспортированных моделей — как при экспорте)")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--iou", type=float, default=0.45)
ap.add_argument("--device", default="auto", help="auto | cpu | 0 | mps")
ap.add_argument("--fp16", action="store_true", help="FP16-вычисления (только GPU: CUDA/MPS)")
ap.add_argument("--vid-stride", type=int, default=1, help="обрабатывать каждый N-й кадр")
ap.add_argument("--max-frames", type=int, default=0, help="ограничить число кадров (0 = всё видео)")
ap.add_argument("--show", action="store_true", help="показывать окно (q — выход)")
ap.add_argument("--out", help="куда сохранить видео (по умолчанию runs/02_video/<имя>_<модель>.mp4)")
args = ap.parse_args()

source = args.source or default_video()
device = pick_device(args.device)
model = YOLO(model_path(args.model), task="detect")
predict_kw = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, verbose=False)
if args.fp16:
    predict_kw["quantize"] = 16  # в Ultralytics 8.4 параметр quantize заменил устаревший half=True

cap = open_video(source)
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_path = Path(args.out) if args.out else RUNS / "02_video" / f"{Path(str(source)).stem}_{Path(args.model).stem}.mp4"
writer = make_writer(out_path, fps_in / args.vid_stride, w, h)
print(f"Видео: {source}  {w}x{h}, {fps_in:.1f} к/с, кадров: {total}")
print(f"Модель: {args.model}  устройство: {device}  imgsz={args.imgsz}  fp16={args.fp16}")

# Прогрев: первые запуски медленные (выделение памяти, автотюнинг ядер, JIT) — не учитываем их
ok, frame = cap.read()
if not ok:
    raise SystemExit("Видео пустое")
for _ in range(3):
    model.predict(frame, **predict_kw)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

st = {k: Stat() for k in ("read", "preprocess", "inference", "postprocess", "draw+write")}
counts, max_per_frame = Counter(), Counter()
n, idx = 0, -1
t_start = time.perf_counter()
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    st["read"].add((time.perf_counter() - t0) * 1000)
    if not ok:
        break
    idx += 1
    if idx % args.vid_stride:
        continue

    r = model.predict(frame, **predict_kw)[0]
    for k in ("preprocess", "inference", "postprocess"):
        st[k].add(r.speed[k])

    t0 = time.perf_counter()
    model_ms = sum(r.speed.values())
    annotated = r.plot(line_width=2)
    cv2.putText(annotated, f"{Path(args.model).name} | {model_ms:.0f} ms ({1000 / model_ms:.0f} FPS)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    writer.write(annotated)
    st["draw+write"].add((time.perf_counter() - t0) * 1000)

    frame_counts = Counter(r.names[int(c)] for c in r.boxes.cls)
    counts.update(frame_counts)
    for k, v in frame_counts.items():
        max_per_frame[k] = max(max_per_frame[k], v)

    n += 1
    if args.show:
        cv2.imshow("YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if n % 50 == 0:
        print(f"  обработано кадров: {n}")
    if args.max_frames and n >= args.max_frames:
        break

wall = time.perf_counter() - t_start
cap.release()
writer.release()
cv2.destroyAllWindows()

model_ms = st["preprocess"].mean + st["inference"].mean + st["postprocess"].mean
print(f"\nОбработано кадров: {n} за {wall:.1f} с")
print("Среднее время на кадр, мс:")
for k, s in st.items():
    print(f"  {k:<12} {s.mean:7.2f}   (p95 {s.p95:.2f})")
print(f"FPS модели (pre+inference+post): {1000 / model_ms:.1f}")
print(f"FPS всего конвейера (с чтением, отрисовкой и записью): {n / wall:.1f}")
print("Максимум объектов на одном кадре:", dict(max_per_frame.most_common()))
print(f"Видео с разметкой: {out_path}")

summary = {
    "source": str(source), "model": args.model, "device": device, "imgsz": args.imgsz, "fp16": args.fp16,
    "frames": n, "ms": {k: round(s.mean, 2) for k, s in st.items()},
    "fps_model": round(1000 / model_ms, 1), "fps_pipeline": round(n / wall, 1),
    "max_objects_per_frame": dict(max_per_frame),
}
(out_path.parent / f"{out_path.stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
