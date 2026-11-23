"""Шаг 1. Запуск предобученной сети YOLO из пакета Ultralytics на изображениях.

Примеры:
    python 01_detect_image.py                         # встроенные картинки bus.jpg и zidane.jpg
    python 01_detect_image.py --source photo.jpg      # своя картинка (или папка с картинками)
    python 01_detect_image.py --model yolo11n.pt      # другая модель
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS

from common import DEFAULT_MODEL, RUNS, model_path, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL, help="веса Ultralytics: yolo26n.pt, yolo11n.pt, yolo26s.pt ...")
ap.add_argument("--source", nargs="*", help="картинки/папка; по умолчанию bus.jpg и zidane.jpg из пакета")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--device", default="auto", help="auto | cpu | 0 (CUDA) | mps (Apple)")
args = ap.parse_args()

device = pick_device(args.device)
model = YOLO(model_path(args.model))  # при первом запуске веса скачаются автоматически (~5 МБ)
print(f"Модель: {args.model}, устройство: {device}")
model.info()  # число слоёв, параметров и GFLOPs

sources = args.source or [str(ASSETS / "bus.jpg"), str(ASSETS / "zidane.jpg")]
results = model.predict(
    sources, imgsz=args.imgsz, conf=args.conf, device=device,
    save=True, project=str(RUNS), name="01_images", exist_ok=True, verbose=False,
)

for r in results:
    s = r.speed
    print(f"\n{Path(r.path).name}: {len(r.boxes)} объектов | "
          f"препроцессинг {s['preprocess']:.1f} мс, инференс {s['inference']:.1f} мс, "
          f"постпроцессинг {s['postprocess']:.1f} мс")
    for b in r.boxes:
        x1, y1, x2, y2 = (round(v) for v in b.xyxy[0].tolist())
        print(f"   {r.names[int(b.cls)]:<14} conf={float(b.conf):.2f}  box=({x1}, {y1}, {x2}, {y2})")

print(f"\nКартинки с рамками сохранены в: {results[0].save_dir}")
