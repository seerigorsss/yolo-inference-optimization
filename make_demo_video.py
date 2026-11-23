"""Создаёт тестовое видео data/demo.mp4 (1280x720, 25 к/с, ~12 с), если под рукой нет своего.

Видео собирается из двух картинок, которые идут в комплекте с Ultralytics (bus.jpg, zidane.jpg):
камера «едет» и приближается, поэтому объекты двигаются и меняют размер — как на реальной съёмке.
Можно использовать и любое своё видео: python 02_detect_video.py --source my.mp4
"""

import argparse

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(DATA / "demo.mp4"))
ap.add_argument("--w", type=int, default=1280)
ap.add_argument("--h", type=int, default=720)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--seconds", type=float, default=12)
args = ap.parse_args()

W, H = args.w, args.h
aspect = W / H
imgs = [cv2.imread(str(ASSETS / "bus.jpg")), cv2.imread(str(ASSETS / "zidane.jpg"))]


def ken_burns(img, t, zoom_from, zoom_to, pan_from, pan_to):
    """Вырезает окно с плавным зумом/панорамой (t in [0,1]) и приводит к W x H."""
    ih, iw = img.shape[:2]
    e = 0.5 - 0.5 * np.cos(np.pi * t)  # плавное ускорение/замедление
    z = zoom_from + (zoom_to - zoom_from) * e
    cw = min(iw, ih * aspect) / z
    ch = cw / aspect
    px = pan_from[0] + (pan_to[0] - pan_from[0]) * e
    py = pan_from[1] + (pan_to[1] - pan_from[1]) * e
    x0 = int((iw - cw) * px)
    y0 = int((ih - ch) * py)
    crop = img[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


# (картинка, зум от, зум до, панорама от, панорама до) — панорама: доли свободного места по x/y
segments = [
    (0, 1.0, 1.6, (0.5, 0.2), (0.2, 0.6)),
    (0, 1.6, 1.1, (0.2, 0.6), (0.9, 0.9)),
    (1, 1.0, 1.8, (0.5, 0.5), (0.85, 0.3)),
    (1, 1.8, 1.2, (0.85, 0.3), (0.1, 0.6)),
]
n_total = int(args.fps * args.seconds)
per_seg = n_total // len(segments)

DATA.mkdir(exist_ok=True)
writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
rng = np.random.default_rng(0)
for img_id, z0, z1, p0, p1 in segments:
    for i in range(per_seg):
        frame = ken_burns(imgs[img_id], i / (per_seg - 1), z0, z1, p0, p1)
        noise = rng.normal(0, 3, frame.shape).astype(np.int16)  # лёгкий шум «как у камеры»
        writer.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
writer.release()
print(f"Готово: {args.out}  ({W}x{H}, {args.fps} к/с, {per_seg * len(segments)} кадров)")
