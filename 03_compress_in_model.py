"""Шаг 3. Сжатие изображения внутри модели.

Обычно кадр перед подачей в сеть готовят в Python: BGR->RGB, уменьшение до 640, поля, /255,
HWC->CHW. Здесь всё это (включая уменьшение кадра) встраивается В САМ ГРАФ МОДЕЛИ:

    кадр uint8 [1,H,W,3] (как есть из cv2.VideoCapture)
      -> BGR->RGB, uint8->float
      -> сжатие: AvgPool k x k (усреднение блоков, заодно антиалиасинг) + билинейный resize
      -> дополнение до кратного 32 (stride сети), нормировка /255
      -> YOLO
      -> рамки пересчитываются обратно в координаты исходного кадра

Плюсы: вызывающий код (Python или C++) только передаёт кадр — никакого препроцессинга,
все операции выполняет ONNX Runtime/TensorRT в нативном коде (и на GPU, если он есть);
степень сжатия — параметр модели, её можно менять, не трогая код приложения.

Скрипт экспортирует несколько вариантов с разной степенью сжатия, прогоняет их по видео
и сравнивает скорость и качество (долю объектов, найденных эталонной моделью без сжатия).

Примеры:
    python 03_compress_in_model.py                            # data/demo.mp4, размеры 0 640 480 320
    python 03_compress_in_model.py --source my.mp4 --sizes 0 640 320 256
    (0 = без сжатия: сеть получает кадр в исходном разрешении)
"""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from common import (DEFAULT_MODEL, MODELS, RUNS, Stat, Timer, count_matches, default_video, draw, make_ort_session,
                    make_writer, md_table, model_path, onnx_names, open_video, ort_providers, postprocess)


class CompressedInputYOLO(nn.Module):
    """YOLO, которая принимает «сырой» кадр uint8 [1,H,W,3] BGR и сама его сжимает.

    target — длинная сторона после сжатия (0 = не сжимать). Размер кадра фиксируется при создании,
    поэтому граф полностью статический (лучше всего для ONNX Runtime / TensorRT / OpenCV DNN).
    """

    def __init__(self, det_model: nn.Module, frame_hw: tuple[int, int], target: int, stride: int = 32):
        super().__init__()
        H, W = frame_hw
        r = 1.0 if target <= 0 else target / max(H, W)
        self.nh, self.nw = round(H * r), round(W * r)  # размер после сжатия
        self.k = max(int(1 / r), 1)  # целый коэффициент для AvgPool (k x k пикселей -> 1)
        self.ph, self.pw = (-self.nh) % stride, (-self.nw) % stride  # поля до кратного stride
        self.net_hw = (self.nh + self.ph, self.nw + self.pw)  # фактический вход сети
        self.model = det_model
        sx, sy = W / self.nw, H / self.nh  # обратный масштаб для рамок
        self.register_buffer("scale_cxcywh", torch.tensor([sx, sy, sx, sy]).view(1, 4, 1))  # выход (1,4+nc,N)
        self.register_buffer("scale_xyxy", torch.tensor([sx, sy, sx, sy]).view(1, 1, 4))  # выход (1,N,6)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = frame.permute(0, 3, 1, 2).float()  # NHWC uint8 -> NCHW float
        x = x[:, [2, 1, 0]]  # BGR -> RGB
        if self.k > 1:  # грубое сжатие усреднением (не теряет мелкие детали так, как простое прореживание)
            x = F.avg_pool2d(x, self.k, self.k)
        if tuple(x.shape[-2:]) != (self.nh, self.nw):  # точная подгонка размера
            x = F.interpolate(x, size=(self.nh, self.nw), mode="bilinear", align_corners=False)
        if self.ph or self.pw:
            x = F.pad(x, (0, self.pw, 0, self.ph), value=114.0)  # серые поля справа/снизу, как в Ultralytics
        y = self.model(x / 255.0)
        if y.shape[-1] == 6:  # end-to-end выход [x1,y1,x2,y2,score,cls]
            return torch.cat([y[..., :4] * self.scale_xyxy, y[..., 4:]], -1)
        return torch.cat([y[:, :4] * self.scale_cxcywh, y[:, 4:]], 1)  # [cx,cy,w,h,scores...]


def load_export_ready(weights: str) -> tuple[nn.Module, dict]:
    """Загрузить YOLO и перевести в «режим экспорта» (так же, как это делает Ultralytics Exporter)."""
    y = YOLO(weights)
    m = y.model.float().eval()
    for p in m.parameters():
        p.requires_grad = False
    m = m.fuse()  # склейка Conv+BatchNorm
    for mod in m.modules():
        if isinstance(mod, Detect):
            mod.export, mod.format, mod.dynamic = True, "onnx", False  # выход — один тензор
        if hasattr(mod, "export") and type(mod).__name__ == "SPPF":
            mod.export = True
    return m, y.names


def export_onnx(module: CompressedInputYOLO, frame_hw, path: Path, names: dict):
    dummy = torch.zeros(1, *frame_hw, 3, dtype=torch.uint8)
    with warnings.catch_warnings():  # экспортёр TorchScript шумит предупреждениями, на результат они не влияют
        warnings.simplefilter("ignore")
        torch.onnx.export(module, dummy, str(path), opset_version=17, input_names=["frame"],
                          output_names=["output0"], do_constant_folding=True, dynamo=False)
    model = onnx.load(str(path))
    try:  # упрощение графа (убирает лишние узлы), если установлен onnxslim
        import onnxslim

        model = onnxslim.slim(model)
    except Exception:  # noqa: BLE001
        pass
    meta = {"names": str(names), "frame_hw": str(list(frame_hw)), "net_hw": str(list(module.net_hw)),
            "description": "YOLO with in-model input compression (uint8 BGR NHWC frame -> boxes in frame coords)"}
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="видео (по умолчанию data/demo.mp4)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 640, 480, 320, 256],
                    help="длинная сторона после сжатия; 0 = без сжатия (сеть получает исходный кадр)")
    ap.add_argument("--ref", type=int, default=640,
                    help="эталон для сравнения детекций на видео: 640 — родное разрешение, на котором обучена YOLO")
    ap.add_argument("--val-data", default="coco128.yaml",
                    help="датасет для mAP при тех же степенях сжатия (скачается ~7 МБ); 'none' — не считать")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--no-save", action="store_true", help="не сохранять видео с разметкой")
    args = ap.parse_args()

    source = args.source or default_video()
    cap = open_video(source)
    ok, frame0 = cap.read()
    cap.release()
    H, W = frame0.shape[:2]
    print(f"Кадр видео: {W}x{H}")

    det_model, names = load_export_ready(model_path(args.model))
    stem = Path(args.model).stem
    variants = []
    for S in args.sizes:
        m = CompressedInputYOLO(det_model, (H, W), S).eval()
        path = MODELS / f"{stem}_compressed_{W}x{H}_to_{S or 'full'}.onnx"
        if not path.exists():
            export_onnx(m, (H, W), path, names)
        variants.append((S, m.net_hw, path))
        print(f"  экспорт: сжатие до {S or 'нет'} -> вход сети {m.net_hw[1]}x{m.net_hw[0]}  ({path.name})")

    # --- проверка: та же модель в PyTorch и в ONNX Runtime должна давать одинаковый результат
    with torch.no_grad():
        ref_torch = CompressedInputYOLO(det_model, (H, W), args.sizes[-1])(torch.from_numpy(frame0[None])).numpy()
    sess = ort.InferenceSession(str(variants[-1][2]), providers=["CPUExecutionProvider"])
    ref_ort = sess.run(None, {"frame": frame0[None]})[0]
    print(f"Проверка PyTorch vs ONNX Runtime: max |разница| = {np.abs(ref_torch - ref_ort).max():.4f}")

    # --- прогон по видео
    providers = ort_providers()
    print(f"ONNX Runtime providers: {providers}")
    results, rows = {}, []
    for S, net_hw, path in variants:
        sess = make_ort_session(path, providers)
        cls_names = onnx_names(sess)
        cap = open_video(source)
        writer = None if args.no_save else make_writer(RUNS / "03_compress" / f"{stem}_{S or 'full'}.mp4",
                                                       cap.get(cv2.CAP_PROP_FPS), W, H)
        for _ in range(3):  # прогрев
            sess.run(None, {"frame": frame0[None]})
        t_inf, t_post, dets = Stat(), Stat(), []
        while len(dets) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            with Timer() as t:  # ВЕСЬ препроцессинг + сеть внутри одного вызова
                out = sess.run(None, {"frame": frame[None]})[0]
            t_inf.add(t.ms)
            with Timer() as t:  # остаётся только NMS; рамки уже в координатах кадра
                d = postprocess(out, args.conf, args.iou, orig_shape=(H, W))
            t_post.add(t.ms)
            dets.append(d)
            if writer:
                writer.write(draw(frame, *d, cls_names, info=f"in-model resize -> {net_hw[1]}x{net_hw[0]}, "
                                                             f"{t_inf.v[-1]:.0f} ms"))
        cap.release()
        if writer:
            writer.release()
        results[S] = dets
        rows.append({"S": S, "net_hw": net_hw, "path": path, "inf": t_inf, "post": t_post})

    ref_size = args.ref if args.ref in results else args.sizes[0]
    ref = results[ref_size]
    n_ref = sum(len(d[0]) for d in ref)
    table = []
    for row in rows:
        dets = results[row["S"]]
        matched = sum(count_matches(r, d) for r, d in zip(ref, dets))
        h_, w_ = row["net_hw"]
        total_ms = row["inf"].mean + row["post"].mean
        table.append({
            "Сжатие до": f"{row['S'] or 'нет'}{' (эталон)' if row['S'] == ref_size else ''}",
            "Вход сети": f"{w_}x{h_}",
            "Пикселей, % от кадра": f"{100 * h_ * w_ / (H * W):.0f}",
            "Сеть+препроцесс, мс": f"{row['inf'].mean:.1f}",
            "NMS, мс": f"{row['post'].mean:.2f}",
            "FPS": f"{1000 / total_ms:.1f}",
            "Объектов/кадр": f"{np.mean([len(d[0]) for d in dets]):.2f}",
            "Найдено от эталона, %": f"{100 * matched / max(n_ref, 1):.1f}",
            "Ср. уверенность": f"{np.mean(np.concatenate([d[1] for d in dets]) if n_ref else 0):.3f}",
        })
    # Точность на размеченных данных (COCO): сжатие до S == imgsz=S в Ultralytics (та же операция resize+поля)
    if args.val_data.lower() != "none":
        print(f"\nСчитаю mAP на {args.val_data} для каждой степени сжатия...")
        yolo = YOLO(model_path(args.model))
        for row, t in zip(rows, table):
            if row["S"] <= 0:  # картинки COCO сами ~640 px, «без сжатия» для них не отличается от 640
                t["mAP50-95 (COCO)"] = "—"
                continue
            m = yolo.val(data=args.val_data, imgsz=row["S"], batch=8, plots=False, verbose=False,
                         project=str(RUNS / "03_compress" / "val"), name=str(row["S"]), exist_ok=True)
            t["mAP50-95 (COCO)"] = f"{m.box.map:.3f}"
    cols = list(table[0].keys())
    md = md_table(table, cols)
    print(f"\nРезультаты ({len(ref)} кадров, видео {W}x{H}):\n")
    print(md)
    out_dir = RUNS / "03_compress"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nТаблица: {out_dir / 'results.md'}" + ("" if args.no_save else f"\nВидео: {out_dir}"))


if __name__ == "__main__":
    main()
