"""Шаг 4. Квантование модели (FP32 -> FP16 / INT8) и сравнение размера, скорости и точности.

Что делает скрипт:
  1. Экспортирует YOLO в ONNX FP32 (models/yolo26n.onnx) и FP16 (models/yolo26n_fp16.onnx).
  2. Собственная реализация статического INT8-квантования (post-training quantization) через
     ONNX Runtime: калибровка на кадрах видео, формат QDQ, веса — per-channel INT8,
     активации — UINT8. В INT8 переводится вся сеть, кроме декодера на выходе головы
     (сборка рамок и вероятностей классов в один тензор) — он остаётся во float.
  3. Для сравнения ещё три варианта INT8:
       * «только свёртки» — float между свёртками, много переходов float<->int8;
       * «весь граф» — квантуется и декодер (видно, почему так делать нельзя);
       * динамическое квантование — только веса, масштаб активаций считается на лету.
  4. Для каждого варианта: размер файла, задержка в ONNX Runtime (CPU), mAP на COCO128
     и совпадение детекций на видео с FP32.
  5. --extra openvino engine coreml — то же через «родные» инструменты железа (Ultralytics export):
     OpenVINO INT8 (Intel/AMD CPU), TensorRT FP16/INT8 (NVIDIA), CoreML INT8 (Apple).

Примеры:
    python 04_quantize.py
    python 04_quantize.py --calib my_video.mp4 --calib-n 200 --method entropy
    python 04_quantize.py --extra openvino          # + OpenVINO FP32/INT8 (pip install openvino)
    python 04_quantize.py --extra engine            # + TensorRT FP16/INT8 (нужна NVIDIA GPU)
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

from common import (DEFAULT_MODEL, MODELS, RUNS, Stat, Timer, count_matches, default_video, export_model, file_mb,
                    make_ort_session, md_table, model_path, postprocess, preprocess, read_frames)

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- калибровка
class FrameCalibrationReader(CalibrationDataReader):
    """Отдаёт ONNX Runtime подготовленные кадры по одному — по ним собирается статистика активаций."""

    def __init__(self, blobs: list[np.ndarray], input_name: str = "images"):
        self.blobs, self.input_name, self.i = blobs, input_name, 0

    def get_next(self):
        if self.i >= len(self.blobs):
            return None
        self.i += 1
        return {self.input_name: self.blobs[self.i - 1]}

    def rewind(self):
        self.i = 0


def load_calibration(sources: list[str], n: int, imgsz: int) -> list[np.ndarray]:
    """Кадры из видео и/или картинки из папок -> список тензоров [1,3,imgsz,imgsz]."""
    images = []
    per_source = max(n // len(sources), 1)
    for src in sources:
        p = Path(src)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            step = max(len(files) // per_source, 1)
            images += [cv2.imread(str(f)) for f in files[::step][:per_source]]
        else:
            images += read_frames(str(p), per_source)
    return [preprocess(im, imgsz)[0] for im in images if im is not None]


def copy_metadata(src: Path, dst: Path):
    """Квантование теряет метаданные Ultralytics (имена классов, stride, imgsz) — переносим их."""
    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    have = {p.key for p in m_dst.metadata_props}
    for p in m_src.metadata_props:
        if p.key not in have:
            m_dst.metadata_props.add(key=p.key, value=p.value)
    onnx.save(m_dst, str(dst))


def nodes_to_keep_float(model: onnx.ModelProto, scope: str) -> list[str]:
    """Какие узлы графа НЕ квантовать.

    Декодер на выходе YOLO склеивает координаты рамок (0..640 пикс.) и вероятности классов (0..1)
    в один тензор. Один INT8-масштаб на такой тензор округлит все вероятности до нуля, поэтому:
      head-float — квантуем всё, кроме узлов декодера (узлы последнего модуля /model.N/ вне ветвей cv*);
      convs      — квантуем только операции с весами (Conv/MatMul), всё остальное во float;
      whole      — квантуем всё (демонстрация поломки).
    """
    if scope == "whole":
        return []
    weighted = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
    idx = [int(m.group(1)) for n in model.graph.node if (m := re.match(r"/model\.(\d+)/", n.name))]
    if scope == "convs" or not idx:  # имена узлов не в стиле Ultralytics -> безопасный вариант
        return [n.name for n in model.graph.node if n.op_type not in weighted]
    head = f"/model.{max(idx)}/"  # последний модуль сети — голова Detect
    return [n.name for n in model.graph.node if n.name.startswith(head) and not n.name.startswith(head + "cv")]


def int8_static(fp32: Path, out: Path, blobs, method: str, scope: str = "head-float"):
    prep = out.with_name(out.stem + "_prep.onnx")
    # 1) символьный вывод форм + оптимизация графа (склейка операций) — рекомендовано перед квантованием
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=False)
    model = onnx.load(str(prep))
    exclude = nodes_to_keep_float(model, scope)
    del model
    calib = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
             "percentile": CalibrationMethod.Percentile}[method]
    quantize_static(
        str(prep), str(out), FrameCalibrationReader(blobs),
        quant_format=QuantFormat.QDQ,  # QuantizeLinear/DequantizeLinear вокруг узлов — понимают ORT, TensorRT
        per_channel=True,  # свой масштаб на каждый выходной канал свёртки — меньше ошибка
        activation_type=QuantType.QUInt8,  # U8S8 — рекомендованная схема для x86/ARM CPU
        weight_type=QuantType.QInt8,
        calibrate_method=calib,
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)
    copy_metadata(fp32, out)


def int8_dynamic(fp32: Path, out: Path):
    prep = out.with_name(out.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prep))
    quantize_dynamic(str(prep), str(out), weight_type=QuantType.QUInt8)  # ConvInteger на CPU требует uint8-веса
    prep.unlink(missing_ok=True)
    copy_metadata(fp32, out)


# --------------------------------------------------------------------------- оценка
def ort_latency_and_dets(path: Path, frames, imgsz: int, providers, conf=0.25, iou=0.45):
    sess = make_ort_session(path, providers)
    name = sess.get_inputs()[0].name
    blobs = [preprocess(f, imgsz) for f in frames]
    for _ in range(5):
        sess.run(None, {name: blobs[0][0]})
    st, dets = Stat(), []
    for (blob, r, pad), f in zip(blobs, frames):
        with Timer() as t:
            out = sess.run(None, {name: blob})[0]
        st.add(t.ms)
        dets.append(postprocess(out, conf, iou, r, pad, f.shape[:2]))
    return st, dets


def val_map(path, data: str, imgsz: int, device: str = "cpu"):
    m = YOLO(str(path), task="detect").val(data=data, imgsz=imgsz, batch=1, device=device, plots=False,
                                           verbose=False, project=str(RUNS / "04_quantize" / "val"),
                                           name=Path(path).stem, exist_ok=True)
    return m.box.map50, m.box.map, m.speed["inference"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib", nargs="+", help="видео и/или папки с картинками для калибровки (по умолчанию data/demo.mp4)")
    ap.add_argument("--calib-n", type=int, default=64, help="число калибровочных кадров")
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--video", help="видео для сравнения скорости/детекций (по умолчанию data/demo.mp4)")
    ap.add_argument("--frames", type=int, default=60, help="кадров видео для замера")
    ap.add_argument("--val-data", default="coco128.yaml", help="датасет для mAP; 'none' — пропустить")
    ap.add_argument("--extra", nargs="*", default=[], choices=["openvino", "engine", "coreml"])
    ap.add_argument("--no-demo", action="store_true", help="не строить демонстрационные INT8 (весь граф, динамический)")
    args = ap.parse_args()

    weights = model_path(args.model)
    stem = Path(weights).stem
    video = args.video or default_video()
    calib_sources = args.calib or [video]

    # 1. FP16 и FP32 ONNX
    print("== Экспорт ONNX FP16 / FP32 ==")
    fp32 = export_model(weights, "onnx", MODELS / f"{stem}.onnx", imgsz=args.imgsz)
    # FP16 экспортируем на CPU: веса в FP16, а вход/выход остаются float32 — модель подходит для любого рантайма
    fp16 = export_model(weights, "onnx", MODELS / f"{stem}_fp16.onnx", quantize=16, imgsz=args.imgsz, device="cpu")

    # 2-3. INT8
    print(f"== Калибровка INT8: {calib_sources}, {args.calib_n} кадров, метод {args.method} ==")
    blobs = load_calibration(calib_sources, args.calib_n, args.imgsz)
    int8 = MODELS / f"{stem}_int8.onnx"
    int8_static(fp32, int8, blobs, args.method, scope="head-float")
    variants = [("FP32 (исходная)", fp32), ("FP16", fp16), ("INT8 static, декодер во float (основной)", int8)]
    if not args.no_demo:
        demo = [("INT8 static, только свёртки", "convs", "_int8_convs"),
                ("INT8 static, весь граф (с декодером)", "whole", "_int8_wholegraph")]
        for title, scope, suffix in demo:
            path = MODELS / f"{stem}{suffix}.onnx"
            int8_static(fp32, path, blobs, args.method, scope=scope)
            variants.append((title, path))
        int8_dyn = MODELS / f"{stem}_int8_dynamic.onnx"
        int8_dynamic(fp32, int8_dyn)
        variants.append(("INT8 dynamic (только веса)", int8_dyn))

    # 4. Оценка ONNX-вариантов в ONNX Runtime на CPU (INT8 даёт выигрыш прежде всего на CPU)
    frames = read_frames(video, args.frames)
    rows, ref_dets = [], None
    for title, path in variants:
        print(f"== Оценка: {title} ==")
        st, dets = ort_latency_and_dets(path, frames, args.imgsz, ["CPUExecutionProvider"])
        if ref_dets is None:
            ref_dets = dets
        n_ref = sum(len(d[0]) for d in ref_dets)
        agree = 100 * sum(count_matches(r, d) for r, d in zip(ref_dets, dets)) / max(n_ref, 1)
        row = {"Вариант": title, "Файл": path.name, "Размер, МБ": f"{file_mb(path):.1f}",
               "Задержка CPU, мс": f"{st.mean:.1f}", "Ускорение": "", "Совпадение с FP32 на видео, %": f"{agree:.1f}"}
        if args.val_data.lower() != "none":
            m50, m, _ = val_map(path, args.val_data, args.imgsz)
            row.update({"mAP50": f"{m50:.3f}", "mAP50-95": f"{m:.3f}"})
        row["_ms"] = st.mean
        rows.append(row)
    for r in rows:
        r["Ускорение"] = f"x{rows[0]['_ms'] / r['_ms']:.2f}"

    # 5. Родные инструменты железа (через Ultralytics)
    extra_rows = []
    for fmt in args.extra:
        dev = {"engine": "0"}.get(fmt, "cpu")
        suffix = {"openvino": "_openvino_model", "engine": ".engine", "coreml": ".mlpackage"}[fmt]
        precisions = {"openvino": [None, 8], "engine": [16, 8], "coreml": [16, 8]}[fmt]
        for q in precisions:
            tag = {None: "fp32", 16: "fp16", 8: "int8"}[q]
            dst = MODELS / f"{stem}_{tag}{suffix}"
            print(f"== Экспорт {fmt} {tag} ==")
            try:
                kw = dict(imgsz=args.imgsz, device=dev)
                if q == 8 and fmt != "coreml":
                    kw["data"] = args.val_data if args.val_data.lower() != "none" else "coco128.yaml"
                export_model(weights, fmt, dst, quantize=q, **kw)
                m50, m, inf_ms = val_map(dst, args.val_data if args.val_data.lower() != "none" else "coco128.yaml",
                                         args.imgsz, dev)
                extra_rows.append({"Вариант": f"{fmt} {tag.upper()}", "Файл": dst.name,
                                   "Размер, МБ": f"{file_mb(dst):.1f}", "Инференс, мс": f"{inf_ms:.1f}",
                                   "mAP50": f"{m50:.3f}", "mAP50-95": f"{m:.3f}"})
            except Exception as e:  # noqa: BLE001
                print(f"   пропущено: {type(e).__name__}: {e}")

    out_dir = RUNS / "04_quantize"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in rows[0] if not c.startswith("_")]
    md = "### ONNX Runtime, CPU\n\n" + md_table(rows, cols)
    if extra_rows:
        md += "\n\n### Родные инструменты (Ultralytics export)\n\n" + md_table(extra_rows, list(extra_rows[0]))
    print("\n" + md)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows + extra_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nТаблица: {out_dir / 'results.md'}")
    print(f"INT8-модель для дальнейших шагов: {int8}")


if __name__ == "__main__":
    main()
