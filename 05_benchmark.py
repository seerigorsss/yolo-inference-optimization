"""Шаг 5. Сравнение способов ускорения распознавания видеопотока на одном и том же видео.

Для каждого доступного на компьютере бэкенда одинаково измеряются:
  препроцессинг (letterbox, NumPy/OpenCV) -> инференс -> постпроцессинг (NMS),
а также совпадение детекций с эталоном (PyTorch FP32) — чтобы убедиться, что ускорение
не сломало результат.

Бэкенды (берутся только те, что реально есть в системе):
  PyTorch            CPU / CUDA FP32 / CUDA FP16 / Apple MPS
  ONNX Runtime       CPU FP32, CPU INT8 (из шага 4), CUDA EP, TensorRT EP, CoreML EP, DirectML EP, OpenVINO EP
  OpenCV DNN         CPU, CUDA, CUDA FP16 (CUDA — только у OpenCV 4.x, собранного с CUDA)
  TensorRT           .engine FP16/INT8 (Ultralytics export; нужна NVIDIA GPU и pip install tensorrt)
  OpenVINO           FP32/INT8 (Intel/AMD CPU, Intel GPU/NPU; pip install openvino)
  CoreML             .mlpackage (только macOS)

Примеры:
    python 05_benchmark.py                       # всё, что найдётся; недостающие модели экспортируются
    python 05_benchmark.py --frames 200 --source my.mp4
    python 05_benchmark.py --only ort opencv     # только выбранные группы
    python 05_benchmark.py --no-export           # не экспортировать (TensorRT-движок строится несколько минут)
"""

import argparse
import json
import platform
import warnings
from pathlib import Path

import cv2
import numpy as np

from common import (DEFAULT_MODEL, MODELS, RUNS, Stat, Timer, count_matches, cpu_name, default_video, export_model,
                    file_mb, md_table, model_path, postprocess, preprocess, read_frames)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- раннеры
class UltraRunner:
    """Любой формат, который загружает Ultralytics AutoBackend: .pt, .engine, OpenVINO, CoreML, ..."""

    def __init__(self, weights, device="cpu", fp16=False):
        import torch
        from ultralytics.nn.autobackend import AutoBackend

        self.torch = torch
        self.device = torch.device("cuda:0" if device == "0" else device)
        self.model = AutoBackend(model=str(weights), device=self.device, fp16=fp16, verbose=False)
        self.fp16 = self.model.fp16

    def __call__(self, blob):
        x = self.torch.from_numpy(blob).to(self.device)
        x = x.half() if self.fp16 else x.float()
        with self.torch.inference_mode():
            y = self.model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y.float().cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)  # .cpu() = синхронизация с GPU


class OrtRunner:
    def __init__(self, path, provider, options=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "DmlExecutionProvider":  # требование DirectML EP
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [(provider, options or {})] + ([] if provider == "CPUExecutionProvider" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        used = self.sess.get_providers()[0]
        if used != provider:
            raise RuntimeError(f"{provider} не запустился, ORT откатился на {used}")
        self.name = self.sess.get_inputs()[0].name
        self.half = self.sess.get_inputs()[0].type == "tensor(float16)"

    def __call__(self, blob):
        return self.sess.run(None, {self.name: blob.astype(np.float16) if self.half else blob})[0]


class OpenCVRunner:
    def __init__(self, path, target="cpu"):
        # Ultralytics при импорте делает cv2.setNumThreads(0) (однопоточный OpenCV) — для честного сравнения
        # возвращаем число потоков по умолчанию, как у ONNX Runtime
        cv2.setNumThreads(-1)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        if target != "cpu":
            if int(cv2.__version__.split(".")[0]) >= 5:
                raise RuntimeError("в OpenCV 5.x новый DNN-движок пока работает только на CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16 if target == "cuda-fp16"
                                         else cv2.dnn.DNN_TARGET_CUDA)

    def __call__(self, blob):
        self.net.setInput(blob)
        return self.net.forward()


# --------------------------------------------------------------------------- что есть в системе
def detect_backends(args, stem, weights):
    import onnxruntime as ort
    import torch

    fp32, int8 = MODELS / f"{stem}.onnx", MODELS / f"{stem}_int8.onnx"
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    ort_eps = ort.get_available_providers()
    cv_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    only = set(args.only or [])
    want = lambda g: not only or g in only  # noqa: E731

    def ensure(dst: Path, fmt: str, q=None, **kw):
        """Экспорт через Ultralytics, если файла ещё нет."""
        if not dst.exists() and args.no_export:
            raise FileNotFoundError(f"{dst.name} нет (запущено с --no-export)")
        return export_model(weights, fmt, dst, quantize=q, imgsz=args.imgsz, **kw)

    b = []  # (группа, название, фабрика раннера, файл модели)
    if want("torch"):
        b.append(("torch", "PyTorch CPU FP32", lambda: UltraRunner(weights, "cpu"), weights))
        if cuda:
            b.append(("torch", "PyTorch CUDA FP32", lambda: UltraRunner(weights, "0"), weights))
            b.append(("torch", "PyTorch CUDA FP16", lambda: UltraRunner(weights, "0", fp16=True), weights))
        if mps:
            b.append(("torch", "PyTorch Apple MPS", lambda: UltraRunner(weights, "mps"), weights))
    if want("ort"):
        b.append(("ort", "ONNX Runtime CPU FP32", lambda: OrtRunner(ensure(fp32, "onnx"), "CPUExecutionProvider"), fp32))
        if int8.exists():
            b.append(("ort", "ONNX Runtime CPU INT8 (шаг 4)", lambda: OrtRunner(int8, "CPUExecutionProvider"), int8))
        if "CUDAExecutionProvider" in ort_eps:
            b.append(("ort", "ONNX Runtime CUDA EP", lambda: OrtRunner(ensure(fp32, "onnx"), "CUDAExecutionProvider"), fp32))
        if "TensorrtExecutionProvider" in ort_eps:
            # Точность задаётся самой моделью (FP16-веса или INT8 Q/DQ) — так работает и TensorRT 10, и TensorRT 11,
            # где флаги FP16/INT8 у сборщика убраны («strongly typed» сети).
            opts = {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(MODELS / "trt_cache")}
            fp16 = MODELS / f"{stem}_fp16.onnx"
            b.append(("ort", "ONNX Runtime TensorRT EP FP16", lambda: OrtRunner(
                ensure(fp16, "onnx", 16, device="cpu"), "TensorrtExecutionProvider", opts), fp16))
            if int8.exists():
                b.append(("ort", "ONNX Runtime TensorRT EP INT8 (Q/DQ из шага 4)",
                          lambda: OrtRunner(int8, "TensorrtExecutionProvider", opts), int8))
        if "CoreMLExecutionProvider" in ort_eps:
            b.append(("ort", "ONNX Runtime CoreML EP",
                      lambda: OrtRunner(ensure(fp32, "onnx"), "CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}), fp32))
        if "DmlExecutionProvider" in ort_eps:
            b.append(("ort", "ONNX Runtime DirectML EP", lambda: OrtRunner(ensure(fp32, "onnx"), "DmlExecutionProvider"), fp32))
        if "OpenVINOExecutionProvider" in ort_eps:
            b.append(("ort", "ONNX Runtime OpenVINO EP",
                      lambda: OrtRunner(ensure(fp32, "onnx"), "OpenVINOExecutionProvider", {"device_type": "CPU"}), fp32))
    if want("opencv"):
        b.append(("opencv", f"OpenCV {cv2.__version__} DNN CPU", lambda: OpenCVRunner(ensure(fp32, "onnx")), fp32))
        if cv_cuda:
            b.append(("opencv", "OpenCV DNN CUDA FP32", lambda: OpenCVRunner(ensure(fp32, "onnx"), "cuda"), fp32))
            b.append(("opencv", "OpenCV DNN CUDA FP16", lambda: OpenCVRunner(ensure(fp32, "onnx"), "cuda-fp16"), fp32))
    if want("tensorrt") and cuda:
        try:
            import tensorrt  # noqa: F401

            for q, tag in [(16, "fp16"), (8, "int8")]:
                dst = MODELS / f"{stem}_{tag}.engine"
                kw = {"device": 0} | ({"data": "coco128.yaml"} if q == 8 else {})
                b.append(("tensorrt", f"TensorRT {tag.upper()}",
                          (lambda d=dst, q=q, kw=kw: UltraRunner(ensure(d, "engine", q, **kw), "0")), dst))
        except ImportError:
            print("TensorRT не установлен: pip install tensorrt  (нужна NVIDIA GPU)")
    if want("openvino"):
        try:
            import openvino  # noqa: F401

            for q, tag in [(None, "fp32"), (8, "int8")]:
                dst = MODELS / f"{stem}_{tag}_openvino_model"
                kw = {"data": "coco128.yaml"} if q == 8 else {}
                b.append(("openvino", f"OpenVINO {tag.upper()}",
                          (lambda d=dst, q=q, kw=kw: UltraRunner(ensure(d, "openvino", q, **kw), "cpu")), dst))
        except ImportError:
            print("OpenVINO не установлен (pip install openvino) — пропускаю")
    if want("coreml") and platform.system() == "Darwin":
        dst = MODELS / f"{stem}.mlpackage"
        b.append(("coreml", "CoreML (ANE/GPU/CPU)", lambda: UltraRunner(ensure(dst, "coreml"), "cpu"), dst))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="видео (по умолчанию data/demo.mp4)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only", nargs="*", choices=["torch", "ort", "opencv", "tensorrt", "openvino", "coreml"])
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    weights = model_path(args.model)
    stem = Path(weights).stem
    frames = read_frames(args.source or default_video(), args.frames, stride=1)
    pre = [preprocess(f, args.imgsz) for f in frames]  # препроцессинг одинаковый для всех — считаем один раз
    with Timer() as t:
        for f in frames:
            preprocess(f, args.imgsz)
    pre_ms = t.ms / len(frames)
    print(f"Кадров: {len(frames)}, препроцессинг (letterbox+blob): {pre_ms:.2f} мс/кадр\n")

    rows, ref = [], None
    for group, title, factory, path in detect_backends(args, stem, weights):
        print(f"--> {title}")
        try:
            run = factory()
            for _ in range(args.warmup):
                run(pre[0][0])
            st_inf, st_post, dets = Stat(), Stat(), []
            for (blob, r, pad), f in zip(pre, frames):
                with Timer() as t:
                    out = run(blob)
                st_inf.add(t.ms)
                with Timer() as t:
                    d = postprocess(out, 0.25, 0.45, r, pad, f.shape[:2])
                st_post.add(t.ms)
                dets.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"    пропущен: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            continue
        if ref is None:
            ref = dets
        n_ref = sum(len(d[0]) for d in ref)
        agree = 100 * sum(count_matches(a, b) for a, b in zip(ref, dets)) / max(n_ref, 1)
        total = pre_ms + st_inf.mean + st_post.mean
        rows.append({"Бэкенд": title, "Инференс, мс": f"{st_inf.mean:.1f}", "p95, мс": f"{st_inf.p95:.1f}",
                     "Pre+Post, мс": f"{pre_ms + st_post.mean:.1f}", "FPS": f"{1000 / total:.1f}",
                     "Модель, МБ": f"{file_mb(path):.1f}" if Path(path).exists() else "",
                     "Совпадение с эталоном, %": f"{agree:.1f}", "_inf": st_inf.mean})
        print(f"    инференс {st_inf.mean:.1f} мс, FPS {1000 / total:.1f}, совпадение {agree:.1f}%")

    if not rows:
        raise SystemExit("Ни один бэкенд не запустился")
    base = rows[0]["_inf"]
    for r in rows:
        r["Ускорение инференса"] = f"x{base / r['_inf']:.2f}"
    cols = ["Бэкенд", "Инференс, мс", "p95, мс", "Ускорение инференса", "Pre+Post, мс", "FPS", "Модель, МБ",
            "Совпадение с эталоном, %"]
    import torch

    hw = cpu_name() + (f" + {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
    md = f"Железо: {hw}; {platform.system()} {platform.release()}; вход {args.imgsz}x{args.imgsz}, " \
         f"{len(frames)} кадров\n\n" + md_table(rows, cols)
    print("\n" + md)
    out_dir = RUNS / "05_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nТаблица: {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
