#!/usr/bin/env python3
"""Benchmark UFLD ONNX models on a Raspberry Pi 5 (or any CPU) with ONNX Runtime - no PyTorch needed.

Measures latency (per thread count), FPS, energy per inference (Pi 5 PMIC via vcgencmd, RAPL on x86 Linux,
else estimate), process memory, optional TuSimple accuracy, and logs everything to an MLflow server.

Examples
  python scripts/pi5_benchmark.py --models deploy/*.onnx --threads 1 2 4 --iters 100
  python scripts/pi5_benchmark.py --models deploy/*.onnx --data-root ~/tusimple --eval-images 300 \
         --mlflow-uri http://192.168.1.10:5000 --experiment ufld-tusimple-compression
  python scripts/pi5_benchmark.py --models deploy/ufld_pruned_medium_int8_percentile.onnx --demo road.mp4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ufld_compress.energy import EnergyMeter  # noqa: E402
from ufld_compress.lane_utils import LaneEval, logits_to_lanes  # noqa: E402
from ufld_compress.runners import OrtRunner  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
H_SAMPLES = list(range(160, 720, 10))


def preprocess_pil(img, size=(800, 288)):
    """Same as torchvision Resize((288, 800)) + ToTensor + Normalize on a PIL image."""
    from PIL import Image

    img = img.convert("RGB").resize(size, Image.BILINEAR)
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


def device_name():
    p = Path("/proc/device-tree/model")
    if p.exists():
        return p.read_text(errors="ignore").strip("\x00 \n")
    return platform.processor() or platform.machine()


def soc_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return None


def bench_model(path, args):
    import psutil

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    runner = OrtRunner(path, max(args.threads))
    rss_loaded = proc.memory_info().rss
    x = np.random.randn(1, 3, 288, 800).astype(np.float32)
    res = {"model": Path(path).name, "file_mb": os.path.getsize(path) / 2**20,
           "mem_model_mb": (rss_loaded - rss0) / 2**20}
    for t in args.threads:
        runner.set_threads(t)
        for _ in range(args.warmup):
            runner(x)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            runner(x)
            times.append((time.perf_counter() - t0) * 1000)
        a = np.array(times)
        res.update({f"lat_ms_t{t}": a.mean(), f"lat_p90_ms_t{t}": np.percentile(a, 90), f"fps_t{t}": 1000 / a.mean()})
    runner.set_threads(max(args.threads))
    with EnergyMeter(p_idle_w=args.p_idle, p_max_w=args.p_max) as em:
        for _ in range(args.energy_iters):
            runner(x)
    res.update(energy_mj_per_inf=1000 * em.joules / args.energy_iters, avg_power_w=em.avg_power_w,
               energy_method=em.method, cpu_util=em.cpu_util)
    res["mem_process_peak_mb"] = proc.memory_info().rss / 2**20
    res["soc_temp_c"] = soc_temp()
    if args.data_root:
        res.update(evaluate(runner, args))
    return res


def evaluate(runner, args):
    from PIL import Image

    items = [json.loads(l) for l in open(os.path.join(args.data_root, "test_label.json")) if l.strip()]
    if args.eval_images:
        rng = np.random.RandomState(0)
        idx = sorted(rng.permutation(len(items))[:args.eval_images])
        items = [items[i] for i in idx]
    acc = fp = fn = 0.0
    for it in items:
        x = preprocess_pil(Image.open(os.path.join(args.data_root, it["raw_file"])))
        lanes = logits_to_lanes(runner(x)[0], 100, 1280)
        a, p, n = LaneEval.bench(lanes, it["lanes"], it["h_samples"], 10)
        acc, fp, fn = acc + a, fp + p, fn + n
    n = max(len(items), 1)
    return {"accuracy": acc / n, "fp": fp / n, "fn": fn / n, "eval_images": n}


def demo(path, video, args):
    import cv2
    from PIL import Image

    runner = OrtRunner(path, max(args.threads))
    cap = cv2.VideoCapture(0 if video == "camera" else video)
    out = None
    n, t_sum = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        x = preprocess_pil(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        t0 = time.perf_counter()
        logits = runner(x)[0]
        t_sum += time.perf_counter() - t0
        n += 1
        for lane in logits_to_lanes(logits, 100, w):
            for xx, yy in zip(lane, H_SAMPLES):
                if xx >= 0:
                    cv2.circle(frame, (xx, int(yy * h / 720)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"{1000 * t_sum / n:.0f} ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if out is None:
            out = cv2.VideoWriter(args.demo_out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        out.write(frame)
    if out is not None:
        out.release()
    print(f"demo: {n} frames, model {1000 * t_sum / max(n, 1):.1f} ms/frame -> {args.demo_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--energy-iters", type=int, default=50)
    ap.add_argument("--p-idle", type=float, default=2.7, help="estimate mode only (W)")
    ap.add_argument("--p-max", type=float, default=8.0, help="estimate mode only (W)")
    ap.add_argument("--data-root", default=None, help="TuSimple folder with clips/ and test_label.json")
    ap.add_argument("--eval-images", type=int, default=None)
    ap.add_argument("--mlflow-uri", default=None)
    ap.add_argument("--experiment", default="ufld-tusimple-compression")
    ap.add_argument("--csv", default="pi5_results.csv")
    ap.add_argument("--demo", default=None, help="video file or 'camera'")
    ap.add_argument("--demo-out", default="demo_out.mp4")
    args = ap.parse_args()

    paths = sorted({p for pat in args.models for p in glob.glob(pat) if not p.endswith(".pre.onnx")})
    if args.demo:
        demo(paths[0], args.demo, args)
        return
    dev = device_name()
    print(f"device: {dev} | models: {len(paths)}")
    rows = []
    for p in paths:
        r = bench_model(p, args)
        r["device"] = dev
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
        if args.mlflow_uri:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name=f"pi5::{Path(p).stem}"):
                mlflow.set_tags({"group": "pi5", "device": dev, "backend": "onnxruntime",
                                 "energy_method": r["energy_method"]})
                mlflow.log_params({"model_file": Path(p).name, "threads": ",".join(map(str, args.threads)),
                                   "iters": args.iters})
                mlflow.log_metrics({k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and v is not None})
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved", args.csv)


if __name__ == "__main__":
    main()
