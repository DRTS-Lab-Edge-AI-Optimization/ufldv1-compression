#!/usr/bin/env python3
"""Quick end-to-end check of every compression method on synthetic data with random weights (~3-6 min on a
Ryzen 5 5500U). Run this first after installing the requirements:   python scripts/smoke_test.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ufld_compress import data as D  # noqa: E402
from ufld_compress import distillation as KD  # noqa: E402
from ufld_compress import model as M  # noqa: E402
from ufld_compress import profiling as P  # noqa: E402
from ufld_compress import pruning as PR  # noqa: E402
from ufld_compress import quantization as Q  # noqa: E402
from ufld_compress.benchmark import Benchmarker  # noqa: E402
from ufld_compress.config import Config  # noqa: E402
from ufld_compress.utils import default_quant_backend, set_threads, system_info  # noqa: E402

OUT = Path(tempfile.mkdtemp(prefix="ufld_smoke_"))
cfg = Config(dry_run=True, output_dir=str(OUT), num_workers=0, num_threads=4)
set_threads(cfg.num_threads)
BACKEND = default_quant_backend()
train_ds, test_ds = D.build_datasets(cfg, train_subset=8, test_subset=4)
train_loader = D.make_loader(train_ds, 2, shuffle=True)
test_loader = D.make_loader(test_ds, 2)
calib = lambda n: D.images_only(D.make_loader(train_ds, 2), n)  # noqa: E731
teacher = M.build_teacher(cfg)
bench = Benchmarker(cfg, test_loader, test_ds, tracker=None, out_dir=str(OUT), latency_threads=(1, 4),
                    latency_iters=2, warmup=1, energy_iters=2)
results = {}


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        results[name] = f"OK   ({time.time() - t0:.1f}s)"
    except Exception as e:
        traceback.print_exc()
        results[name] = f"FAIL {type(e).__name__}: {str(e)[:150]}"
    print(f"{name:32s} {results[name]}", flush=True)


tc = KD.TrainConfig(max_steps=2, log_every=1, warmup_steps=1, kd_weight=1.0, loc_weight=1.0)

check("baseline benchmark", lambda: bench.run("fp32", teacher, group="baseline", is_reference=True))
check("dynamic int8", lambda: bench.run("dyn", Q.dynamic_quantize(teacher, backend=BACKEND)))
check("dynamic fp16", lambda: bench.run("dyn16", Q.dynamic_quantize(teacher, torch.float16, BACKEND)))
check("ptq fx (histogram, mixed)", lambda: bench.run("ptq", Q.static_quantize_fx(
    teacher, calib(2), 2, BACKEND, "histogram", True, ("model.conv1", "cls.2"))))
check("ptq fx (minmax per-tensor)", lambda: bench.run("ptq2", Q.static_quantize_fx(
    teacher, calib(2), 2, BACKEND, "minmax", False)))


def qat():
    q = Q.prepare_qat(teacher, BACKEND)
    KD.Trainer(q, train_loader, tc, teacher=teacher, callbacks=[Q.qat_step_callback(None, 1)]).fit()
    bench.run("qat", Q.convert_qat(q))


check("qat + kd", qat)


def ort():
    f = Q.export_onnx(teacher, str(OUT / "t.onnx"))
    bench.run("ort_fp32", onnx_path=f, ref_model=teacher)
    bench.run("ort_dyn", onnx_path=Q.ort_quantize_dynamic(f, str(OUT / "d.onnx")), ref_model=teacher)
    for m in ["minmax", "percentile"]:
        bench.run(f"ort_{m}", onnx_path=Q.ort_quantize_static(f, str(OUT / f"s_{m}.onnx"), calib(2), 2, method=m),
                  ref_model=teacher, bits=8)


check("onnx export + ort quant", ort)
check("unstructured global", lambda: bench.run("ug", PR.unstructured_prune(teacher, 0.5)))
check("unstructured layer/random", lambda: (PR.unstructured_prune(teacher, 0.5, scope="layer"),
                                            PR.unstructured_prune(teacher, 0.5, method="random")))


def structured():
    for crit in ["l1", "l2", "bn_gamma", "random"]:
        m = PR.channel_prune(teacher, 0.5, crit, prune_residual=True, fc_ratio=0.75,
                             layer_ratios={"model.stage": 0.25})
        PR.recalibrate_bn(m, calib(2), 2)
    src = copy.deepcopy(teacher)
    ts = PR.compute_taylor_scores(src, calib(2), lambda mm, b: PR.pseudo_label_loss(mm, b), 2)
    m = PR.channel_prune(src, 0.5, "taylor", prune_residual=True, fc_ratio=0.5, strategy="global", taylor_scores=ts)
    print(m.prune_report)
    KD.Trainer(m, train_loader, KD.TrainConfig(max_steps=2, log_every=1, warmup_steps=1, kd_weight=1,
                                               feat_weight=1, at_weight=100), teacher=teacher).fit()
    m2 = m
    bench.run("struct", m2)
    f = Q.export_onnx(m2, str(OUT / "p.onnx"))
    bench.run("struct_ort", onnx_path=Q.ort_quantize_static(f, str(OUT / "ps.onnx"), calib(2), 2), ref_model=m2, bits=8)
    bench.run("struct_ptq", Q.static_quantize_fx(m2, calib(2), 2, BACKEND))


check("structured + kd(feat, at) + int8", structured)


def agp():
    m = copy.deepcopy(teacher)
    pr = PR.AGPPruner(m, 0.8, 0, 3, 1)
    KD.Trainer(m, train_loader, KD.TrainConfig(max_steps=4, log_every=2, warmup_steps=1, kd_weight=1), teacher,
               callbacks=[pr]).fit()
    PR.make_pruning_permanent(m)
    print("agp sparsity", PR.sparsity_report(m)["__global__"], pr.history)


check("agp + kd", agp)


def student():
    for arch in ["mobilenet_v2", "mobilenet_v3_large"]:
        s = M.build_student(cfg, arch, pretrained=False)
        KD.Trainer(s, train_loader, KD.TrainConfig(max_steps=1, log_every=1, warmup_steps=1, kd_weight=1,
                                                   feat_weight=1, at_weight=10), teacher).fit()
        P.count_macs(s)


check("mobilenet students", student)
check("profiler", lambda: print(P.torch_profile(teacher, str(OUT / "prof"), active=1)["profiler_gflops"]))
check("results table", lambda: print(bench.table().round(3).to_string()))

print("\n==== summary ====")
print(system_info())
for k, v in results.items():
    print(f"{k:32s} {v}")
print("outputs in", OUT)
sys.exit(int(any(v.startswith("FAIL") for v in results.values())))
