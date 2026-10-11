"""Model statistics: parameters, sparsity, size, MACs / FLOPs / BOPs, latency, memory, torch.profiler."""
from __future__ import annotations

import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------------
# parameters & size
# ---------------------------------------------------------------------------------------------


def _is_quantized_module(m) -> bool:
    return type(m).__module__.startswith(("torch.ao.nn.quantized", "torch.nn.quantized",
                                          "torch.ao.nn.intrinsic.quantized"))


def _module_param_tensors(m):
    """(tensors, quantized?) owned directly by module m. Handles quantized modules (packed weights)."""
    if _is_quantized_module(m):
        out = []
        for attr in ("weight", "bias"):
            f = getattr(m, attr, None)
            if callable(f):
                try:
                    t = f()
                    if t is not None:
                        out.append(t)
                except Exception:
                    pass
        return out, True
    return [p.detach() for p in m.parameters(recurse=False)], False


def _dense(t):
    return t.dequantize() if t.is_quantized else t


def param_stats(model: nn.Module) -> dict:
    total = nonzero = nbytes = 0
    for m in model.modules():
        tensors, _ = _module_param_tensors(m)
        for t in tensors:
            total += t.numel()
            nonzero += int(torch.count_nonzero(_dense(t)))
            nbytes += t.numel() * t.element_size()
    return {"params_total": total, "params_nonzero": nonzero,
            "sparsity": 1 - nonzero / max(total, 1), "param_mb": nbytes / 2**20}


def state_dict_size(model: nn.Module, compress: bool = True) -> dict:
    """Serialized size (MB) and zlib-compressed size (shows the storage gain of unstructured sparsity)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    res = {"size_mb": len(raw) / 2**20}
    if compress:
        res["size_compressed_mb"] = len(zlib.compress(raw, 1)) / 2**20
    return res


def file_size_mb(path) -> float:
    return os.path.getsize(path) / 2**20


# ---------------------------------------------------------------------------------------------
# MACs / FLOPs / BOPs (hook based; works for float, pruned and quantized modules)
# ---------------------------------------------------------------------------------------------


def _weight_density(m) -> float:
    w = getattr(m, "weight", None)
    try:
        w = w() if callable(w) else w
        if w is None:
            return 1.0
        w = _dense(w.detach())
        return float(torch.count_nonzero(w)) / w.numel()
    except Exception:
        return 1.0


def count_macs(model: nn.Module, input_shape=(1, 3, 288, 800)) -> dict:
    """MACs per image. `macs_effective` scales each layer by its weight density (theoretical sparse MACs).
    `bops` = MACs * w_bits * a_bits (32x32 for float, 8x8 for int8 layers)."""
    records = []

    def hook(m, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(o):
            return
        n_out = o.numel()
        if hasattr(m, "in_channels") and hasattr(m, "kernel_size") and hasattr(m, "groups"):
            k = int(np.prod(m.kernel_size)) if isinstance(m.kernel_size, tuple) else int(m.kernel_size) ** 2
            macs = n_out * (m.in_channels // m.groups) * k
        elif hasattr(m, "in_features") and hasattr(m, "out_features"):
            macs = n_out * m.in_features
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            macs = n_out
        else:
            return
        bits = 8 if _is_quantized_module(m) else 32
        records.append((macs, macs * _weight_density(m), macs * bits * bits))

    # hook every module: the hook only counts modules that look like conv / linear / BN
    # (quantized Linear is not a leaf - it owns a packed-params child - so we can't filter on leaves)
    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.randn(*input_shape))
    finally:
        for h in handles:
            h.remove()
    b = input_shape[0]
    macs = sum(r[0] for r in records) / b
    res = {"macs_g": macs / 1e9, "flops_g": 2 * macs / 1e9,
           "macs_effective_g": sum(r[1] for r in records) / b / 1e9,
           "bops_g": sum(r[2] for r in records) / b / 1e9}
    return res


def torch_flop_counter(model: nn.Module, input_shape=(1, 3, 288, 800)) -> float | None:
    """Cross-check with torch.utils.flop_counter (float models only). Returns GFLOPs or None."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.randn(*input_shape))
        return fc.get_total_flops() / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------------------------


def measure_latency(runner, input_shape=(1, 3, 288, 800), threads=(1, 4, 6), warmup: int = 5,
                    iters: int = 30) -> dict:
    """Latency (ms) for each thread count. 4 threads ~ Raspberry Pi 5 core count."""
    x = torch.randn(*input_shape)
    if getattr(runner, "backend", "") == "onnxruntime":
        x = x.numpy()
    res = {}
    old = getattr(runner, "num_threads", None)
    for t in threads:
        runner.set_threads(t)
        for _ in range(warmup):
            runner(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            runner(x)
            times.append((time.perf_counter() - t0) * 1000)
        a = np.array(times)
        res[f"lat_ms_t{t}"] = float(a.mean())
        res[f"lat_p90_ms_t{t}"] = float(np.percentile(a, 90))
        res[f"lat_std_ms_t{t}"] = float(a.std())
        res[f"fps_t{t}"] = 1000.0 / float(a.mean())
    if old:
        runner.set_threads(old)
    return res


# ---------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------


class _RssSampler:
    def __init__(self, period=0.001):
        import psutil

        self.p = psutil.Process()
        self.period = period

    def __enter__(self):
        gc.collect()
        self.base = self.p.memory_info().rss
        self.peak = self.base
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.p.memory_info().rss)
                time.sleep(self.period)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)


def inference_memory_inprocess(runner, input_shape=(1, 3, 288, 800), iters=3) -> dict:
    """Extra RSS while running inference (activations + workspace) in the current process."""
    x = torch.randn(*input_shape)
    if getattr(runner, "backend", "") == "onnxruntime":
        x = x.numpy()
    runner(x)
    with _RssSampler() as s:
        for _ in range(iters):
            runner(x)
    return {"mem_activation_mb": (s.peak - s.base) / 2**20}


_MEM_SCRIPT = r"""
import sys, json, time, threading, gc
import psutil
kind, path, threads, engine, shape = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], json.loads(sys.argv[5])
proc = psutil.Process()
peak = [0]
stop = threading.Event()
def loop():
    while not stop.is_set():
        peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.001)
import numpy as np
if kind == "onnx":
    import onnxruntime as ort
    gc.collect(); base = proc.memory_info().rss
    th = threading.Thread(target=loop, daemon=True); th.start()
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    loaded = proc.memory_info().rss
    x = np.random.randn(*shape).astype(np.float32); name = sess.get_inputs()[0].name
    for _ in range(5): sess.run(None, {name: x})
else:
    import torch
    torch.set_num_threads(threads)
    if engine != "none": torch.backends.quantized.engine = engine
    gc.collect(); base = proc.memory_info().rss
    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        m = torch.jit.load(path, map_location="cpu")
    except Exception:
        m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval(); loaded = proc.memory_info().rss
    x = torch.randn(*shape)
    with torch.inference_mode():
        for _ in range(5): m(x)
stop.set(); th.join()
p = max(peak[0], proc.memory_info().rss)
try:
    import resource
    p = max(p, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
except Exception:
    p = max(p, getattr(proc.memory_info(), "peak_wset", 0))
print("@@" + json.dumps({"mem_model_mb": (loaded - base) / 2**20, "mem_runtime_mb": (p - base) / 2**20,
                         "mem_process_peak_mb": p / 2**20}))
"""


def isolated_memory(path: str, kind: str = "torch", threads: int = 4, qengine: str | None = None,
                    input_shape=(1, 3, 288, 800), timeout: int = 300) -> dict:
    """Load a saved model in a fresh Python process and measure its memory footprint.

    mem_model_mb      RSS increase caused by loading the model (weights)
    mem_runtime_mb    peak RSS increase while loading + running 5 inferences (weights + activations + workspace)
    mem_process_peak_mb absolute peak RSS of that process (incl. the framework itself)
    """
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_MEM_SCRIPT)
        script = f.name
    try:
        out = subprocess.run([sys.executable, script, kind, str(path), str(threads), qengine or "none",
                              json.dumps(list(input_shape))], capture_output=True, text=True, env=env,
                             timeout=timeout)
        for line in out.stdout.splitlines():
            if line.startswith("@@"):
                return json.loads(line[2:])
        raise RuntimeError(out.stderr[-2000:])
    finally:
        os.unlink(script)


def save_torchscript(model: nn.Module, path: str, input_shape=(1, 3, 288, 800)) -> str:
    """TorchScript (traced) artifact; falls back to pickling the whole module."""
    model.eval()
    try:
        with torch.no_grad():
            ts = torch.jit.trace(model, torch.randn(*input_shape), check_trace=False)
        torch.jit.save(ts, path)
    except Exception:
        torch.save(model, path)
    return path


# ---------------------------------------------------------------------------------------------
# PyTorch profiler (TensorBoard / Chrome trace)
# ---------------------------------------------------------------------------------------------


def torch_profile(model: nn.Module, logdir: str, input_shape=(1, 3, 288, 800), active: int = 3,
                  row_limit: int = 20, record_shapes: bool = True) -> dict:
    """Runs torch.profiler; writes a trace for TensorBoard (tensorboard --logdir <logdir>) / Perfetto.

    Returns the op table (sorted by self CPU time), profiler-counted FLOPs and peak CPU memory of ops.
    """
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    Path(logdir).mkdir(parents=True, exist_ok=True)
    x = torch.randn(*input_shape)
    qengine = getattr(model, "qengine", None)
    if qengine:
        torch.backends.quantized.engine = qengine
    model.eval()
    with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait=1, warmup=1, active=active, repeat=1),
                 on_trace_ready=tensorboard_trace_handler(str(logdir)), record_shapes=record_shapes,
                 profile_memory=True, with_flops=True) as prof:
        for _ in range(active + 2):
            with torch.inference_mode():
                model(x)
            prof.step()
    ka = prof.key_averages()
    table = ka.table(sort_by="self_cpu_time_total", row_limit=row_limit)
    flops = sum(getattr(e, "flops", 0) or 0 for e in ka) / max(active, 1)
    top = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:5]
    return {"table": table, "profiler_gflops": flops / 1e9,
            "top_ops": {e.key: e.self_cpu_time_total / 1000 / max(active, 1) for e in top}}
