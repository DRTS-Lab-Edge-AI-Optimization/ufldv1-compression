"""Benchmark any variant (PyTorch module or ONNX file): accuracy + efficiency, logged to MLflow / TensorBoard."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import profiling as P
from .energy import measure_energy
from .evaluation import evaluate
from .runners import OrtRunner, TorchRunner
from .utils import ensure_dir

# columns shown in the summary table
SUMMARY_COLS = ["name", "group", "backend", "accuracy", "f1", "fp", "fn", "teacher_agreement",
                "params_total", "params_nonzero", "sparsity", "artifact_mb", "size_compressed_mb", "macs_g",
                "macs_effective_g", "bops_g", "lat_ms_t1", "lat_ms_t4", "fps_t4", "mem_runtime_mb",
                "mem_activation_mb", "energy_mj_per_inf", "energy_method"]


class Benchmarker:
    def __init__(self, cfg, test_loader, test_dataset, tracker=None, out_dir: str | None = None,
                 latency_threads=(1, 4, 6), latency_iters: int = 30, warmup: int = 5, energy_iters: int = 20,
                 do_energy: bool = True, do_memory: bool = True, do_profile: bool = True,
                 save_artifacts: bool = True, keep_artifacts: bool = False, log_model_max_mb: float = 100.0,
                 tb_hparams: bool = True):
        """save_artifacts: write a TorchScript file per variant (needed for artifact size + isolated memory).
        keep_artifacts:  keep those files in outputs/models (otherwise deleted after measuring; ~0.1-0.25 GB each).
        log_model_max_mb: models up to this size are also stored in MLflow (ONNX files are always kept on disk)."""
        self.cfg = cfg
        self.loader, self.dataset = test_loader, test_dataset
        self.tracker = tracker
        self.out = ensure_dir(out_dir or Path(cfg.output_dir))
        self.models_dir = ensure_dir(self.out / "models")
        self.tb_dir = ensure_dir(self.out / "tensorboard")
        self.latency_threads, self.latency_iters, self.warmup = tuple(latency_threads), latency_iters, warmup
        self.energy_iters, self.do_energy, self.do_memory, self.do_profile = energy_iters, do_energy, do_memory, do_profile
        self.save_artifacts, self.keep_artifacts, self.tb_hparams = save_artifacts, keep_artifacts, tb_hparams
        self.log_model_max_mb = log_model_max_mb
        self.reference_argmax = None
        self.results = []

    # ------------------------------------------------------------------------------------------
    def run(self, name: str, model=None, onnx_path: str | None = None, group: str = "", params: dict | None = None,
            ref_model=None, is_reference: bool = False, bits: int | None = None,
            train_history: list | None = None, log_every: int = 10) -> dict:
        """Evaluate + profile one variant.

        model / onnx_path   what to benchmark (one of them)
        ref_model           float/pruned torch model the ONNX file came from (for params/MACs)
        is_reference        True for the uncompressed teacher (stores its predictions for 'teacher_agreement')
        bits                weight/activation bit-width when it can't be read from the modules (ONNX INT8 -> 8)
        train_history       Trainer.fit() output -> training curves are stored in the same MLflow run
        """
        cfg = self.cfg
        shape = cfg.input_shape
        params = dict(params or {})
        row = {"name": name, "group": group, "backend": "onnxruntime" if onnx_path else "pytorch"}
        t_start = time.perf_counter()
        torch.set_num_threads(cfg.num_threads)
        runner = OrtRunner(onnx_path, cfg.num_threads) if onnx_path else TorchRunner(model, cfg.num_threads)
        tags = {"group": group, "backend": row["backend"], "quant_engine": getattr(model, "qengine", "") or ""}
        ctx = self.tracker.run(name, params={**params, "eval_images": len(self.dataset)}, tags=tags) \
            if self.tracker else _null()
        with ctx:
            # ---- accuracy -------------------------------------------------------------------
            ev = evaluate(runner, self.loader, self.dataset, cfg, reference_argmax=self.reference_argmax,
                          collect_argmax=is_reference)
            if is_reference:
                self.reference_argmax = ev.pop("argmax")
            row.update(ev)
            # ---- static stats ---------------------------------------------------------------
            stat_model = model if model is not None else ref_model
            if stat_model is not None:
                row.update(P.param_stats(stat_model))
                row.update(P.count_macs(stat_model, shape))
                if model is not None:
                    row.update(P.state_dict_size(model))
            if onnx_path:
                row["artifact_mb"] = P.file_size_mb(onnx_path)
                row["artifact_path"] = str(onnx_path)
            if bits is not None and "macs_g" in row:
                row["bops_g"] = row["macs_g"] * bits * bits
            # ---- latency / energy / memory ---------------------------------------------------
            row.update(P.measure_latency(runner, shape, self.latency_threads, self.warmup, self.latency_iters))
            runner.set_threads(cfg.num_threads)
            x = torch.randn(*shape)
            x_in = x.numpy() if onnx_path else x
            if self.do_energy:
                try:
                    row.update(measure_energy(runner, x_in, self.energy_iters))
                except Exception as e:
                    print(f"[{name}] energy skipped: {e}")
            try:
                row.update(P.inference_memory_inprocess(runner, shape))
            except Exception as e:
                print(f"[{name}] in-process memory skipped: {e}")
            artifact = onnx_path
            if model is not None and self.save_artifacts:
                artifact = str(self.models_dir / f"{_safe(name)}.pt")
                P.save_torchscript(model, artifact, shape)
                row["artifact_mb"] = P.file_size_mb(artifact)
                row["artifact_path"] = artifact
            if self.do_memory and artifact:
                try:
                    row.update(P.isolated_memory(artifact, "onnx" if onnx_path else "torch",
                                                 threads=4, qengine=getattr(model, "qengine", None), input_shape=shape))
                except Exception as e:
                    print(f"[{name}] isolated memory skipped: {str(e)[-300:]}")
            # ---- torch.profiler -> TensorBoard --------------------------------------------
            if self.do_profile and model is not None:
                try:
                    prof = P.torch_profile(model, str(self.tb_dir / "profile" / _safe(name)), shape)
                    row["profiler_gflops"] = prof["profiler_gflops"]
                    (self.out / "profiles").mkdir(exist_ok=True)
                    tfile = self.out / "profiles" / f"{_safe(name)}.txt"
                    tfile.write_text(prof["table"])
                    if self.tracker:
                        self.tracker.log_artifact(tfile, "profiler")
                except Exception as e:
                    print(f"[{name}] profiler skipped: {e}")
            row["bench_seconds"] = time.perf_counter() - t_start
            # ---- logging ---------------------------------------------------------------------
            if self.tracker:
                self.tracker.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
                self.tracker.set_tags({"energy_method": row.get("energy_method", "")})
                if artifact and row.get("artifact_mb", 1e9) <= self.log_model_max_mb:
                    self.tracker.log_artifact(artifact, "model")
                if train_history:
                    for step in range(log_every, len(train_history) + 1, log_every):
                        win = train_history[step - log_every:step]
                        avg = {f"train/{k}": sum(h.get(k, 0.0) for h in win) / len(win) for k in win[-1]}
                        self.tracker.log_metrics(avg, step=step)
                self.tracker.log_dict(_jsonable(row), "summary.json")
        if model is not None and artifact and not self.keep_artifacts:
            Path(artifact).unlink(missing_ok=True)
            row["artifact_path"] = None
        if self.tb_hparams:
            self._tb_hparams(name, params, row)
        self.results = [r for r in self.results if r["name"] != name] + [row]
        print(_short(row))
        return row

    # ------------------------------------------------------------------------------------------
    def _tb_hparams(self, name, params, row):
        try:
            from torch.utils.tensorboard import SummaryWriter

            hp = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
            hp["group"] = row["group"]
            hp["backend"] = row["backend"]
            metrics = {f"hparam/{k}": float(row[k]) for k in ("accuracy", "f1", "lat_ms_t4", "artifact_mb",
                                                              "macs_g", "energy_mj_per_inf") if k in row}
            with SummaryWriter(str(self.tb_dir / "hparams" / _safe(name))) as w:
                w.add_hparams(hp, metrics, run_name=".")
        except Exception:
            pass

    def table(self, cols=None):
        import pandas as pd

        df = pd.DataFrame(self.results)
        if df.empty:
            return df
        base = df[df["name"] == self.results[0]["name"]].iloc[0]
        if "accuracy" in df:
            df["acc_drop"] = base["accuracy"] - df["accuracy"]
        if "lat_ms_t4" in df:
            df["speedup_t4"] = base["lat_ms_t4"] / df["lat_ms_t4"]
        if "artifact_mb" in df and pd.notna(base.get("artifact_mb")):
            df["compression_x"] = base["artifact_mb"] / df["artifact_mb"]
        cols = cols or [c for c in SUMMARY_COLS + ["acc_drop", "speedup_t4", "compression_x"] if c in df]
        return df[cols]

    def save(self, name="results"):
        df = self.table(cols=None)
        import pandas as pd

        full = pd.DataFrame(self.results)
        full.to_csv(self.out / f"{name}_full.csv", index=False)
        df.to_csv(self.out / f"{name}.csv", index=False)
        return self.out / f"{name}.csv"

    def pareto_plot(self, x="lat_ms_t4", y="accuracy", logx=False, ax=None, title=None):
        import matplotlib.pyplot as plt

        df = self.table(cols=None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = ax.figure
        for g, sub in df.groupby("group"):
            ax.scatter(sub[x], sub[y], label=g or "baseline", s=50)
            for _, r in sub.iterrows():
                ax.annotate(r["name"], (r[x], r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        # Pareto front (lower x, higher y)
        pts = df[[x, y]].dropna().sort_values(x).values
        front, best = [], -np.inf
        for px, py in pts:
            if py > best:
                front.append((px, py))
                best = py
        if front:
            fx, fy = zip(*front)
            ax.plot(fx, fy, "k--", lw=1, alpha=0.6, label="Pareto front")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(title or f"{y} vs {x}")
        fig.tight_layout()
        return fig


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _jsonable(d):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in d.items()}


def _short(r):
    f = lambda k, fmt: (fmt.format(r[k]) if k in r and r[k] is not None else "-")
    return (f"[{r['name']}] acc={f('accuracy', '{:.4f}')} f1={f('f1', '{:.4f}')} agree={f('teacher_agreement', '{:.3f}')} "
            f"params={f('params_total', '{:,}')} size={f('artifact_mb', '{:.1f}')}MB GMACs={f('macs_g', '{:.2f}')} "
            f"lat@4t={f('lat_ms_t4', '{:.1f}')}ms E={f('energy_mj_per_inf', '{:.0f}')}mJ "
            f"mem={f('mem_runtime_mb', '{:.0f}')}MB")
