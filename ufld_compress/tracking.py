"""Thin MLflow wrapper (degrades to a no-op if mlflow is not installed).

UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db      (run from the folder that contains mlflow.db)
"""
from __future__ import annotations

import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, experiment: str = "ufld-compression", tracking_uri: str | None = None,
                 artifact_dir: str = "mlartifacts", enabled: bool = True):
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            warnings.warn("mlflow not installed -> tracking disabled (pip install mlflow)")
            self.enabled = False
            return
        self.mlflow = mlflow
        if tracking_uri is None:
            tracking_uri = "sqlite:///" + Path("mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            mlflow.create_experiment(experiment, artifact_location=Path(artifact_dir).resolve().as_uri())
        mlflow.set_experiment(experiment)
        self.tracking_uri = tracking_uri

    @contextmanager
    def run(self, name: str, params: dict | None = None, tags: dict | None = None):
        if not self.enabled:
            yield None
            return
        with self.mlflow.start_run(run_name=name) as r:
            if params:
                self.log_params(params)
            if tags:
                self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield r

    def log_params(self, params: dict):
        if not self.enabled:
            return
        flat = {k: (str(v)[:500]) for k, v in _flatten(params).items()}
        self.mlflow.log_params(flat)

    def log_metrics(self, metrics: dict, step: int | None = None):
        if not self.enabled or self.mlflow.active_run() is None:
            return
        clean = {}
        for k, v in metrics.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                clean[k.replace("@", "_")] = f
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path, artifact_path: str | None = None):
        if self.enabled and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path)

    def log_text(self, text: str, file: str):
        if self.enabled:
            self.mlflow.log_text(text, file)

    def log_dict(self, d: dict, file: str):
        if self.enabled:
            self.mlflow.log_text(json.dumps(d, indent=2, default=str), file)

    def log_figure(self, fig, file: str):
        if self.enabled:
            self.mlflow.log_figure(fig, file)

    def set_tags(self, tags: dict):
        if self.enabled and self.mlflow.active_run() is not None:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})
