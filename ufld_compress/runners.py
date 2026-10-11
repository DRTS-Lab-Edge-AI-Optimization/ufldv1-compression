"""Unified inference wrappers so every metric works the same way for PyTorch and ONNX Runtime models."""
from __future__ import annotations

import numpy as np


class TorchRunner:
    backend = "pytorch"

    def __init__(self, model, num_threads: int | None = None):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.qengine = getattr(model, "qengine", None)  # set by quantization.py on quantized models
        self.num_threads = num_threads
        self._activate()

    def _activate(self):
        if self.qengine:
            self.torch.backends.quantized.engine = self.qengine
        if self.num_threads:
            self.torch.set_num_threads(self.num_threads)

    def set_threads(self, n: int):
        self.num_threads = n
        self._activate()

    def __call__(self, x) -> np.ndarray:
        torch = self.torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.qengine and torch.backends.quantized.engine != self.qengine:
            torch.backends.quantized.engine = self.qengine
        with torch.inference_mode():
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.float().numpy()


class OrtRunner:
    backend = "onnxruntime"

    def __init__(self, onnx_path: str, num_threads: int | None = None):
        self.path = onnx_path
        self.num_threads = num_threads
        self._build()

    def _build(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if self.num_threads:
            so.intra_op_num_threads = int(self.num_threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def set_threads(self, n: int):
        if n != self.num_threads:
            self.num_threads = n
            self._build()

    def __call__(self, x) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        return self.sess.run(None, {self.input_name: np.ascontiguousarray(x, dtype=np.float32)})[0]


def make_runner(model=None, onnx_path=None, num_threads=None):
    if onnx_path is not None:
        return OrtRunner(onnx_path, num_threads)
    return TorchRunner(model, num_threads)
