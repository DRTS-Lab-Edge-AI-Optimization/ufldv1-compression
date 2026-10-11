"""Quantization methods.

PyTorch (eager / FX graph mode, CPU kernels: x86 = fbgemm/onednn on the Ryzen, qnnpack on the Raspberry Pi):
  1. dynamic_quantize      - weight-only INT8 (or FP16) for the huge FC head, activations quantized at runtime
  2. static_quantize_fx    - post-training static INT8 (calibration; observer / per-channel / mixed precision options)
  3. prepare_qat / convert_qat - quantization-aware training (fine-tune with fake-quant, optionally with KD)

ONNX Runtime (recommended deployment path for the Raspberry Pi 5):
  4. export_onnx + ort_quantize_dynamic  - INT8 weights for MatMul/Gemm
  5. export_onnx + ort_quantize_static   - full INT8 QDQ, calibration = MinMax / Entropy / Percentile

Note: torch.ao.quantization is in "phase-out" (no new features) but remains available; warnings are silenced.
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao")


def _check_engine(backend: str):
    engines = torch.backends.quantized.supported_engines
    if backend not in engines:
        raise RuntimeError(f"quantized engine '{backend}' not available here (have {engines})")
    torch.backends.quantized.engine = backend


# ---------------------------------------------------------------------------------------------
# 1) dynamic quantization
# ---------------------------------------------------------------------------------------------


def dynamic_quantize(model: nn.Module, dtype=torch.qint8, backend: str = "x86") -> nn.Module:
    """Quantize nn.Linear layers. For UFLD the head holds ~50M of the ~61M parameters, so this alone
    shrinks the model ~3x with almost no accuracy loss. dtype=torch.float16 gives FP16 weight storage."""
    from torch.ao.quantization import quantize_dynamic

    _check_engine(backend)
    qm = quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=dtype)
    qm.qengine = backend
    return qm


# ---------------------------------------------------------------------------------------------
# 2) static post-training quantization (FX graph mode)
# ---------------------------------------------------------------------------------------------


def make_qconfig_mapping(backend: str = "x86", act_observer: str = "histogram", per_channel: bool = True,
                         skip_modules=()):
    from torch.ao.quantization import QConfig, get_default_qconfig_mapping
    from torch.ao.quantization.observer import (HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
                                                PerChannelMinMaxObserver)

    reduce_range = backend in ("x86", "fbgemm")  # 7-bit activations avoid overflow on CPUs without VNNI
    act_cls = {"minmax": MinMaxObserver, "histogram": HistogramObserver,
               "moving_avg": MovingAverageMinMaxObserver}[act_observer]
    act = act_cls.with_args(dtype=torch.quint8, reduce_range=reduce_range)
    if per_channel:
        wt = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    else:
        wt = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    mapping = get_default_qconfig_mapping(backend)
    mapping.set_global(QConfig(activation=act, weight=wt))
    for name in skip_modules:  # mixed precision: keep these modules in FP32
        mapping.set_module_name(name, None)
    return mapping


@torch.no_grad()
def calibrate(model, batches, max_batches: int | None = None):
    model.eval()
    for i, x in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        model(x)


def static_quantize_fx(model: nn.Module, calib_batches, num_batches: int = 16, backend: str = "x86",
                       act_observer: str = "histogram", per_channel: bool = True, skip_modules=(),
                       input_shape=(1, 3, 288, 800)) -> nn.Module:
    """Post-training static INT8 quantization (conv+bn+relu fused automatically by FX)."""
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    _check_engine(backend)
    m = copy.deepcopy(model).eval()
    mapping = make_qconfig_mapping(backend, act_observer, per_channel, skip_modules)
    prepared = prepare_fx(m, mapping, (torch.randn(*input_shape),))
    calibrate(prepared, calib_batches, num_batches)
    qm = convert_fx(prepared)
    qm.qengine = backend
    return qm


# ---------------------------------------------------------------------------------------------
# 3) quantization-aware training (FX)
# ---------------------------------------------------------------------------------------------


def prepare_qat(model: nn.Module, backend: str = "x86", input_shape=(1, 3, 288, 800)) -> nn.Module:
    """Insert fake-quant modules (x86: per-channel weights, qnnpack: per-tensor). Train it with distillation.Trainer."""
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    _check_engine(backend)
    m = copy.deepcopy(model).train()
    qat = prepare_qat_fx(m, get_default_qat_qconfig_mapping(backend), (torch.randn(*input_shape),))
    qat.qengine_target = backend
    return qat


def qat_step_callback(freeze_bn_step: int | None = None, freeze_observer_step: int | None = None):
    """Trainer callback: freeze BN statistics / observers late in QAT (standard recipe)."""
    from torch.ao.quantization import disable_observer

    try:
        from torch.ao.nn.intrinsic.qat import freeze_bn_stats
    except ImportError:  # older layouts
        from torch.nn.intrinsic.qat import freeze_bn_stats

    def cb(model, step):
        if freeze_bn_step is not None and step == freeze_bn_step:
            model.apply(freeze_bn_stats)
        if freeze_observer_step is not None and step == freeze_observer_step:
            model.apply(disable_observer)

    return cb


def convert_qat(qat_model: nn.Module) -> nn.Module:
    from torch.ao.quantization.quantize_fx import convert_fx

    backend = getattr(qat_model, "qengine_target", "x86")
    _check_engine(backend)
    qm = convert_fx(copy.deepcopy(qat_model).eval())
    qm.qengine = backend
    return qm


# ---------------------------------------------------------------------------------------------
# 4/5) ONNX export + ONNX Runtime quantization
# ---------------------------------------------------------------------------------------------


def export_onnx(model: nn.Module, path: str, input_shape=(1, 3, 288, 800), opset: int = 17,
                dynamic_batch: bool = True) -> str:
    """Export a float (optionally pruned) model. Quantize the ONNX file afterwards with ORT."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).eval()
    x = torch.randn(*input_shape)
    kw = dict(input_names=["input"], output_names=["cls"], opset_version=opset, do_constant_folding=True)
    if dynamic_batch:
        kw["dynamic_axes"] = {"input": {0: "batch"}, "cls": {0: "batch"}}
    with torch.no_grad():
        try:
            torch.onnx.export(m, (x,), path, dynamo=False, **kw)  # classic TorchScript exporter
        except TypeError:  # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(m, (x,), path, **kw)
        except Exception as e:  # legacy exporter removed -> dynamo exporter (needs onnxscript)
            warnings.warn(f"legacy ONNX export failed ({e}); trying dynamo exporter")
            kw.pop("dynamic_axes", None)
            torch.onnx.export(m, (x,), path, dynamo=True, input_names=["input"], output_names=["cls"],
                              opset_version=max(opset, 18))
    try:
        import onnx

        onnx.checker.check_model(path)
    except ImportError:
        pass
    return path


def make_calibration_reader(batches, input_name: str = "input", max_batches: int = 16):
    """ORT CalibrationDataReader built from an iterable of image tensors."""
    from onnxruntime.quantization import CalibrationDataReader

    data = []
    for i, x in enumerate(batches):
        if i >= max_batches:
            break
        data.append({input_name: x.numpy() if hasattr(x, "numpy") else x})

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(data)

        def get_next(self):
            return next(self._it, None)

        def rewind(self):
            self._it = iter(data)

    return _Reader()


def onnx_node_names(path: str, op_types=("Conv", "Gemm", "MatMul")) -> list:
    import onnx

    g = onnx.load(path).graph
    return [n.name for n in g.node if n.op_type in op_types]


def ort_quantize_dynamic(fp32_path: str, out_path: str, per_channel: bool = False,
                         op_types=("MatMul", "Gemm")) -> str:
    """INT8 weights for the FC head (dynamic quantization of convs is not recommended in ORT)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=fp32_path, model_output=out_path, per_channel=per_channel,
                     weight_type=QuantType.QInt8, op_types_to_quantize=list(op_types))
    return out_path


def ort_quantize_static(fp32_path: str, out_path: str, calib_batches, num_batches: int = 16,
                        method: str = "minmax", per_channel: bool = True, reduce_range: bool = False,
                        quant_format: str = "qdq", exclude_first_last: bool = False,
                        percentile: float = 99.999, preprocess: bool = True) -> str:
    """Full INT8 (QDQ) quantization. method: minmax | entropy | percentile.

    reduce_range=True can help on AVX2 CPUs without VNNI (e.g. Ryzen 5500U) if accuracy drops due to
    U8S8 saturation; on the Raspberry Pi 5 (ARM) keep False.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    src = fp32_path
    tmpdir = tempfile.mkdtemp(prefix="ufld_ortq_")
    if preprocess:
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            pre = os.path.join(tmpdir, "pre.onnx")
            quant_pre_process(fp32_path, pre)
            src = pre
        except Exception as e:
            warnings.warn(f"quant_pre_process skipped: {e}")
    reader = make_calibration_reader(calib_batches, "input", num_batches)
    exclude = []
    if exclude_first_last:
        names = onnx_node_names(src, ("Conv", "Gemm", "MatMul"))
        exclude = [names[0], names[-1]] if names else []
    quantize_static(
        model_input=src, model_output=out_path, calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator,
        per_channel=per_channel, reduce_range=reduce_range,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method={"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                          "percentile": CalibrationMethod.Percentile}[method],
        nodes_to_exclude=exclude,
        extra_options={"CalibPercentile": percentile} if method == "percentile" else {},
    )
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path
