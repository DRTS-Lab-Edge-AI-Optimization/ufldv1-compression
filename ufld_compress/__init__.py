"""UFLD-Compress: a compression toolkit for Ultra-Fast-Lane-Detection v1 (cfzd), TuSimple / ResNet-18.

Sub-modules (imported lazily so that `lane_utils` works on a Raspberry Pi without PyTorch):

    config        - experiment configuration
    lane_utils    - numpy-only post-processing, TuSimple LaneEval, label generation
    model         - UFLD network (checkpoint compatible with tusimple_18.pth) + student backbones
    data          - TuSimple datasets (labels generated on-the-fly from the json files)
    runners       - unified inference wrappers (PyTorch / ONNX Runtime)
    evaluation    - TuSimple accuracy / FP / FN / F1 + teacher agreement
    profiling     - params, size, MACs/FLOPs/BOPs, latency, memory, torch.profiler
    energy        - energy meters (Linux RAPL, Raspberry Pi 5 PMIC, estimation)
    quantization  - PyTorch dynamic / static-FX PTQ / QAT, ONNX Runtime dynamic & static INT8
    pruning       - unstructured (one-shot, AGP) and structured channel pruning + BN recalibration
    distillation  - KD losses (logit, location, feature/FitNet, attention transfer) + Trainer
    tracking      - MLflow wrapper
    benchmark     - evaluate + profile + log any model variant, results table and Pareto plots
"""

__version__ = "0.1.0"
