# UFLD-Compress

A compression framework for **Ultra-Fast-Lane-Detection v1** ([cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)), using ResNet-18 and the TuSimple dataset.

Designed for deployment on the **Raspberry Pi 5**, with CPU inference support.

## Project Structure

```text
ufld_compress/
  config.py         Configuration (row anchors, dimensions, paths, thread count)
  model.py          UFLDNet compatible with tusimple_18.pth + MobileNetV2/V3 students
  data.py           TuSimple dataset (in-memory label generation, no convert_tusimple.py required) + synthetic data
  lane_utils.py     Post-processing and official TuSimple LaneEval (NumPy only; runs on Pi without PyTorch)
  evaluation.py     Accuracy / FP / FN / F1 + teacher-model consistency
  runners.py        Unified PyTorch and ONNX Runtime inference
  profiling.py      Parameters, sparsity, model size, MACs/FLOPs/BOPs, latency, isolated memory profiling, torch.profiler
  energy.py         Energy measurement: RAPL (Linux) / Raspberry Pi 5 PMIC / estimation
  quantization.py   Dynamic quantization, static PTQ (FX), QAT, ONNX Runtime dynamic/static quantization (MinMax/Entropy/Percentile)
  pruning.py        Unstructured pruning (global/layer/random), AGP, structured channel pruning (L1/L2/BN-γ/Taylor/random), BN recalibration
  distillation.py   Logit / Location / Feature (FitNets) / Attention Transfer knowledge distillation + trainer
  tracking.py       MLflow experiment tracking
  benchmark.py      Evaluation, profiling, experiment tracking, results tables, and Pareto plots

notebooks/
  ufld_compression.ipynb   All experiments

scripts/
  smoke_test.py            Quick test of all methods using synthetic data
  pi5_benchmark.py         ONNX benchmarking on Pi 5 (latency, PMIC energy, memory, accuracy, video demo) + MLflow
```

## Quick Start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run all smoke tests (approximately 5 minutes)
python scripts/smoke_test.py

# Launch the experiment notebook
jupyter lab notebooks/ufld_compression.ipynb

# Launch the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Launch TensorBoard
tensorboard --logdir outputs/tensorboard
```

### Dataset and Checkpoint

- **Dataset:** `data/tusimple/` — contains video clips, `label_data_*.json`, and `test_label.json`.
- **Pretrained weights:** `weights/tusimple_18.pth`.
##ONNX files
ONNX files for Quantized and Pruned models are available at: https://drive.google.com/open?id=1wXEp_KevI4iOxYSn1-5Jw0EGCei3s_fu&usp=drive_fs
## Direct API Usage

```python
from ufld_compress.config import Config
from ufld_compress import model as M, quantization as Q, pruning as PR, distillation as KD

# Configure the dataset and pretrained checkpoint
cfg = Config(
    data_root="data/tusimple",
    checkpoint="weights/tusimple_18.pth"
)

# Build the teacher model
teacher = M.build_teacher(cfg)

# Structurally prune the teacher model
student = PR.channel_prune(
    teacher,
    ratio=0.5,
    criterion="l1",
    fc_ratio=0.75
)

# Recover accuracy through knowledge distillation
KD.Trainer(
    student,
    train_loader,
    KD.TrainConfig(kd_weight=1, loc_weight=1),
    teacher=teacher
).fit()

# Export the student model to ONNX
onnx = Q.export_onnx(student, "out/student.onnx")

# Apply static INT8 quantization using percentile calibration
int8 = Q.ort_quantize_static(
    onnx,
    "out/student_int8.onnx",
    calib_batches,
    method="percentile"
)
```

## Key Features

- **Model Compression:** Unstructured and structured channel pruning.
- **Quantization:** Dynamic quantization, post-training quantization (PTQ), and quantization-aware training (QAT).
- **Knowledge Distillation:** Logit-based, location-based, feature-based, and attention-transfer distillation.
- **Evaluation:** Lane detection accuracy, false positives (FP), false negatives (FN), and F1 score.
- **Performance Profiling:** Model size, parameter count, sparsity, MACs/FLOPs/BOPs, latency, and memory usage.
- **Energy Measurement:** Raspberry Pi 5 PMIC readings, Linux RAPL, and estimation-based methods.
- **Experiment Tracking:** MLflow integration, benchmark comparisons, and Pareto-frontier visualization.
- **Edge Deployment:** ONNX Runtime inference and benchmarking on Raspberry Pi 5.
