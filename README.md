# UFLD-Compress

فریم‌ورک فشرده‌سازی **Ultra-Fast-Lane-Detection v1** ([cfzd](https://github.com/cfzd/Ultra-Fast-Lane-Detection)) — ResNet-18 / TuSimple —
برای استقرار روی **Raspberry Pi 5**، قابل اجرا روی CPU (طراحی‌شده برای Ryzen 5 5500U).

```
ufld_compress/
  config.py        پیکربندی (row anchorها، ابعاد، مسیرها، تعداد رشته‌ها)
  model.py         UFLDNet سازگار با tusimple_18.pth + دانش‌آموزهای MobileNetV2/V3
  data.py          دیتاست TuSimple (ساخت برچسب در حافظه، بدون convert_tusimple.py) + داده‌ی مصنوعی
  lane_utils.py    پس‌پردازش و LaneEval رسمی TuSimple (فقط NumPy؛ روی Pi بدون PyTorch کار می‌کند)
  evaluation.py    Accuracy / FP / FN / F1 + هم‌خوانی با مدل معلم
  runners.py       اجرای یکسان PyTorch و ONNX Runtime
  profiling.py     پارامتر، sparsity، اندازه، MACs/FLOPs/BOPs، latency، حافظه (ایزوله)، torch.profiler
  energy.py        انرژی: RAPL (لینوکس) / PMIC رزبری‌پای ۵ / تخمین
  quantization.py  Dynamic، Static PTQ (FX)، QAT، ONNX Runtime dynamic/static (MinMax/Entropy/Percentile)
  pruning.py       Unstructured (global/layer/random)، AGP، Structured channel (L1/L2/BN-γ/Taylor/random)، BN recalibration
  distillation.py  Logit / Location / Feature (FitNets) / Attention-Transfer KD + Trainer
  tracking.py      MLflow
  benchmark.py     ارزیابی + پروفایل + ثبت هر نسخه، جدول نتایج و نمودار Pareto
notebooks/ufld_compression.ipynb   همه‌ی آزمایش‌ها
scripts/smoke_test.py              تست سریع همه‌ی روش‌ها روی داده‌ی مصنوعی
scripts/pi5_benchmark.py           بنچمارک ONNX روی Pi 5 (latency، انرژی PMIC، حافظه، دقت، دمو ویدئو) + MLflow
```

## شروع سریع
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/smoke_test.py            # ~5 دقیقه، همه باید OK باشند
jupyter lab notebooks/ufld_compression.ipynb
mlflow ui --backend-store-uri sqlite:///mlflow.db
tensorboard --logdir outputs/tensorboard
```
داده: `data/tusimple/` (clips + `label_data_*.json` + `test_label.json`) — وزن: `weights/tusimple_18.pth`.

## استفاده‌ی مستقیم از API
```python
from ufld_compress.config import Config
from ufld_compress import model as M, quantization as Q, pruning as PR, distillation as KD
cfg = Config(data_root="data/tusimple", checkpoint="weights/tusimple_18.pth")
teacher = M.build_teacher(cfg)
student = PR.channel_prune(teacher, ratio=0.5, criterion="l1", fc_ratio=0.75)        # ساختاری
KD.Trainer(student, train_loader, KD.TrainConfig(kd_weight=1, loc_weight=1), teacher=teacher).fit()  # بازیابی با KD
onnx = Q.export_onnx(student, "out/student.onnx")
int8 = Q.ort_quantize_static(onnx, "out/student_int8.onnx", calib_batches, method="percentile")
```

## نکته‌ها
- **5500U:** `num_threads=6` (هسته‌ی فیزیکی)، `num_workers=2`. روی ویندوز RAPL در دسترس نیست و انرژی تخمینی است (`energy_method=estimate`)؛ روی لینوکس:
  `sudo chmod -R a+r /sys/class/powercap/intel-rapl` تا انرژی واقعی پکیج CPU خوانده شود.
- **Pi 5:** مسیر پیشنهادی ONNX Runtime + INT8 استاتیک روی مدل هرس‌شده است. برای PyTorch روی ARM موتور `qnnpack` استفاده می‌شود.
- `torch.ao.quantization` در PyTorch در وضعیت «phase-out» است (بدون قابلیت جدید) ولی همچنان کار می‌کند؛ مسیر ONNX Runtime به آن وابسته نیست.
- Latency روی x86 برای مقایسه‌ی نسبی است؛ عدد نهایی را با `scripts/pi5_benchmark.py` روی خود Pi بگیرید.
