"""Small helpers: threads, seeding, system information."""
from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np


def set_threads(n: int) -> None:
    """Set intra-op threads for PyTorch (and inter-op once, if still possible)."""
    import torch

    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # can only be set once, before any parallel work


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
            # Raspberry Pi / ARM boards expose the model elsewhere
            model = Path("/proc/device-tree/model")
            if model.exists():
                return model.read_text(errors="ignore").strip("\x00 \n")
        elif system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def system_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
    }
    try:
        import psutil

        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        info["cores_logical"] = os.cpu_count()
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["quant_engines"] = ",".join(e for e in torch.backends.quantized.supported_engines if e != "none")
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


def default_quant_backend() -> str:
    """'x86' on Intel/AMD, 'qnnpack' on ARM (Raspberry Pi)."""
    m = platform.machine().lower()
    return "qnnpack" if ("arm" in m or "aarch64" in m) else "x86"
