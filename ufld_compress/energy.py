"""Energy measurement.

Backends (chosen automatically, `method` tells you which one produced the number):

* ``rapl``     Linux powercap package energy counter. Works on AMD Zen (Ryzen 5 5500U) with kernel >= 5.11.
               Needs read permission:  sudo chmod -R a+r /sys/class/powercap/intel-rapl
* ``pi5_pmic`` Raspberry Pi 5: samples the PMIC rails via ``vcgencmd pmic_read_adc`` (sum of V*I).
* ``estimate`` Anywhere else (e.g. Windows): P = P_idle + (P_max - P_idle) * cpu_utilisation.
               Defaults are for a 15 W TDP laptop CPU; values are approximate and labelled as such.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import threading
import time


class RaplMeter:
    def __init__(self):
        zones = sorted(glob.glob("/sys/class/powercap/*rapl*:[0-9]"))
        self.zone = None
        for z in zones:
            try:
                name = open(os.path.join(z, "name")).read().strip()
                if name.startswith("package"):
                    open(os.path.join(z, "energy_uj")).read()
                    self.zone = z
                    break
            except (OSError, PermissionError):
                continue
        if self.zone is None:
            raise RuntimeError("RAPL not available/readable")
        try:
            self.max_uj = int(open(os.path.join(self.zone, "max_energy_range_uj")).read())
        except OSError:
            self.max_uj = 2**32

    def read_j(self) -> float:
        return int(open(os.path.join(self.zone, "energy_uj")).read()) / 1e6

    def delta(self, e0, e1):
        d = e1 - e0
        return d + self.max_uj / 1e6 if d < 0 else d


class Pi5PmicSampler:
    """Background sampler of the Raspberry Pi 5 PMIC (vcgencmd pmic_read_adc)."""

    _re = re.compile(r"^\s*(\S+)_(A|V)\s+(?:current|volt)\(\d+\)=([0-9.]+)")

    def __init__(self, period_s: float = 0.05):
        if shutil.which("vcgencmd") is None:
            raise RuntimeError("vcgencmd not found")
        self.period = period_s
        if self.power_w() <= 0:
            raise RuntimeError("PMIC readings unavailable")

    @classmethod
    def power_w(cls) -> float:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
        amps, volts = {}, {}
        for line in out.splitlines():
            m = cls._re.match(line)
            if m:
                (amps if m.group(2) == "A" else volts)[m.group(1)] = float(m.group(3))
        return sum(a * volts[k] for k, a in amps.items() if k in volts)

    def start(self):
        self._samples, self._stop = [], threading.Event()

        def loop():
            while not self._stop.is_set():
                self._samples.append((time.perf_counter(), self.power_w()))
                time.sleep(self.period)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> float:
        """Returns the average power (W) over the sampling window."""
        self._stop.set()
        self._t.join()
        if not self._samples:
            return 0.0
        return sum(p for _, p in self._samples) / len(self._samples)


class EnergyMeter:
    """Usage:  with EnergyMeter() as em: ...run N inferences...;  em.joules, em.method, em.avg_power_w"""

    def __init__(self, method: str = "auto", p_idle_w: float = 3.0, p_max_w: float = 15.0):
        self.p_idle_w, self.p_max_w = p_idle_w, p_max_w
        self.method = None
        self.backend = None
        if method in ("auto", "rapl"):
            try:
                self.backend, self.method = RaplMeter(), "rapl"
            except Exception:
                if method == "rapl":
                    raise
        if self.backend is None and method in ("auto", "pi5_pmic"):
            try:
                self.backend, self.method = Pi5PmicSampler(), "pi5_pmic"
            except Exception:
                if method == "pi5_pmic":
                    raise
        if self.backend is None:
            self.method = "estimate"

    def __enter__(self):
        import psutil

        self._proc_cpu = psutil.cpu_percent(interval=None)  # reset the system-wide counter
        self.t0 = time.perf_counter()
        if self.method == "rapl":
            self.e0 = self.backend.read_j()
        elif self.method == "pi5_pmic":
            self.backend.start()
        return self

    def __exit__(self, *exc):
        import psutil

        self.seconds = time.perf_counter() - self.t0
        self.cpu_util = psutil.cpu_percent(interval=None) / 100.0
        if self.method == "rapl":
            self.joules = self.backend.delta(self.e0, self.backend.read_j())
        elif self.method == "pi5_pmic":
            self.joules = self.backend.stop() * self.seconds
        else:
            p = self.p_idle_w + (self.p_max_w - self.p_idle_w) * self.cpu_util
            self.joules = p * self.seconds
        self.avg_power_w = self.joules / max(self.seconds, 1e-9)
        return False


def measure_energy(runner, x, iters: int = 30, warmup: int = 3, **meter_kwargs) -> dict:
    for _ in range(warmup):
        runner(x)
    with EnergyMeter(**meter_kwargs) as em:
        for _ in range(iters):
            runner(x)
    return {
        "energy_mj_per_inf": 1000.0 * em.joules / iters,
        "avg_power_w": em.avg_power_w,
        "cpu_util": em.cpu_util,
        "energy_method": em.method,
    }
