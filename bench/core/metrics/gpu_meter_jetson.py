# bench/core/metrics/gpu_meter_jetson.py
from __future__ import annotations

"""
JetsonGpuMeter (jtop)

Samples:
- GPU utilization ("GPU" key in jtop.stats)
- Jetson RAM usage over time (shared memory; Jetson has no dedicated VRAM in the desktop sense)

This meter is best-effort:
- If jtop is not installed or not supported, available() is False and the meter is a no-op.
"""

import threading
import time
from statistics import mean
from typing import Any, Dict, List, Optional

try:
    from jtop import jtop
except Exception:
    jtop = None


class JetsonGpuMeter:
    def __init__(self, sample_hz: int = 10) -> None:
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")

        self.sample_hz = int(sample_hz)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.gpu_util_samples: List[float] = []
        self.ram_used_bytes_samples: List[int] = []
        self.ram_total_bytes: Optional[int] = None
        self.timestamps_s: List[float] = []

        self._error: Optional[str] = None

    def available(self) -> bool:
        return jtop is not None

    @staticmethod
    def _to_bytes_from_mb(val: Any) -> Optional[int]:
        try:
            return int(float(val) * 1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def _parse_percent(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace("%", "").strip()
            return float(s)
        except Exception:
            return None

    def _sample_loop(self) -> None:
        assert jtop is not None

        target_dt = 1.0 / max(1, self.sample_hz)
        t0 = time.perf_counter()
        next_tick = t0

        try:
            with jtop() as jetson:
                while self._running and jetson.ok():
                    st = jetson.stats or {}

                    util = self._parse_percent(st.get("GPU"))
                    if util is not None:
                        self.gpu_util_samples.append(util)

                    ram = st.get("RAM")
                    if isinstance(ram, dict):
                        used_b = self._to_bytes_from_mb(ram.get("used"))
                        total_b = self._to_bytes_from_mb(ram.get("total"))
                        if used_b is not None:
                            self.ram_used_bytes_samples.append(used_b)
                        if total_b is not None:
                            self.ram_total_bytes = total_b

                    self.timestamps_s.append(time.perf_counter() - t0)

                    # Drift-corrected schedule
                    next_tick += target_dt
                    sleep_time = max(0.0, next_tick - time.perf_counter())
                    time.sleep(sleep_time)

        except Exception as e:
            self._error = str(e)

    def start(self) -> None:
        if not self.available():
            self._error = "jtop not available"
            return

        self._error = None
        self.gpu_util_samples.clear()
        self.ram_used_bytes_samples.clear()
        self.timestamps_s.clear()
        self.ram_total_bytes = None

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": self.available()}

        if self._error:
            out["error"] = self._error

        if self.gpu_util_samples:
            samples_sorted = sorted(self.gpu_util_samples)
            p95_idx = int(0.95 * (len(samples_sorted) - 1)) if len(samples_sorted) > 1 else 0
            out["gpu_utilization_pct"] = {
                "mean": float(mean(self.gpu_util_samples)),
                "p95": float(samples_sorted[p95_idx]),
                "samples": list(self.gpu_util_samples),
                "timestamps_s": list(self.timestamps_s),
            }

        if self.ram_used_bytes_samples:
            out["jetson_ram_used_bytes"] = {
                "mean": float(mean(self.ram_used_bytes_samples)),
                "peak": int(max(self.ram_used_bytes_samples)),
                "samples": list(self.ram_used_bytes_samples),
                "timestamps_s": list(self.timestamps_s),
            }

        if self.ram_total_bytes is not None:
            out["jetson_ram_total_bytes"] = int(self.ram_total_bytes)

        return out
