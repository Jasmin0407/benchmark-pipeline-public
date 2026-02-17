"""
MemoryMeter

Measures host RAM usage (RSS) over time and reports:
- start/end/peak RSS
- RSS delta (total and inference-window)
- RSS time series + timestamps aligned to a common time axis

Supports an optional post-delay window to capture memory settling after inference.

Notes
-----
- Timestamps use time.perf_counter() and are shifted so that t=0 corresponds to the start
  of the inference phase (pre-roll is negative time).
- GPU memory sampling is optional:
  - Prefer NVML (pynvml) if available (lazy import + one-time init per instance).
  - Fall back to torch.cuda.* if PyTorch is available and CUDA is enabled.
- infer_start_idx / infer_end_idx follow Python slicing semantics:
  - infer_start_idx is inclusive
  - infer_end_idx is exclusive
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import psutil

try:
    import torch  # Optional dependency
except Exception:
    torch = None


class MemoryMeter:
    """
    Samples RSS over time and optionally adds GPU memory info (best-effort).

    Parameters
    ----------
    sample_hz:
        Sampling frequency (samples per second).
    gpu_index:
        GPU index to query via NVML (default: 0). If you need multi-GPU reporting,
        pass the desired index (or extend to a list).
    """

    def __init__(self, sample_hz: int = 75, gpu_index: int = 0):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")

        self.sample_hz = int(sample_hz)
        self.gpu_index = int(gpu_index)

        self.samples_rss: list[int] = []
        self._process = psutil.Process()

        self._rss_start = 0
        self._rss_peak = 0
        self._rss_end = 0

        # NVML state: lazy import + one-time init (no module-level side effects)
        self._pynvml = None
        self._nvml_initialized = False

    # ---------------------------------------------------------------------
    # NVML LAZY IMPORT / INIT (GPU memory - optional)
    # ---------------------------------------------------------------------
    def _get_pynvml(self):
        """Lazy import for NVML (pynvml). Returns module or None if unavailable."""
        if self._pynvml is not None:
            return self._pynvml
        try:
            import pynvml  # local import by design
            self._pynvml = pynvml
            return self._pynvml
        except Exception:
            self._pynvml = None
            return None

    def _ensure_nvml_initialized(self) -> bool:
        """Initialize NVML exactly once per instance. Returns True if ready."""
        pynvml = self._get_pynvml()
        if pynvml is None:
            return False
        if self._nvml_initialized:
            return True
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception:
            self._nvml_initialized = False
            return False

    # ---------------------------------------------------------------------
    # GPU MEMORY HELPERS
    # ---------------------------------------------------------------------
    def _read_gpu_bytes(self) -> Optional[int]:
        """
        Read current GPU memory usage in bytes (best-effort).

        Priority:
        1) NVML (pynvml) if available
        2) torch.cuda.memory_allocated() as fallback (process allocator only)

        Returns None if unavailable.
        """
        # 1) NVML preferred
        if self._ensure_nvml_initialized():
            try:
                pynvml = self._pynvml
                h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                return int(mem.used)
            except Exception:
                pass

        # 2) Torch fallback: process allocator usage
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                pass

        return None

    def _read_gpu_peak(self) -> Optional[int]:
        """
        Read peak GPU memory (bytes) from PyTorch if available.

        NVML does not provide an allocator peak for the current process.
        torch peak refers to the process allocator peak.
        """
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.max_memory_allocated())
            except Exception:
                return None
        return None

    # ---------------------------------------------------------------------
    # INTERNAL SAMPLING
    # ---------------------------------------------------------------------
    def _sample_rss(self) -> int:
        """Read RSS once (bytes)."""
        return int(self._process.memory_info().rss)

    def _run_sampling_window(
        self,
        duration_s: float,
        interval_s: float,
        t0: float,
        timestamps: list[float],
    ) -> None:
        """
        Sample RSS for a fixed duration using a drift-corrected schedule.
        """
        next_tick = time.perf_counter()
        end_time = next_tick + max(0.0, float(duration_s))

        while time.perf_counter() < end_time:
            now = time.perf_counter()
            rss = self._sample_rss()

            self.samples_rss.append(rss)
            timestamps.append(float(now - t0))

            if rss > self._rss_peak:
                self._rss_peak = rss

            next_tick += interval_s
            sleep_time = max(0.0, next_tick - time.perf_counter())
            time.sleep(sleep_time)

    # ---------------------------------------------------------------------
    # MAIN MEASUREMENT ROUTINE
    # ---------------------------------------------------------------------
    def measure(
        self,
        fn: Callable[[], Any],
        repeats: int = 1,
        pre_roll_s: float = 5.0,
        post_delay_s: float = 2.0,
        mode: str = "static",
        dynamic_infer_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Measure RSS over time around an inference function.

        Parameters
        ----------
        fn:
            Callable to execute (e.g., model inference).
        repeats:
            Number of repeated executions of fn in static mode.
        pre_roll_s:
            Seconds to sample before inference.
        post_delay_s:
            Seconds to sample after inference ends.
        mode:
            "static":
                - sample pre-roll
                - run fn 'repeats' times (no continuous sampling inside)
                - sample at least once at inference end
                - sample post-delay
            "dynamic":
                - sample pre-roll
                - run fn in a loop for a fixed inference window while sampling
                - sample post-delay
        dynamic_infer_s:
            Optional inference window duration for "dynamic" mode (seconds).
            If None, falls back to the legacy behavior: use pre_roll_s.

        Returns
        -------
        Dict[str, Any]
            Memory measurement block.
        """
        if mode not in ("static", "dynamic"):
            raise ValueError("mode must be 'static' or 'dynamic'")

        repeats = max(1, int(repeats))
        pre_roll_s = max(0.0, float(pre_roll_s))
        post_delay_s = max(0.0, float(post_delay_s))

        # In dynamic mode, allow a separate inference window duration.
        # Backward-compatible default: legacy behavior was "infer for ~pre_roll_s seconds".
        if dynamic_infer_s is not None:
            dynamic_infer_s = max(0.0, float(dynamic_infer_s))

        self.samples_rss.clear()
        timestamps: list[float] = []

        self._rss_start = self._sample_rss()
        self._rss_peak = self._rss_start

        interval_s = 1.0 / max(float(self.sample_hz), 1.0)

        # Global time origin
        t0 = time.perf_counter()

        if mode == "static":
            # 1) Pre-roll sampling
            self._run_sampling_window(pre_roll_s, interval_s, t0, timestamps)

            infer_start_idx = len(self.samples_rss)

            # 2) Inference block
            t_inf0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            t_inf1 = time.perf_counter()
            inference_duration_s = float(t_inf1 - t_inf0)

            # Ensure we have at least one sample at inference end
            rss_end_inf = self._sample_rss()
            self.samples_rss.append(rss_end_inf)
            timestamps.append(float(time.perf_counter() - t0))
            self._rss_peak = max(self._rss_peak, rss_end_inf)

            infer_end_idx = len(self.samples_rss)  # exclusive

            # 3) Post-delay sampling
            self._run_sampling_window(post_delay_s, interval_s, t0, timestamps)

        else:
            # 1) Pre-roll sampling
            self._run_sampling_window(pre_roll_s, interval_s, t0, timestamps)

            infer_start_idx = len(self.samples_rss)

            # 2) Inference loop for a fixed window while sampling.
            # Use dynamic_infer_s if provided; otherwise keep legacy behavior (pre_roll_s).
            infer_window_s = pre_roll_s if dynamic_infer_s is None else dynamic_infer_s

            t_inf0 = time.perf_counter()
            t_end = t_inf0 + infer_window_s
            next_tick = time.perf_counter()

            while time.perf_counter() < t_end:
                fn()

                # Sample once per tick (drift-corrected)
                now = time.perf_counter()
                rss = self._sample_rss()
                self.samples_rss.append(rss)
                timestamps.append(float(now - t0))
                self._rss_peak = max(self._rss_peak, rss)

                next_tick += interval_s
                sleep_time = max(0.0, next_tick - time.perf_counter())
                time.sleep(sleep_time)

            t_inf1 = time.perf_counter()
            inference_duration_s = float(t_inf1 - t_inf0)
            infer_end_idx = len(self.samples_rss)  # exclusive

            # 3) Post-delay sampling
            self._run_sampling_window(post_delay_s, interval_s, t0, timestamps)

        # Shift time axis so inference start ~ 0 (pre-roll becomes negative)
        # Convention: we keep pre_roll_s as the offset (baseline is always [-pre_roll_s, 0]).
        timestamps_shifted = [float(t - pre_roll_s) for t in timestamps]

        self._rss_end = self._sample_rss()
        rss_delta = int(self._rss_end - self._rss_start)

        rss_array = np.asarray(self.samples_rss, dtype=float)

        # Inference-window peak and delta (best-effort even with few samples)
        if infer_start_idx < len(rss_array):
            slice_end = min(infer_end_idx, len(rss_array))
            rss_slice = rss_array[infer_start_idx:slice_end]
            if rss_slice.size > 0:
                rss_peak_inference = float(np.max(rss_slice))
                rss_delta_inference = float(rss_peak_inference - float(rss_array[infer_start_idx]))
            else:
                rss_peak_inference = float(self._rss_peak)
                rss_delta_inference = 0.0
        else:
            rss_peak_inference = float(self._rss_peak)
            rss_delta_inference = 0.0

        # Prefer duration derived from timestamp axis (stable export metadata)
        duration_s = 0.0
        if len(timestamps_shifted) >= 2:
            duration_s = float(timestamps_shifted[-1] - timestamps_shifted[0])
        if duration_s <= 0:
            duration_s = float(time.perf_counter() - t0)

        result: Dict[str, Any] = {
            "rss_start_bytes": int(self._rss_start),
            "rss_end_bytes": int(self._rss_end),
            "rss_delta_bytes": int(rss_delta),
            "rss_peak_bytes": int(self._rss_peak),
            "rss_peak_inference_bytes": float(rss_peak_inference),
            "rss_delta_inference_bytes": float(rss_delta_inference),
            "duration_s": float(round(duration_s, 3)),
            "pre_roll_s": float(pre_roll_s),
            "inference_duration_s": float(round(inference_duration_s, 3)),
            "post_delay_s": float(post_delay_s),
            "infer_start_idx": int(infer_start_idx),
            "infer_end_idx": int(infer_end_idx),
            "rss_samples": list(self.samples_rss),
            "timestamps_s": list(timestamps_shifted),
            "sample_hz": int(self.sample_hz),
            "mode": str(mode),
        }
        # Optional: expose configured inference window for audit (only if provided).
        if mode == "dynamic" and dynamic_infer_s is not None:
            result["dynamic_infer_s"] = float(dynamic_infer_s)

        # Optional GPU memory block (best-effort)
        gpu_used = self._read_gpu_bytes()
        gpu_peak = self._read_gpu_peak()
        if gpu_used is not None or gpu_peak is not None:
            gpu_block: Dict[str, int] = {}
            if gpu_used is not None:
                gpu_block["used_bytes"] = int(gpu_used)
            if gpu_peak is not None:
                gpu_block["peak_bytes"] = int(gpu_peak)
            result["gpu_memory"] = gpu_block

        return result
