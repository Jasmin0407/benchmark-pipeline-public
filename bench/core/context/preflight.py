"""Preflight sampling and validation for reproducible hardware benchmarks."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import psutil

from .runtime_telemetry import summarize_cpu_frequency, summarize_temperature


def _default_cpu_reader(interval_s: float) -> float:
    return float(psutil.cpu_percent(interval=max(0.0, interval_s)))


def _default_frequency_reader() -> Optional[float]:
    freq = psutil.cpu_freq()
    if freq is None:
        return None
    current = getattr(freq, "current", None)
    return None if current is None else float(current)


def collect_preflight_observations(
    *,
    duration_s: float,
    sample_hz: float = 2.0,
    cpu_reader: Optional[Callable[[float], float]] = None,
    frequency_reader: Optional[Callable[[], Optional[float]]] = None,
    temperature_reader: Optional[Callable[[], Optional[float]]] = None,
    throttling_reader: Optional[Callable[[], Optional[bool]]] = None,
    frequency_source: str = "psutil.cpu_freq",
    temperature_source: str = "unavailable",
) -> Dict[str, Any]:
    """Collect background CPU load and low-rate runtime telemetry in one window."""
    if isinstance(duration_s, bool) or float(duration_s) <= 0:
        raise ValueError("duration_s must be > 0")
    if isinstance(sample_hz, bool) or float(sample_hz) <= 0:
        raise ValueError("sample_hz must be > 0")

    duration = float(duration_s)
    hz = float(sample_hz)
    count = max(1, int(math.ceil(duration * hz)))
    interval = duration / count

    cpu_fn = cpu_reader or _default_cpu_reader
    freq_fn = frequency_reader or _default_frequency_reader

    cpu_samples: List[float] = []
    freq_samples: List[float] = []
    temp_samples: List[float] = []
    throttle_samples: List[bool] = []

    for _ in range(count):
        try:
            cpu_value = float(cpu_fn(interval))
            if math.isfinite(cpu_value):
                cpu_samples.append(cpu_value)
        except Exception:
            pass

        try:
            freq_value = freq_fn()
            if freq_value is not None:
                freq_value = float(freq_value)
                if math.isfinite(freq_value) and freq_value > 0:
                    freq_samples.append(freq_value)
        except Exception:
            pass

        if temperature_reader is not None:
            try:
                temp_value = temperature_reader()
                if temp_value is not None:
                    temp_value = float(temp_value)
                    if math.isfinite(temp_value):
                        temp_samples.append(temp_value)
            except Exception:
                pass

        if throttling_reader is not None:
            try:
                throttle = throttling_reader()
                if throttle is not None:
                    throttle_samples.append(bool(throttle))
            except Exception:
                pass

    background: Dict[str, Any]
    if cpu_samples:
        background = {
            "available": True,
            "source": "psutil.cpu_percent",
            "preflight_duration_s": duration,
            "sample_hz": hz,
            "cpu_mean_pct": sum(cpu_samples) / len(cpu_samples),
            "cpu_max_pct": max(cpu_samples),
            "n_samples": len(cpu_samples),
            "process_count": len(psutil.pids()),
        }
    else:
        background = {
            "available": False,
            "source": "psutil.cpu_percent",
            "preflight_duration_s": duration,
            "reason": "Background CPU utilization could not be sampled.",
        }

    temperature = summarize_temperature(
        temp_samples,
        source=temperature_source,
        throttling_samples=throttle_samples if throttle_samples else None,
    )

    return {
        "background": background,
        "cpu_frequency": summarize_cpu_frequency(freq_samples, source=frequency_source),
        "thermal": temperature,
    }


def evaluate_preflight(
    *,
    rules: Dict[str, Any],
    power: Dict[str, Any],
    background: Dict[str, Any],
    thermal: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate configured reproducibility gates without inventing thresholds."""
    warnings: List[str] = []
    failed = False
    checks_applied = 0

    require_ac = bool(rules.get("require_ac_power", False))
    if require_ac:
        checks_applied += 1
        ac_connected = power.get("ac_connected")
        if ac_connected is not True:
            failed = True
            warnings.append("AC power is required but was not confirmed.")

    max_background = rules.get("max_background_cpu_pct")
    if max_background is not None:
        checks_applied += 1
        observed = background.get("cpu_max_pct")
        if observed is None:
            failed = True
            warnings.append("Background CPU limit is configured but CPU load is unavailable.")
        elif float(observed) > float(max_background):
            failed = True
            warnings.append(
                f"Background CPU load {float(observed):.3f}% exceeds configured limit "
                f"{float(max_background):.3f}%."
            )

    max_start_temp = rules.get("max_start_temp_c")
    if max_start_temp is not None:
        checks_applied += 1
        observed = thermal.get("start_c") if thermal.get("available") else None
        if observed is None:
            failed = True
            warnings.append("Start-temperature limit is configured but temperature is unavailable.")
        elif float(observed) > float(max_start_temp):
            failed = True
            warnings.append(
                f"Start temperature {float(observed):.3f} °C exceeds configured limit "
                f"{float(max_start_temp):.3f} °C."
            )

    if checks_applied == 0:
        return {
            "passed": None,
            "status": "manual_review_required",
            "warnings": ["No automatic preflight acceptance thresholds are configured."],
        }

    return {
        "passed": not failed,
        "status": "passed" if not failed else "failed",
        "warnings": warnings,
    }
