"""Runtime telemetry aggregation for experiment-context auditing."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _valid_finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_cpu_frequency(samples_mhz: Iterable[Any], *, source: str) -> Dict[str, Any]:
    """Summarize observed CPU-frequency samples in MHz."""
    values = [v for v in _valid_finite(samples_mhz) if v > 0]
    if not values:
        return {
            "available": False,
            "source": source,
            "reason": "No valid CPU-frequency samples were collected.",
            "n_samples": 0,
        }
    return {
        "available": True,
        "source": source,
        "measurement_scope": "preflight_window",
        "mean_mhz": sum(values) / len(values),
        "min_mhz": min(values),
        "max_mhz": max(values),
        "p95_mhz": _percentile(values, 95.0),
        "n_samples": len(values),
    }


def summarize_temperature(
    samples_c: Iterable[Any],
    *,
    source: str,
    throttling_samples: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Summarize validated temperature samples in degrees Celsius."""
    values = _valid_finite(samples_c)
    if not values:
        return {
            "available": False,
            "source": source,
            "reason": "No validated CPU temperature source is configured.",
        }

    throttling_observed = None
    if throttling_samples is not None:
        throttling_observed = any(bool(value) for value in throttling_samples)

    return {
        "available": True,
        "source": source,
        "measurement_scope": "preflight_window",
        "start_c": values[0],
        "end_c": values[-1],
        "mean_c": sum(values) / len(values),
        "max_c": max(values),
        "thermal_throttling_observed": throttling_observed,
        "n_samples": len(values),
    }
