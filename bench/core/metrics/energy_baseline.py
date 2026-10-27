"""Optional idle-baseline correction for gross energy measurements.

Phase 13 keeps gross energy as the primary measurement and optionally estimates
how much energy the selected hardware domain would have consumed during the
same interval without inference work.

The baseline is measured *after* the gross energy pass so enabling Phase 13
cannot perturb latency, CPU, memory, or the gross energy window itself. Multiple
idle windows are used to make baseline variability visible instead of hiding it
behind one sample.
"""

from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional

from .energy_meter import EnergyMeter


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("Baseline summary requires at least one value")
    parsed = [float(value) for value in values]
    mu = mean(parsed)
    sigma = stdev(parsed) if len(parsed) >= 2 else 0.0
    cv_pct = None if mu == 0.0 else abs(sigma / mu) * 100.0
    return {
        "n": len(parsed),
        "mean": mu,
        "sample_stddev": sigma,
        "cv_pct": cv_pct,
        "median": median(parsed),
        "minimum": min(parsed),
        "maximum": max(parsed),
    }


def measure_and_apply_idle_baseline(
    meter: EnergyMeter,
    gross_result: Dict[str, Any],
    *,
    duration_s: float,
    repetitions: int,
    batch_size: int,
    signal_duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure post-energy idle power and compute an optional correction.

    The function never replaces ``gross_result["total_energy_j"]``. It returns a
    separate audit block. The caller may expose selected corrected values next
    to the gross metrics for convenience.

    A correction is only considered domain-consistent when every baseline
    sample uses the same backend and primary domain as the gross measurement.
    """

    if gross_result.get("available") is not True:
        return {
            "enabled": True,
            "available": False,
            "method": "post_energy_idle",
            "reason": "Gross energy measurement is unavailable; baseline correction was not attempted.",
        }

    if (
        isinstance(duration_s, bool)
        or not math.isfinite(float(duration_s))
        or float(duration_s) <= 0.0
    ):
        raise ValueError("baseline duration_s must be a finite number > 0")
    if (
        isinstance(repetitions, bool)
        or int(repetitions) < 1
        or float(repetitions) != float(int(repetitions))
    ):
        raise ValueError("baseline repetitions must be an integer >= 1")
    if isinstance(batch_size, bool) or int(batch_size) < 1:
        raise ValueError("batch_size must be an integer >= 1")

    gross_energy_j = float(gross_result["total_energy_j"])
    gross_duration_s = float(gross_result["duration_s"])
    inference_count = int(gross_result["inference_count"])
    gross_backend = str(gross_result.get("backend") or "")
    gross_domain = gross_result.get("primary_domain")

    samples: List[Dict[str, Any]] = []
    powers: List[float] = []

    for index in range(int(repetitions)):
        sample = meter.measure_idle(float(duration_s))
        if sample.get("available") is False:
            return {
                "enabled": True,
                "available": False,
                "method": "post_energy_idle",
                "reason": (
                    f"Idle baseline sample {index + 1} is unavailable: "
                    f"{sample.get('reason') or 'unknown reason'}"
                ),
                "completed_samples": samples,
            }

        sample_backend = str(sample.get("backend") or "")
        sample_domain = sample.get("primary_domain")
        if gross_backend and sample_backend != gross_backend:
            raise ValueError(
                "Baseline backend differs from gross energy backend: "
                f"{sample_backend!r} != {gross_backend!r}"
            )
        if gross_domain is not None and sample_domain != gross_domain:
            raise ValueError(
                "Baseline primary domain differs from gross energy domain: "
                f"{sample_domain!r} != {gross_domain!r}"
            )

        power_w = float(sample["average_power_w"])
        powers.append(power_w)
        samples.append(
            {
                "index": index + 1,
                "duration_s": float(sample["duration_s"]),
                "duration_source": sample.get("duration_source"),
                "wall_clock_duration_s": sample.get("wall_clock_duration_s"),
                "total_energy_j": float(sample["total_energy_j"]),
                "average_power_w": power_w,
            }
        )

    power_summary = _summary(powers)
    baseline_power_w = float(power_summary["mean"])
    estimated_idle_energy_j = baseline_power_w * gross_duration_s
    adjusted_energy_j = gross_energy_j - estimated_idle_energy_j
    correction_valid = math.isfinite(adjusted_energy_j) and adjusted_energy_j >= 0.0

    per_inference = adjusted_energy_j / inference_count
    per_sample = per_inference / int(batch_size)
    per_signal_s = None
    if signal_duration_s is not None:
        signal_duration = float(signal_duration_s)
        if not math.isfinite(signal_duration) or signal_duration <= 0.0:
            raise ValueError("signal_duration_s must be a finite number > 0")
        per_signal_s = per_sample / signal_duration

    fraction_pct = None
    if gross_energy_j > 0.0:
        fraction_pct = estimated_idle_energy_j / gross_energy_j * 100.0

    notes: List[str] = [
        "Gross energy remains the primary, unmodified measurement.",
        "Idle power was measured after the gross energy pass using the same EnergyMeter backend.",
    ]
    if not correction_valid:
        notes.append(
            "Baseline-adjusted energy is negative; treat the correction as invalid/noise-dominated."
        )

    result: Dict[str, Any] = {
        "enabled": True,
        "available": True,
        "method": "post_energy_idle",
        "backend": gross_backend or None,
        "primary_domain": gross_domain,
        "requested_window_duration_s": float(duration_s),
        "repetitions": int(repetitions),
        "samples": samples,
        "baseline_power_w": power_summary,
        "gross_energy_j": gross_energy_j,
        "gross_duration_s": gross_duration_s,
        "estimated_idle_energy_j": estimated_idle_energy_j,
        "baseline_fraction_of_gross_pct": fraction_pct,
        "baseline_adjusted_energy_j": adjusted_energy_j,
        "baseline_adjusted_energy_per_inference_j": per_inference,
        "baseline_adjusted_energy_per_sample_j": per_sample,
        "correction_valid": correction_valid,
        "notes": notes,
    }
    if per_signal_s is not None:
        result["baseline_adjusted_energy_per_signal_s_j"] = per_signal_s
    return result
