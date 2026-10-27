from __future__ import annotations

import pytest

from bench.core.metrics.energy_baseline import measure_and_apply_idle_baseline


class FakeIdleMeter:
    def __init__(self, samples):
        self.samples = list(samples)
        self.calls = 0

    def measure_idle(self, duration_s):
        self.calls += 1
        sample = dict(self.samples[self.calls - 1])
        sample.setdefault("duration_s", duration_s)
        sample.setdefault("wall_clock_duration_s", duration_s)
        sample.setdefault("duration_source", "backend_counter")
        return sample


def _gross_result():
    return {
        "available": True,
        "backend": "windows_emi",
        "primary_domain": "RAPL_Package0_PKG",
        "duration_s": 0.5,
        "inference_count": 500,
        "total_energy_j": 20.0,
    }


def _idle_sample(power_w):
    duration = 2.0
    return {
        "available": True,
        "backend": "windows_emi",
        "primary_domain": "RAPL_Package0_PKG",
        "duration_s": duration,
        "wall_clock_duration_s": duration,
        "duration_source": "backend_counter",
        "total_energy_j": power_w * duration,
        "average_power_w": power_w,
    }


def test_baseline_correction_keeps_gross_energy_and_subtracts_idle_estimate():
    meter = FakeIdleMeter([_idle_sample(8.0), _idle_sample(10.0), _idle_sample(9.0)])

    result = measure_and_apply_idle_baseline(
        meter,
        _gross_result(),
        duration_s=2.0,
        repetitions=3,
        batch_size=32,
        signal_duration_s=1.0,
    )

    assert meter.calls == 3
    assert result["baseline_power_w"]["mean"] == pytest.approx(9.0)
    assert result["gross_energy_j"] == pytest.approx(20.0)
    assert result["estimated_idle_energy_j"] == pytest.approx(4.5)
    assert result["baseline_adjusted_energy_j"] == pytest.approx(15.5)
    assert result["baseline_adjusted_energy_per_inference_j"] == pytest.approx(0.031)
    assert result["baseline_adjusted_energy_per_sample_j"] == pytest.approx(0.031 / 32)
    assert result["correction_valid"] is True


def test_baseline_correction_rejects_domain_mismatch():
    sample = _idle_sample(8.0)
    sample["primary_domain"] = "RAPL_Package0_PP0"
    meter = FakeIdleMeter([sample])

    with pytest.raises(ValueError, match="primary domain differs"):
        measure_and_apply_idle_baseline(
            meter,
            _gross_result(),
            duration_s=2.0,
            repetitions=1,
            batch_size=32,
        )


def test_baseline_correction_rejects_backend_mismatch():
    sample = _idle_sample(8.0)
    sample["backend"] = "rapl"
    meter = FakeIdleMeter([sample])

    with pytest.raises(ValueError, match="backend differs"):
        measure_and_apply_idle_baseline(
            meter,
            _gross_result(),
            duration_s=2.0,
            repetitions=1,
            batch_size=32,
        )


def test_negative_adjusted_energy_is_preserved_but_marked_invalid():
    meter = FakeIdleMeter([_idle_sample(50.0)])

    result = measure_and_apply_idle_baseline(
        meter,
        _gross_result(),
        duration_s=2.0,
        repetitions=1,
        batch_size=32,
    )

    assert result["baseline_adjusted_energy_j"] == pytest.approx(-5.0)
    assert result["correction_valid"] is False
    assert any("negative" in note.lower() for note in result["notes"])


def test_unavailable_gross_energy_does_not_attempt_baseline():
    meter = FakeIdleMeter([])

    result = measure_and_apply_idle_baseline(
        meter,
        {
            "available": False,
            "backend": "windows_emi",
            "reason": "unavailable",
        },
        duration_s=2.0,
        repetitions=3,
        batch_size=32,
    )

    assert result["available"] is False
    assert meter.calls == 0
