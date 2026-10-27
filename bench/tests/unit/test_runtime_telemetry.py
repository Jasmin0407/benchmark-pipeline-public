from __future__ import annotations

import pytest

from bench.core.context.runtime_telemetry import (
    summarize_cpu_frequency,
    summarize_temperature,
)


def test_frequency_aggregation():
    result = summarize_cpu_frequency([2000, 3000, 4000, 5000], source="fake")
    assert result["available"] is True
    assert result["mean_mhz"] == 3500
    assert result["min_mhz"] == 2000
    assert result["max_mhz"] == 5000
    assert result["n_samples"] == 4


def test_frequency_p95_is_deterministic():
    result = summarize_cpu_frequency([1000, 2000, 3000, 4000, 5000], source="fake")
    assert result["p95_mhz"] == pytest.approx(4800.0)


def test_frequency_invalid_samples_are_ignored():
    result = summarize_cpu_frequency([None, float("nan"), -1, 0, 2500], source="fake")
    assert result["available"] is True
    assert result["n_samples"] == 1
    assert result["mean_mhz"] == 2500


def test_frequency_empty_is_unavailable():
    result = summarize_cpu_frequency([], source="fake")
    assert result["available"] is False
    assert result["n_samples"] == 0


def test_temperature_aggregation_and_throttling():
    result = summarize_temperature(
        [50.0, 55.0, 60.0],
        source="validated_fake",
        throttling_samples=[False, False, True],
    )
    assert result["available"] is True
    assert result["start_c"] == 50.0
    assert result["end_c"] == 60.0
    assert result["mean_c"] == 55.0
    assert result["max_c"] == 60.0
    assert result["thermal_throttling_observed"] is True


def test_temperature_without_validated_source_data_is_unavailable():
    result = summarize_temperature([], source="unavailable")
    assert result["available"] is False
    assert "validated" in result["reason"]
