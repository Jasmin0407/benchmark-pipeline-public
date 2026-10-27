"""Phase 9 tests for the persisted energy result schema."""

import json

import pytest
from pydantic import ValidationError

from bench.core.schemas.run_schema import EnergySchema, MetricsSchema, RunSchema


def _minimal_run_dict():
    return {
        "meta": {
            "run_id": "energy-schema-test",
            "framework": "torch",
            "device_target": "cpu",
            "timestamp": "2026-10-21T12:00:00",
        },
        "model": {
            "path": "inline",
            "input_shape": [1, 1, 256],
            "dtype": "float32",
        },
        "metrics": {},
        "config": {"metrics": {"energy": {"enabled": False}}},
    }


def test_energy_schema_accepts_available_measurement():
    """P9-T1: a complete successful energy result validates."""
    energy = EnergySchema(
        available=True,
        backend="rapl",
        measurement_scope="dedicated_inference_window",
        duration_s=2.0134,
        duration_source="backend_counter",
        wall_clock_duration_s=2.0200,
        inference_count=1840,
        total_energy_j=8.721,
        energy_per_inference_j=0.004739,
        energy_per_sample_j=0.004739,
        average_power_w=4.331,
    )

    assert energy.available is True
    assert energy.backend == "rapl"
    assert energy.total_energy_j == pytest.approx(8.721)


def test_energy_schema_accepts_unavailable_measurement():
    """Unavailable hardware remains a valid, persistable result."""
    energy = EnergySchema(
        available=False,
        backend="rapl",
        reason="No supported energy measurement backend available.",
    )

    assert energy.available is False
    assert energy.reason is not None
    assert energy.total_energy_j is None


def test_metrics_schema_types_energy_block():
    metrics = MetricsSchema(
        energy={
            "available": True,
            "backend": "nvml",
            "duration_s": 2.0,
            "inference_count": 100,
            "total_energy_j": 10.0,
            "energy_per_inference_j": 0.1,
            "energy_per_sample_j": 0.1,
            "average_power_w": 5.0,
        }
    )

    assert isinstance(metrics.energy, EnergySchema)
    assert metrics.energy.backend == "nvml"


def test_legacy_run_without_energy_remains_valid():
    """Backward compatibility: historic results need no energy field."""
    run = RunSchema.model_validate(_minimal_run_dict())

    assert run.metrics.energy is None


def test_energy_schema_rejects_negative_physical_values():
    with pytest.raises(ValidationError):
        EnergySchema(
            available=True,
            total_energy_j=-0.1,
        )


def test_energy_schema_json_roundtrip_preserves_all_fields():
    """P9-T3/P9-T4: schema JSON roundtrip loses no energy information."""
    data = _minimal_run_dict()
    data["metrics"]["energy"] = {
        "available": True,
        "backend": "jetson_ina3221",
        "measurement_scope": "dedicated_inference_window",
        "duration_s": 2.137,
        "inference_count": 1420,
        "total_energy_j": 12.471,
        "energy_per_inference_j": 0.008782,
        "energy_per_sample_j": 0.008782,
        "average_power_w": 5.836,
        "energy_per_signal_s_j": 0.00014637,
        "primary_domain": "VDD_IN",
        "domains": {"VDD_IN": {"energy_j": 12.471}},
        "metadata": {"platform": "jetson"},
    }

    first = RunSchema.model_validate(data)
    payload = first.model_dump_json()
    second = RunSchema.model_validate_json(payload)

    assert json.loads(second.model_dump_json())["metrics"]["energy"] == json.loads(
        first.model_dump_json()
    )["metrics"]["energy"]


def test_energy_schema_accepts_optional_phase13_baseline_fields():
    energy = EnergySchema(
        available=True,
        backend="windows_emi",
        duration_s=0.5,
        inference_count=500,
        total_energy_j=20.0,
        gross_energy_j=20.0,
        baseline_adjusted_energy_j=15.5,
        baseline_adjusted_energy_per_inference_j=0.031,
        baseline_adjusted_energy_per_sample_j=0.00096875,
        baseline_correction={
            "enabled": True,
            "available": True,
            "method": "post_energy_idle",
            "primary_domain": "RAPL_Package0_PKG",
            "baseline_power_w": {"mean": 9.0, "cv_pct": 2.0},
        },
    )

    assert energy.gross_energy_j == pytest.approx(20.0)
    assert energy.baseline_adjusted_energy_j == pytest.approx(15.5)
    assert energy.baseline_correction["method"] == "post_energy_idle"


def test_energy_schema_preserves_negative_adjusted_value_for_diagnostics():
    energy = EnergySchema(
        available=True,
        total_energy_j=1.0,
        baseline_adjusted_energy_j=-0.1,
        baseline_correction={"correction_valid": False},
    )

    assert energy.baseline_adjusted_energy_j == pytest.approx(-0.1)
