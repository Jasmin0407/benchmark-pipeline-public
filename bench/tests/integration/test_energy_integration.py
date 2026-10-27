"""Phase 9 persistence tests for energy benchmark results."""

import json
from pathlib import Path

from bench.core.io.result_writer import write_result
from bench.core.schemas.run_schema import MetaSchema, MetricsSchema, ModelSchema, RunSchema


def _run_with_energy(tmp_path: Path, energy: dict) -> RunSchema:
    return RunSchema(
        meta=MetaSchema(framework="torch", device_target="cpu"),
        model=ModelSchema(
            path="inline",
            input_shape=[1, 1, 256],
            dtype="float32",
        ),
        metrics=MetricsSchema(
            inference_time_ms={"mean": 10.0, "samples": [10.0]},
            energy=energy,
        ),
        config={
            "output": {"base_dir": str(tmp_path)},
            "metrics": {"energy": {"enabled": True}},
        },
    )


def test_energy_is_written_to_bench_json_and_roundtrips(tmp_path):
    """P9-T2/P9-T4: persisted JSON contains the full typed energy block."""
    expected_energy = {
        "available": True,
        "backend": "rapl",
        "measurement_scope": "dedicated_inference_window",
        "duration_s": 2.0,
        "inference_count": 200,
        "total_energy_j": 8.0,
        "energy_per_inference_j": 0.04,
        "energy_per_sample_j": 0.04,
        "average_power_w": 4.0,
    }
    run = _run_with_energy(tmp_path, expected_energy)

    output_path = Path(write_result(run, ts="phase9-energy-roundtrip"))
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metrics"]["energy"]["backend"] == "rapl"
    assert payload["metrics"]["energy"]["total_energy_j"] == 8.0

    restored = RunSchema.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert restored.metrics.energy is not None
    assert restored.metrics.energy.model_dump(exclude_none=True) == expected_energy


def test_unavailable_energy_result_is_persisted_without_failure(tmp_path):
    energy = {
        "available": False,
        "backend": "nvml",
        "reason": "Total energy counter is not supported by this GPU.",
    }
    run = _run_with_energy(tmp_path, energy)

    output_path = Path(write_result(run, ts="phase9-energy-unavailable"))
    restored = RunSchema.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert restored.metrics.energy is not None
    assert restored.metrics.energy.available is False
    assert restored.metrics.energy.reason == energy["reason"]
