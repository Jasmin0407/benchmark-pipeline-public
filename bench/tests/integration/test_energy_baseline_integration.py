"""Phase 13 persistence tests for optional baseline correction."""

from __future__ import annotations

import json
from pathlib import Path

from bench.core.io.result_writer import write_result
from bench.core.schemas.run_schema import MetaSchema, MetricsSchema, ModelSchema, RunSchema


def test_phase13_baseline_fields_persist_without_replacing_gross_energy(tmp_path: Path):
    energy = {
        "available": True,
        "backend": "windows_emi",
        "measurement_scope": "dedicated_inference_window",
        "duration_s": 0.5,
        "duration_source": "backend_counter",
        "wall_clock_duration_s": 0.501,
        "inference_count": 500,
        "total_energy_j": 20.0,
        "gross_energy_j": 20.0,
        "energy_per_inference_j": 0.04,
        "energy_per_sample_j": 0.00125,
        "average_power_w": 40.0,
        "primary_domain": "RAPL_Package0_PKG",
        "baseline_adjusted_energy_j": 16.0,
        "baseline_adjusted_energy_per_inference_j": 0.032,
        "baseline_adjusted_energy_per_sample_j": 0.001,
        "baseline_correction": {
            "enabled": True,
            "available": True,
            "method": "post_energy_idle",
            "backend": "windows_emi",
            "primary_domain": "RAPL_Package0_PKG",
            "baseline_power_w": {
                "n": 5,
                "mean": 8.0,
                "sample_stddev": 0.2,
                "cv_pct": 2.5,
            },
            "estimated_idle_energy_j": 4.0,
            "baseline_adjusted_energy_j": 16.0,
            "correction_valid": True,
        },
    }
    run = RunSchema(
        meta=MetaSchema(framework="onnx", device_target="ort:cpu"),
        model=ModelSchema(
            path="model.onnx",
            input_shape=[32, 361, 1],
            dtype="fp32",
        ),
        metrics=MetricsSchema(energy=energy),
        config={
            "metrics": {
                "energy": {
                    "enabled": True,
                    "baseline_correction": {"enabled": True},
                }
            },
            "output": {"base_dir": str(tmp_path)},
        },
    )

    path = Path(write_result(run, ts="phase13-baseline"))
    payload = json.loads(path.read_text(encoding="utf-8"))

    persisted = payload["metrics"]["energy"]
    assert persisted["total_energy_j"] == 20.0
    assert persisted["gross_energy_j"] == 20.0
    assert persisted["baseline_adjusted_energy_j"] == 16.0
    assert persisted["baseline_correction"]["correction_valid"] is True

    restored = RunSchema.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.metrics.energy is not None
    assert restored.metrics.energy.total_energy_j == 20.0
    assert restored.metrics.energy.gross_energy_j == 20.0
    assert restored.metrics.energy.baseline_adjusted_energy_j == 16.0
