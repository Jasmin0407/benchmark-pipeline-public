from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.tools.energy_baseline_validation import (
    analyze_baseline_results,
    discover_baseline_results,
)


def _payload(count: int, gross: float, adjusted: float, run_name: str) -> dict:
    baseline_power = 8.0 + count / 10000.0
    return {
        "meta": {"framework": "onnx", "device_target": "ort:cpu"},
        "model": {
            "path": "model.onnx",
            "input_shape": [32, 361, 1],
            "dtype": "fp32",
        },
        "metrics": {
            "inference_time_ms": {"batch_size": 32},
            "energy": {
                "available": True,
                "backend": "windows_emi",
                "primary_domain": "RAPL_Package0_PKG",
                "duration_s": gross / 40.0,
                "inference_count": count,
                "total_energy_j": gross,
                "gross_energy_j": gross,
                "energy_per_inference_j": gross / count,
                "energy_per_sample_j": gross / count / 32,
                "average_power_w": 40.0,
                "baseline_adjusted_energy_j": adjusted,
                "baseline_adjusted_energy_per_inference_j": adjusted / count,
                "baseline_adjusted_energy_per_sample_j": adjusted / count / 32,
                "baseline_correction": {
                    "enabled": True,
                    "available": True,
                    "method": "post_energy_idle",
                    "backend": "windows_emi",
                    "primary_domain": "RAPL_Package0_PKG",
                    "baseline_power_w": {
                        "n": 5,
                        "mean": baseline_power,
                        "sample_stddev": 0.1,
                        "cv_pct": 1.2,
                    },
                    "estimated_idle_energy_j": gross - adjusted,
                    "baseline_fraction_of_gross_pct": (gross - adjusted) / gross * 100,
                    "baseline_adjusted_energy_j": adjusted,
                    "baseline_adjusted_energy_per_inference_j": adjusted / count,
                    "correction_valid": True,
                },
            },
        },
        "thread_config": {
            "requested_threads": 4,
            "applied_intra_op_threads": 4,
            "active_provider": "CPUExecutionProvider",
        },
        "config": {"output": {"run_name": run_name}},
        "experiment_context": {
            "power": {
                "active_scheme_guid": "balanced-guid",
                "active_scheme_name": "Balanced",
                "ac_connected": True,
                "battery_saver": False,
            },
            "finalization": {"power_state_consistent": True, "warnings": []},
            "manual": {
                "dell_thermal_mode": None,
                "cooling_setup": "Laptop on desk, no cooling pad",
            },
        },
    }


def _write_results(tmp_path: Path) -> list[Path]:
    values = {
        100: (4.0, 3.2),
        500: (20.0, 16.0),
        1000: (40.0, 32.0),
    }
    paths = []
    for count, (gross, adjusted) in values.items():
        run_name = f"energy_baseline_phase13_n{count}"
        run_dir = tmp_path / run_name
        run_dir.mkdir(parents=True)
        path = run_dir / "bench.json"
        path.write_text(json.dumps(_payload(count, gross, adjusted, run_name)), encoding="utf-8")
        paths.append(path)
    return paths


def test_phase13_analysis_validates_monotonic_gross_and_adjusted_energy(tmp_path):
    paths = _write_results(tmp_path)

    report = analyze_baseline_results(paths)

    assert report.comparable is True
    assert report.gross_total_energy_monotonic is True
    assert report.adjusted_total_energy_monotonic is True
    assert report.overall_status == "complete_manual_review_required"
    assert len(report.points) == 3


def test_phase13_discovery_finds_only_phase13_runs(tmp_path):
    paths = _write_results(tmp_path)
    other = tmp_path / "other" / "bench.json"
    other.parent.mkdir()
    other.write_text(json.dumps({"config": {"output": {"run_name": "other"}}}), encoding="utf-8")

    found = discover_baseline_results(tmp_path)

    assert set(found) == set(paths)


def test_phase13_analysis_rejects_domain_mismatch(tmp_path):
    paths = _write_results(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["metrics"]["energy"]["baseline_correction"]["primary_domain"] = "RAPL_Package0_PP0"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Baseline domain differs"):
        analyze_baseline_results(paths)


def test_phase13_analysis_rejects_invalid_correction(tmp_path):
    paths = _write_results(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["metrics"]["energy"]["baseline_correction"]["correction_valid"] = False
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="marked invalid"):
        analyze_baseline_results(paths)
