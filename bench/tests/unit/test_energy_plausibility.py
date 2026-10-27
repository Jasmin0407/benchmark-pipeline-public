"""Unit tests for Phase 11 energy plausibility utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bench.tools.energy_plausibility import (
    analyze_plausibility_results,
    prepare_plausibility_config,
    write_plausibility_configs,
)


def _result(
    path: Path,
    *,
    count: int,
    total_energy_j: float,
    model_path: str = "model.onnx",
    backend: str = "rapl",
) -> Path:
    payload = {
        "meta": {"framework": "onnx", "device_target": "cpu"},
        "model": {
            "path": model_path,
            "input_shape": [1, 8],
            "dtype": "float32",
        },
        "metrics": {
            "energy": {
                "available": True,
                "backend": backend,
                "measurement_scope": "dedicated_inference_window",
                "duration_s": float(count) * 0.01,
                "inference_count": count,
                "total_energy_j": total_energy_j,
                "energy_per_inference_j": total_energy_j / count,
                "energy_per_sample_j": total_energy_j / count,
                "average_power_w": total_energy_j / (float(count) * 0.01),
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_prepare_plausibility_config_forces_exact_energy_count():
    base = {
        "run": {"warmups": 2, "repeats": 10},
        "metrics": {"energy": {"enabled": False, "min_duration_s": 2.0, "max_repeats": 10000}},
        "output": {"run_name": "baseline"},
    }

    prepared = prepare_plausibility_config(base, 500)

    assert prepared["run"]["repeats"] == 500
    assert prepared["metrics"]["energy"] == {
        "enabled": True,
        "min_duration_s": 1e-9,
        "max_repeats": 500,
    }
    assert prepared["output"]["run_name"] == "baseline_n500"
    assert base["run"]["repeats"] == 10
    assert base["metrics"]["energy"]["enabled"] is False


def test_write_plausibility_configs_creates_expected_yaml_files(tmp_path):
    paths = write_plausibility_configs(
        {"run": {"repeats": 3}, "metrics": {}, "output": {}},
        tmp_path,
        counts=(100, 500, 1000),
    )

    assert [path.name for path in paths] == [
        "energy_plausibility_n100.yaml",
        "energy_plausibility_n500.yaml",
        "energy_plausibility_n1000.yaml",
    ]
    loaded = yaml.safe_load(paths[2].read_text(encoding="utf-8"))
    assert loaded["run"]["repeats"] == 1000
    assert loaded["metrics"]["energy"]["max_repeats"] == 1000


def test_analyze_plausibility_results_passes_monotonic_stable_data(tmp_path):
    files = [
        _result(tmp_path / "n100.json", count=100, total_energy_j=1.00),
        _result(tmp_path / "n500.json", count=500, total_energy_j=5.05),
        _result(tmp_path / "n1000.json", count=1000, total_energy_j=10.10),
    ]

    report = analyze_plausibility_results(files, max_relative_spread_pct=5.0)

    assert report.total_energy_monotonic is True
    assert report.normalized_stability_passed is True
    assert report.overall_status == "passed"
    assert [p["inference_count"] for p in report.points] == [100, 500, 1000]
    assert report.normalized_relative_spread_pct < 5.0


def test_analyze_plausibility_results_flags_non_monotonic_energy(tmp_path):
    files = [
        _result(tmp_path / "n100.json", count=100, total_energy_j=1.0),
        _result(tmp_path / "n500.json", count=500, total_energy_j=4.0),
        _result(tmp_path / "n1000.json", count=1000, total_energy_j=3.5),
    ]

    report = analyze_plausibility_results(files, max_relative_spread_pct=50.0)

    assert report.total_energy_monotonic is False
    assert report.overall_status == "failed"


def test_analyze_without_tolerance_requires_manual_review(tmp_path):
    files = [
        _result(tmp_path / "n100.json", count=100, total_energy_j=1.0),
        _result(tmp_path / "n500.json", count=500, total_energy_j=5.0),
    ]

    report = analyze_plausibility_results(files)

    assert report.total_energy_monotonic is True
    assert report.normalized_stability_passed is None
    assert report.overall_status == "manual_review_required"


def test_analyze_rejects_incomparable_model_or_device_results(tmp_path):
    files = [
        _result(tmp_path / "a.json", count=100, total_energy_j=1.0, model_path="a.onnx"),
        _result(tmp_path / "b.json", count=500, total_energy_j=5.0, model_path="b.onnx"),
    ]

    with pytest.raises(ValueError, match="not comparable"):
        analyze_plausibility_results(files)


def test_analyze_rejects_unavailable_energy_measurement(tmp_path):
    path = _result(tmp_path / "unavailable.json", count=100, total_energy_j=1.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["energy"] = {
        "available": False,
        "backend": "rapl",
        "reason": "No supported counter",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    valid = _result(tmp_path / "valid.json", count=500, total_energy_j=5.0)

    with pytest.raises(ValueError, match="unavailable"):
        analyze_plausibility_results([path, valid])


def test_analyze_rejects_inconsistent_energy_per_inference(tmp_path):
    first = _result(tmp_path / "first.json", count=100, total_energy_j=1.0)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["metrics"]["energy"]["energy_per_inference_j"] = 0.5
    first.write_text(json.dumps(payload), encoding="utf-8")
    second = _result(tmp_path / "second.json", count=500, total_energy_j=5.0)

    with pytest.raises(ValueError, match="inconsistent"):
        analyze_plausibility_results([first, second])

def test_analyze_rejects_windows_emi_without_primary_domain(tmp_path):
    first = _result(
        tmp_path / "first.json",
        count=100,
        total_energy_j=1.0,
        backend="windows_emi",
    )
    second = _result(
        tmp_path / "second.json",
        count=500,
        total_energy_j=5.0,
        backend="windows_emi",
    )

    with pytest.raises(ValueError, match="missing primary_domain"):
        analyze_plausibility_results([first, second])


def test_analyze_rejects_power_state_change(tmp_path):
    first = _result(tmp_path / "first.json", count=100, total_energy_j=1.0)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["experiment_context"] = {
        "finalization": {
            "power_state_consistent": False,
            "warnings": ["Windows power scheme changed during the benchmark."],
        }
    }
    first.write_text(json.dumps(payload), encoding="utf-8")
    second = _result(tmp_path / "second.json", count=500, total_energy_j=5.0)

    with pytest.raises(ValueError, match="inconsistent power context"):
        analyze_plausibility_results([first, second])

def test_prepare_plausibility_config_preserves_explicit_emi_backend_and_channel():
    base = {
        "run": {"repeats": 10},
        "metrics": {
            "energy": {
                "enabled": True,
                "backend": "windows_emi",
                "emi_channel": "RAPL_Package0_PKG",
                "min_duration_s": 2.0,
                "max_repeats": 10000,
            }
        },
    }

    prepared = prepare_plausibility_config(base, 500)

    assert prepared["metrics"]["energy"]["backend"] == "windows_emi"
    assert prepared["metrics"]["energy"]["emi_channel"] == "RAPL_Package0_PKG"
    assert prepared["metrics"]["energy"]["max_repeats"] == 500

