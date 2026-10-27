"""Unit tests for Phase 12 energy repeatability tooling."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import pytest
import yaml

from bench.tools.energy_repeatability import (
    analyze_repeatability_results,
    build_counterbalanced_schedule,
    prepare_repeatability_experiment,
)


def _base_config(count: int) -> dict:
    return {
        "model": {"path": "model.onnx", "backend": "onnx"},
        "input": {"shape": [32, 361, 1], "batch": 32},
        "device": {"target": "ort:cpu", "allow_fallback": False},
        "run": {"warmups": 5, "repeats": count},
        "metrics": {
            "energy": {
                "enabled": True,
                "backend": "windows_emi",
                "emi_channel": "RAPL_Package0_PKG",
                "min_duration_s": 1e-9,
                "max_repeats": count,
            }
        },
        "experiment": {
            "context": {
                "enabled": True,
                "preflight": {"enabled": True, "require_ac_power": True},
                "manual": {"cooling_setup": "Laptop on desk, no cooling pad"},
            }
        },
        "output": {"dir": "runs/onnx_cpu", "run_name": f"phase11_n{count}"},
    }


def _write_base_configs(tmp_path: Path) -> list[Path]:
    paths = []
    for count in (100, 500, 1000):
        path = tmp_path / f"base_n{count}.yaml"
        path.write_text(
            yaml.safe_dump(_base_config(count), sort_keys=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _result_payload(
    *,
    count: int,
    round_index: int,
    position: int,
    schedule_id: str,
    energy_per_inference_j: float,
    background_mean: float = 3.0,
) -> dict:
    total = energy_per_inference_j * count
    duration = count * 0.001
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
                "measurement_scope": "dedicated_inference_window",
                "duration_s": duration,
                "inference_count": count,
                "total_energy_j": total,
                "energy_per_inference_j": energy_per_inference_j,
                "energy_per_sample_j": energy_per_inference_j / 32,
                "average_power_w": total / duration,
                "primary_domain": "RAPL_Package0_PKG",
            },
        },
        "thread_config": {
            "requested_threads": 4,
            "applied_intra_op_threads": 4,
            "active_provider": "CPUExecutionProvider",
        },
        "config": {
            "output": {
                "run_name": (
                    f"energy_repeatability_{schedule_id}_r{round_index:02d}_"
                    f"p{position:02d}_n{count}"
                )
            }
        },
        "experiment_context": {
            "power": {
                "active_scheme_guid": "balanced-guid",
                "active_scheme_name": "Balanced",
                "ac_connected": True,
                "battery_saver": False,
            },
            "finalization": {"power_state_consistent": True, "warnings": []},
            "background": {
                "cpu_mean_pct": background_mean,
                "cpu_max_pct": background_mean + 2.0,
            },
            "manual": {
                "dell_thermal_mode": None,
                "cooling_setup": "Laptop on desk, no cooling pad",
            },
            "preflight": {"passed": True, "warnings": []},
        },
    }


def _write_complete_results(tmp_path: Path, schedule_id: str = "test") -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    schedule = build_counterbalanced_schedule((100, 500, 1000), repetitions=6, seed=7)
    paths = []
    offsets = {100: 0.036, 500: 0.037, 1000: 0.0355}
    for round_index, order in enumerate(schedule, start=1):
        for position, count in enumerate(order, start=1):
            # Small deterministic variation so sample SD and CV are non-zero.
            per_inference = offsets[count] * (1.0 + (round_index - 3.5) * 0.002)
            payload = _result_payload(
                count=count,
                round_index=round_index,
                position=position,
                schedule_id=schedule_id,
                energy_per_inference_j=per_inference,
                background_mean=2.0 + round_index * 0.1,
            )
            run_dir = tmp_path / f"r{round_index:02d}_p{position:02d}_n{count}"
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "bench.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
    return paths


def test_six_round_schedule_uses_all_permutations_and_is_position_balanced():
    schedule = build_counterbalanced_schedule((100, 500, 1000), repetitions=6, seed=42)

    assert len(schedule) == 6
    assert set(schedule) == set(permutations((100, 500, 1000)))

    for count in (100, 500, 1000):
        positions = [order.index(count) + 1 for order in schedule]
        assert positions.count(1) == 2
        assert positions.count(2) == 2
        assert positions.count(3) == 2


def test_prepare_repeatability_experiment_writes_18_configs_and_runner(tmp_path):
    bases = _write_base_configs(tmp_path)
    out = tmp_path / "phase12_configs"

    result = prepare_repeatability_experiment(
        bases,
        out,
        repetitions=6,
        seed=123,
        schedule_id="scheduleA",
        results_dir="runs/phase12",
    )

    assert result["manifest"]["total_runs"] == 18
    assert len(result["config_paths"]) == 18
    assert Path(result["runner_path"]).exists()
    assert Path(result["manifest_path"]).exists()

    first = yaml.safe_load(Path(result["config_paths"][0]).read_text(encoding="utf-8"))
    assert first["metrics"]["energy"]["backend"] == "windows_emi"
    assert first["metrics"]["energy"]["emi_channel"] == "RAPL_Package0_PKG"
    assert first["output"]["base_dir"] == "runs/phase12"
    assert "dir" not in first["output"]
    assert first["output"]["run_name"].startswith("energy_repeatability_scheduleA_")
    assert "Phase 12 repeatability" in first["experiment"]["context"]["manual"]["notes"]


def test_analyze_repeatability_calculates_sample_statistics(tmp_path):
    files = _write_complete_results(tmp_path)

    report = analyze_repeatability_results(files, expected_repetitions=6)

    assert report.comparable is True
    assert report.overall_status == "complete_manual_review_required"
    assert report.cv_acceptance_passed is None
    assert len(report.condition_summaries) == 3
    assert report.position_balance["100"] == {"1": 2, "2": 2, "3": 2}

    n100 = next(item for item in report.condition_summaries if item["inference_count"] == 100)
    assert n100["run_count"] == 6
    assert n100["energy_per_inference_j"]["sample_stddev"] > 0.0
    assert n100["energy_per_inference_j"]["cv_pct"] > 0.0


def test_analyze_repeatability_can_apply_explicit_cv_threshold(tmp_path):
    files = _write_complete_results(tmp_path)

    report = analyze_repeatability_results(
        files,
        expected_repetitions=6,
        max_cv_pct=5.0,
    )

    assert report.cv_acceptance_passed is True
    assert report.overall_status == "passed"


def test_analyze_repeatability_rejects_missing_result(tmp_path):
    files = _write_complete_results(tmp_path)

    with pytest.raises(ValueError, match="Expected exactly 18"):
        analyze_repeatability_results(files[:-1], expected_repetitions=6)


def test_analyze_repeatability_rejects_incomparable_fingerprint(tmp_path):
    files = _write_complete_results(tmp_path)
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    payload["model"]["path"] = "different.onnx"
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not comparable"):
        analyze_repeatability_results(files, expected_repetitions=6)


def test_analyze_repeatability_rejects_power_state_change(tmp_path):
    files = _write_complete_results(tmp_path)
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    payload["experiment_context"]["finalization"] = {
        "power_state_consistent": False,
        "warnings": ["Power scheme changed"],
    }
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent power context"):
        analyze_repeatability_results(files, expected_repetitions=6)


def test_discover_repeatability_results_filters_schedule(tmp_path):
    from bench.tools.energy_repeatability import discover_repeatability_results

    files = _write_complete_results(tmp_path / "wanted", schedule_id="wanted")
    _write_complete_results(tmp_path / "other", schedule_id="other")

    discovered = discover_repeatability_results(tmp_path, schedule_id="wanted")

    assert len(discovered) == 18
    assert set(discovered) == set(files)
