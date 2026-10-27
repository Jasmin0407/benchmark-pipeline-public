"""Phase 10 end-to-end regression tests for optional energy measurement.

These tests exercise the public benchmark entry point and persisted ``bench.json``
artifacts. They verify that enabling the new energy feature is additive, while
legacy and explicitly disabled configurations retain the pre-energy output
shape and measurement behavior.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("torch")

from bench.core.main import main
from bench.core.metrics.energy_backend import EnergyBackend
from bench.tests.helpers.create_models import create_tiny_torch_model


class _DeterministicEnergyBackend(EnergyBackend):
    """Minimal cumulative-joule backend used only for phase-10 integration."""

    name = "phase10_fake"

    def __init__(self) -> None:
        self._read_count = 0

    def available(self) -> bool:
        return True

    def read_energy(self) -> float:
        values = (10.0, 12.0)
        value = values[min(self._read_count, len(values) - 1)]
        self._read_count += 1
        return value

    def metadata(self) -> dict:
        return {"source": "phase10_deterministic_fake"}


def _base_cfg(model_path: Path, output_dir: Path, run_name: str) -> dict:
    return {
        "model": {
            "backend": "torch",
            "path": model_path.as_posix(),
            "dtype": "float32",
        },
        "input": {
            "shape": [1, 8],
            "dtype": "float32",
        },
        "device": {"target": "cpu"},
        "run": {
            "warmups": 0,
            "repeats": 3,
            "threads": 1,
        },
        "metrics": {
            "sampler_hz": 20,
            "pre_roll_s": 0.0,
            "post_delay_s": 0.0,
            "memory_mode": "static",
        },
        "output": {
            "base_dir": output_dir.as_posix(),
            "run_name": run_name,
        },
    }


def _run_and_load(cfg: dict) -> dict:
    written = main(cfg=cfg)
    assert len(written) == 1
    _, json_path = written[0]
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def _non_energy_contract(payload: dict) -> dict:
    """Return stable structural facts used for legacy-vs-disabled comparison."""
    metrics = payload["metrics"]
    timing = metrics["inference_time_ms"]
    memory = metrics["memory"]
    return {
        "framework": payload["meta"]["framework"],
        "device_target": payload["meta"]["device_target"],
        "input_shape": payload["model"]["input_shape"],
        "dtype": payload["model"]["dtype"],
        "metric_keys": sorted(metrics.keys()),
        "timing_sample_count": len(timing["samples"]),
        "timing_batch_size": timing["batch_size"],
        "cpu_scope": metrics["cpu_utilization_pct"]["scope"],
        "memory_mode": memory["mode"],
        "requested_threads": payload["thread_config"]["requested_threads"],
    }


def test_phase10_legacy_and_explicitly_disabled_energy_are_functionally_equivalent(
    tmp_path,
    monkeypatch,
):
    """P10: legacy config and ``energy.enabled=false`` keep the same contract."""
    model_path = create_tiny_torch_model(tmp_path / "tiny_phase10.pt")

    # Disabled energy must never create or probe an energy backend.
    def fail_if_called(_cfg):
        raise AssertionError("Energy backend must not be created when energy is disabled")

    monkeypatch.setattr(
        "bench.core.measure.measure_controller.create_energy_backend",
        fail_if_called,
    )

    legacy_cfg = _base_cfg(model_path, tmp_path / "runs", "phase10_legacy")
    disabled_cfg = deepcopy(legacy_cfg)
    disabled_cfg["output"]["run_name"] = "phase10_disabled"
    disabled_cfg["metrics"]["energy"] = {
        "enabled": False,
        "min_duration_s": 2.0,
        "max_repeats": 10000,
    }

    legacy = _run_and_load(legacy_cfg)
    disabled = _run_and_load(disabled_cfg)

    assert "energy" not in legacy["metrics"]
    assert "energy" not in disabled["metrics"]
    assert _non_energy_contract(disabled) == _non_energy_contract(legacy)


def test_phase10_enabled_energy_is_additive_and_persisted(tmp_path, monkeypatch):
    """P10: enabled energy adds only the dedicated persisted energy result."""
    model_path = create_tiny_torch_model(tmp_path / "tiny_phase10_energy.pt")
    cfg = _base_cfg(model_path, tmp_path / "runs", "phase10_enabled")
    cfg["metrics"]["energy"] = {
        "enabled": True,
        # Keep this integration test short. run.repeats remains the lower bound.
        "min_duration_s": 0.001,
        "max_repeats": 3,
    }

    backend = _DeterministicEnergyBackend()
    monkeypatch.setattr(
        "bench.core.measure.measure_controller.create_energy_backend",
        lambda _cfg: backend,
    )

    payload = _run_and_load(cfg)

    energy = payload["metrics"]["energy"]
    assert energy["available"] is True
    assert energy["backend"] == "phase10_fake"
    assert energy["measurement_scope"] == "dedicated_inference_window"
    assert energy["inference_count"] == 3
    assert energy["total_energy_j"] == pytest.approx(2.0)
    assert energy["energy_per_inference_j"] == pytest.approx(2.0 / 3.0)
    assert energy["energy_per_sample_j"] == pytest.approx(2.0 / 3.0)
    assert energy["duration_s"] > 0.0
    assert energy["average_power_w"] >= 0.0
    assert energy["metadata"]["source"] == "phase10_deterministic_fake"

    # Dedicated energy inference calls are not appended to latency samples.
    assert len(payload["metrics"]["inference_time_ms"]["samples"]) == 3


def test_phase10_unavailable_energy_backend_is_persisted_non_fatally(
    tmp_path,
    monkeypatch,
):
    """P10: unavailable energy hardware must not abort the normal benchmark."""

    class UnavailableBackend(EnergyBackend):
        name = "phase10_unavailable"

        def available(self) -> bool:
            return False

        def read_energy(self) -> float:
            raise AssertionError("Unavailable backend must not be read")

        def unavailable_reason(self) -> str:
            return "Phase 10 simulated unsupported hardware."

    model_path = create_tiny_torch_model(tmp_path / "tiny_phase10_unavailable.pt")
    cfg = _base_cfg(model_path, tmp_path / "runs", "phase10_unavailable")
    cfg["metrics"]["energy"] = {
        "enabled": True,
        "min_duration_s": 0.001,
        "max_repeats": 3,
    }

    monkeypatch.setattr(
        "bench.core.measure.measure_controller.create_energy_backend",
        lambda _cfg: UnavailableBackend(),
    )

    payload = _run_and_load(cfg)

    assert len(payload["metrics"]["inference_time_ms"]["samples"]) == 3
    assert payload["metrics"]["energy"] == {
        "available": False,
        "backend": "phase10_unavailable",
        "reason": "Phase 10 simulated unsupported hardware.",
    }
