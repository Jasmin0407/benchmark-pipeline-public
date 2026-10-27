from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from bench.core.main import main
from bench.tests.helpers.create_models import create_tiny_torch_model


def _cfg(model_path: Path, output_dir: Path, enabled: bool) -> dict:
    return {
        "model": {"backend": "torch", "path": model_path.as_posix(), "dtype": "float32"},
        "input": {"shape": [1, 8], "dtype": "float32"},
        "device": {"target": "cpu"},
        "run": {"warmups": 0, "repeats": 2, "threads": 1},
        "metrics": {
            "sampler_hz": 20,
            "pre_roll_s": 0.0,
            "post_delay_s": 0.0,
            "memory_mode": "static",
            "energy": {"enabled": False, "min_duration_s": 2.0, "max_repeats": 10000},
        },
        "experiment": {
            "context": {
                "enabled": enabled,
                "preflight": {"enabled": False},
                "manual": {
                    "dell_thermal_mode": "Ultra Performance",
                    "cooling_setup": "Laptop on desk",
                    "room_temperature_c": 22.0,
                    "notes": "phase11 windows context",
                },
            }
        },
        "output": {"base_dir": output_dir.as_posix(), "run_name": "phase11_context"},
    }


def _run(cfg: dict) -> dict:
    written = main(cfg=cfg)
    _, path = written[0]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_context_disabled_keeps_legacy_json_shape(tmp_path):
    model = create_tiny_torch_model(tmp_path / "tiny.pt")
    payload = _run(_cfg(model, tmp_path / "runs", False))
    assert "experiment_context" not in payload
    assert "energy" not in payload["metrics"]
    assert len(payload["metrics"]["inference_time_ms"]["samples"]) == 2


def test_context_enabled_is_persisted_without_changing_timing_samples(tmp_path, monkeypatch):
    model = create_tiny_torch_model(tmp_path / "tiny_context.pt")
    monkeypatch.setattr(
        "bench.core.main.collect_experiment_context",
        lambda _cfg: {
            "platform": "windows",
            "power": {"available": True, "ac_connected": True, "active_scheme_name": "Balanced"},
            "manual": {"dell_thermal_mode": "Ultra Performance"},
            "preflight": {"passed": None, "status": "disabled", "warnings": []},
        },
    )
    monkeypatch.setattr(
        "bench.core.main.finalize_experiment_context",
        lambda _cfg, context: {
            **context,
            "power_end": {"available": True, "ac_connected": True, "active_scheme_name": "Balanced"},
            "finalization": {"power_state_consistent": True, "compared_fields": 1, "warnings": []},
        },
    )
    payload = _run(_cfg(model, tmp_path / "runs", True))
    assert payload["experiment_context"]["platform"] == "windows"
    assert payload["experiment_context"]["power"]["ac_connected"] is True
    assert payload["experiment_context"]["manual"]["dell_thermal_mode"] == "Ultra Performance"
    assert payload["experiment_context"]["finalization"]["power_state_consistent"] is True
    assert len(payload["metrics"]["inference_time_ms"]["samples"]) == 2


def test_experiment_context_roundtrip_via_schema():
    from bench.core.schemas.run_schema import RunSchema

    payload = {
        "meta": {"framework": "onnx", "device_target": "ort:cpu"},
        "model": {"path": "model.onnx", "input_shape": [1, 8], "dtype": "fp32"},
        "metrics": {},
        "config": {},
        "experiment_context": {
            "platform": "windows",
            "power": {"available": True, "ac_connected": True},
            "cpu_frequency": {"available": True, "mean_mhz": 3000.0},
            "thermal": {"available": False, "reason": "No validated source."},
            "background": {"cpu_mean_pct": 2.0},
            "manual": {"room_temperature_c": 22.0},
            "preflight": {"passed": True, "warnings": []},
        },
    }
    run = RunSchema.model_validate(payload)
    restored = RunSchema.model_validate_json(run.model_dump_json())
    assert restored.experiment_context is not None
    assert restored.experiment_context.power["ac_connected"] is True
    assert restored.experiment_context.manual["room_temperature_c"] == 22.0
