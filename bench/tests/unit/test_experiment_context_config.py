from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.core.config.config_loader import load_config


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _minimal() -> dict:
    return {
        "model": {"path": "model.onnx", "backend": "onnx"},
        "run": {"warmups": 0, "repeats": 1},
    }


def test_context_defaults_are_disabled(tmp_path):
    cfg = load_config(str(_write(tmp_path / "cfg.yaml", _minimal())))
    context = cfg["experiment"]["context"]
    assert context["enabled"] is False
    assert context["preflight"]["enabled"] is False
    assert context["preflight"]["duration_s"] == 5.0
    assert context["manual"]["dell_thermal_mode"] is None


def test_legacy_config_without_experiment_section_remains_valid(tmp_path):
    cfg = load_config(str(_write(tmp_path / "legacy.yaml", _minimal())))
    assert cfg["experiment"]["context"]["enabled"] is False


def test_manual_context_is_preserved(tmp_path):
    payload = _minimal()
    payload["experiment"] = {
        "context": {
            "enabled": True,
            "manual": {
                "dell_thermal_mode": "Ultra Performance",
                "cooling_setup": "Laptop on desk",
                "room_temperature_c": 22.0,
                "notes": "Reference run",
            },
        }
    }
    cfg = load_config(str(_write(tmp_path / "manual.yaml", payload)))
    assert cfg["experiment"]["context"]["manual"]["dell_thermal_mode"] == "Ultra Performance"
    assert cfg["experiment"]["context"]["manual"]["room_temperature_c"] == 22.0


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"enabled": "yes"}, "experiment.context.enabled"),
        ({"preflight": {"enabled": "yes"}}, "experiment.context.preflight.enabled"),
        ({"preflight": {"duration_s": 0}}, "experiment.context.preflight.duration_s"),
        ({"preflight": {"sample_hz": -1}}, "experiment.context.preflight.sample_hz"),
        ({"preflight": {"max_background_cpu_pct": -1}}, "max_background_cpu_pct"),
        ({"preflight": {"require_ac_power": "yes"}}, "require_ac_power"),
    ],
)
def test_invalid_context_config_is_rejected(tmp_path, patch, expected):
    payload = _minimal()
    payload["experiment"] = {"context": patch}
    with pytest.raises(ValueError, match=expected):
        load_config(str(_write(tmp_path / "invalid.yaml", payload)))
