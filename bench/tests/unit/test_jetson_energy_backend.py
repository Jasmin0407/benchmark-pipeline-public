from __future__ import annotations

from pathlib import Path

import pytest

from bench.core.metrics.energy_backend import EnergyBackendError
from bench.core.metrics.energy_meter import EnergyMeter
from bench.core.metrics.jetson_energy_backend import (
    JetsonEnergyBackend,
    _trapezoid_energy_j,
)


def _write_rail(hwmon: Path, channel: int, label: str, voltage_mv: float, current_ma: float):
    hwmon.mkdir(parents=True, exist_ok=True)
    (hwmon / f"in{channel}_label").write_text(label, encoding="utf-8")
    (hwmon / f"in{channel}_input").write_text(str(voltage_mv), encoding="utf-8")
    (hwmon / f"curr{channel}_input").write_text(str(current_ma), encoding="utf-8")


class FakeIntegrator:
    def __init__(self, energies):
        self.energies = dict(energies)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return dict(self.energies)


def _factory(energies, captured=None):
    def build(read_powers, interval):
        if captured is not None:
            captured["powers"] = read_powers()
            captured["interval"] = interval
        obj = FakeIntegrator(energies)
        if captured is not None:
            captured["integrator"] = obj
        return obj
    return build


def _orin_nx_tree(root: Path):
    hwmon = root / "1-0040" / "hwmon" / "hwmon7"
    _write_rail(hwmon, 1, "VDD_IN", 5000, 2000)          # 10 W
    _write_rail(hwmon, 2, "VDD_CPU_GPU_CV", 1000, 3000) # 3 W
    _write_rail(hwmon, 3, "VDD_SOC", 1000, 1000)        # 1 W
    return hwmon


def test_jetson_discovers_rails_and_prefers_vdd_in(tmp_path):
    """P7-T1: discover hwmon rails and select total-module VDD_IN when present."""
    _orin_nx_tree(tmp_path)
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        integrator_factory=_factory({"VDD_IN": 1.0, "VDD_CPU_GPU_CV": 0.3, "VDD_SOC": 0.1}),
    )

    assert backend.available() is True
    meta = backend.metadata()
    assert meta["primary_domain"] == "VDD_IN"
    assert meta["primary_domain_scope"] == "total_module_power"
    assert meta["dla_relevant_domains"] == ["VDD_CPU_GPU_CV"]
    assert set(backend.domains()) == {"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"}


def test_jetson_power_conversion_and_domain_integration(tmp_path):
    """P7-T2: voltage/current readings feed independent per-rail integration."""
    _orin_nx_tree(tmp_path)
    captured = {}
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        sample_interval_s=0.05,
        integrator_factory=_factory(
            {"VDD_IN": 2.0, "VDD_CPU_GPU_CV": 0.6, "VDD_SOC": 0.2},
            captured,
        ),
    )

    start = backend.read_energy()
    end = backend.read_energy()

    assert start == 0.0
    assert end == pytest.approx(2.0)
    assert captured["powers"]["VDD_IN"] == pytest.approx(10.0)
    assert captured["powers"]["VDD_CPU_GPU_CV"] == pytest.approx(3.0)
    assert captured["powers"]["VDD_SOC"] == pytest.approx(1.0)
    assert captured["interval"] == pytest.approx(0.05)
    domains = backend.domains()
    assert domains["VDD_IN"]["energy_j"] == pytest.approx(2.0)
    assert domains["VDD_CPU_GPU_CV"]["energy_j"] == pytest.approx(0.6)


def test_jetson_does_not_sum_overlapping_rails(tmp_path):
    """P7-T3: scalar result is primary rail only, never a sum of overlapping rails."""
    _orin_nx_tree(tmp_path)
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        integrator_factory=_factory(
            {"VDD_IN": 4.0, "VDD_CPU_GPU_CV": 3.0, "VDD_SOC": 2.0}
        ),
    )

    assert backend.read_energy() == 0.0
    assert backend.read_energy() == pytest.approx(4.0)
    assert backend.read_energy() == 0.0  # new measurement window resets cleanly


def test_jetson_requested_missing_primary_is_unavailable(tmp_path):
    """P7-T4: an explicitly requested absent rail is reported without crashing."""
    _orin_nx_tree(tmp_path)
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        primary_domain="NOT_PRESENT",
        integrator_factory=_factory({"VDD_IN": 1.0}),
    )

    assert backend.available() is False
    assert "NOT_PRESENT" in backend.unavailable_reason()


def test_jetson_non_linux_is_unavailable(tmp_path):
    _orin_nx_tree(tmp_path)
    backend = JetsonEnergyBackend(tmp_path, system_name="Windows")

    assert backend.available() is False
    assert "requires Linux" in backend.unavailable_reason()


def test_jetson_energy_meter_integration(tmp_path):
    _orin_nx_tree(tmp_path)
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        integrator_factory=_factory(
            {"VDD_IN": 2.0, "VDD_CPU_GPU_CV": 0.8, "VDD_SOC": 0.2}
        ),
    )
    meter = EnergyMeter(backend=backend, clock=iter([10.0, 12.0]).__next__)
    calls = 0

    def infer():
        nonlocal calls
        calls += 1

    result = meter.measure(infer, repeats=4, batch_size=2)

    assert result["available"] is True
    assert result["backend"] == "jetson_ina3221"
    assert result["total_energy_j"] == pytest.approx(2.0)
    assert result["energy_per_inference_j"] == pytest.approx(0.5)
    assert result["energy_per_sample_j"] == pytest.approx(0.25)
    assert result["average_power_w"] == pytest.approx(1.0)
    assert result["metadata"]["primary_domain"] == "VDD_IN"
    assert result["domains"]["VDD_CPU_GPU_CV"]["dla_relevant"] is True
    assert calls == 4


def test_jetson_agx_orin_summation_is_not_claimed_as_total_module_power(tmp_path):
    hwmon = tmp_path / "1-0040" / "hwmon" / "hwmon2"
    _write_rail(hwmon, 1, "VDD_GPU_SOC", 1000, 2000)
    _write_rail(hwmon, 2, "VDD_CPU_CV", 1000, 3000)
    _write_rail(hwmon, 4, "SUMMATION", 1000, 5000)
    backend = JetsonEnergyBackend(
        tmp_path,
        system_name="Linux",
        integrator_factory=_factory(
            {"VDD_GPU_SOC": 0.4, "VDD_CPU_CV": 0.6, "SUMMATION": 1.0}
        ),
    )

    assert backend.available() is True
    metadata = backend.metadata()
    assert metadata["primary_domain"] == "SUMMATION"
    assert "not_guaranteed_total_module_power" in metadata["primary_domain_scope"]
    assert metadata["dla_relevant_domains"] == ["VDD_CPU_CV"]


def test_jetson_duplicate_rail_labels_are_rejected(tmp_path):
    first = tmp_path / "1-0040" / "hwmon" / "hwmon1"
    second = tmp_path / "1-0041" / "hwmon" / "hwmon2"
    _write_rail(first, 1, "VDD_IN", 5000, 1000)
    _write_rail(second, 1, "VDD_IN", 5000, 1000)
    backend = JetsonEnergyBackend(tmp_path, system_name="Linux")

    assert backend.available() is False
    assert "Duplicate Jetson rail label" in backend.unavailable_reason()


def test_jetson_invalid_sensor_data_is_recoverable(tmp_path):
    hwmon = tmp_path / "1-0040" / "hwmon" / "hwmon1"
    _write_rail(hwmon, 1, "VDD_IN", 5000, -1)
    backend = JetsonEnergyBackend(tmp_path, system_name="Linux")

    assert backend.available() is False
    assert "invalid voltage/current" in backend.unavailable_reason()


def test_jetson_validates_sample_interval():
    with pytest.raises(ValueError, match="sample_interval_s"):
        JetsonEnergyBackend(sample_interval_s=0.0)


def test_jetson_trapezoid_integration_math():
    """Constantly test the numerical rule independently of threads and hardware."""
    assert _trapezoid_energy_j(2.0, 4.0, 2.0) == pytest.approx(6.0)
    assert _trapezoid_energy_j(5.0, 5.0, 0.5) == pytest.approx(2.5)


def test_jetson_trapezoid_rejects_invalid_values():
    with pytest.raises(EnergyBackendError):
        _trapezoid_energy_j(-1.0, 1.0, 1.0)
    with pytest.raises(EnergyBackendError):
        _trapezoid_energy_j(1.0, 1.0, float("nan"))
