from __future__ import annotations

import pytest

from bench.core.metrics.energy_backend import EnergyBackendError
from bench.core.metrics.energy_meter import EnergyMeter
from bench.core.metrics.nvml_energy_backend import NvmlEnergyBackend


class FakeNVMLError(Exception):
    pass


class FakeNotSupported(FakeNVMLError):
    pass


class FakeNvml:
    NVMLError = FakeNVMLError
    NVMLError_NotSupported = FakeNotSupported

    def __init__(self, energies_mj=None):
        self.energies_mj = list(energies_mj or [1000.0])
        self.init_calls = 0
        self.handle_calls = 0
        self.energy_calls = 0
        self.handle = object()
        self.energy_error = None
        self.init_error = None

    def nvmlInit(self):
        self.init_calls += 1
        if self.init_error is not None:
            raise self.init_error

    def nvmlDeviceGetHandleByIndex(self, index):
        self.handle_calls += 1
        if index != 0:
            raise FakeNVMLError("invalid device")
        return self.handle

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        assert handle is self.handle
        self.energy_calls += 1
        if self.energy_error is not None:
            raise self.energy_error
        if len(self.energies_mj) > 1:
            return self.energies_mj.pop(0)
        return self.energies_mj[0]

    def nvmlDeviceGetName(self, handle):
        return b"Fake NVIDIA GPU"

    def nvmlDeviceGetUUID(self, handle):
        return b"GPU-FAKE-UUID"


def test_nvml_converts_millijoules_to_joules():
    """P6-T1: native NVML millijoules are normalized to joules."""
    fake = FakeNvml([2500])
    backend = NvmlEnergyBackend(nvml_module=fake)

    assert backend.read_energy() == pytest.approx(2.5)


def test_nvml_available_when_total_energy_counter_is_supported():
    fake = FakeNvml([1200])
    backend = NvmlEnergyBackend(nvml_module=fake)

    assert backend.available() is True
    assert fake.init_calls == 1
    assert fake.handle_calls == 1


def test_nvml_not_supported_is_nonfatal():
    """P6-T2: unsupported GPUs are reported as unavailable."""
    fake = FakeNvml()
    fake.energy_error = FakeNotSupported("not supported")
    backend = NvmlEnergyBackend(nvml_module=fake)

    assert backend.available() is False
    assert "does not support" in backend.unavailable_reason()


def test_nvml_not_supported_integrates_with_energy_meter():
    fake = FakeNvml()
    fake.energy_error = FakeNotSupported("not supported")
    backend = NvmlEnergyBackend(nvml_module=fake)
    meter = EnergyMeter(backend=backend, clock=iter([0.0, 1.0]).__next__)
    inference_calls = 0

    def infer():
        nonlocal inference_calls
        inference_calls += 1

    result = meter.measure(infer, repeats=5)

    assert result["available"] is False
    assert result["backend"] == "nvml"
    assert inference_calls == 0


def test_nvml_read_error_becomes_backend_error():
    """P6-T3: generic NVML read failures become recoverable backend errors."""
    fake = FakeNvml()
    fake.energy_error = FakeNVMLError("GPU is lost")
    backend = NvmlEnergyBackend(nvml_module=fake)

    with pytest.raises(EnergyBackendError, match="Could not read NVML energy counter"):
        backend.read_energy()


def test_nvml_energy_meter_turns_read_error_into_unavailable_result():
    fake = FakeNvml([1000, 2000])
    backend = NvmlEnergyBackend(nvml_module=fake)
    meter = EnergyMeter(backend=backend, clock=iter([0.0, 2.0]).__next__)

    # available() consumes one valid probe; make the subsequent start read fail.
    original = fake.nvmlDeviceGetTotalEnergyConsumption
    calls = 0

    def fail_after_probe(handle):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1000
        raise FakeNVMLError("read failure")

    fake.nvmlDeviceGetTotalEnergyConsumption = fail_after_probe

    result = meter.measure(lambda: None, repeats=2)

    assert result["available"] is False
    assert "Energy counter could not be read" in result["reason"]


def test_nvml_successfully_integrates_with_energy_meter():
    fake = FakeNvml([1000, 1000, 3000])
    backend = NvmlEnergyBackend(nvml_module=fake)
    meter = EnergyMeter(backend=backend, clock=iter([10.0, 12.0]).__next__)
    inference_calls = 0

    def infer():
        nonlocal inference_calls
        inference_calls += 1

    result = meter.measure(infer, repeats=4, batch_size=2)

    assert result["available"] is True
    assert result["backend"] == "nvml"
    assert result["total_energy_j"] == pytest.approx(2.0)
    assert result["energy_per_inference_j"] == pytest.approx(0.5)
    assert result["energy_per_sample_j"] == pytest.approx(0.25)
    assert result["average_power_w"] == pytest.approx(1.0)
    assert inference_calls == 4


def test_nvml_initializes_only_once_for_repeated_reads():
    fake = FakeNvml([1000, 2000])
    backend = NvmlEnergyBackend(nvml_module=fake)

    backend.read_energy()
    backend.read_energy()

    assert fake.init_calls == 1
    assert fake.handle_calls == 1


def test_nvml_metadata_is_nonfatal_and_normalized():
    fake = FakeNvml()
    backend = NvmlEnergyBackend(nvml_module=fake)

    metadata = backend.metadata()

    assert metadata["source"] == "nvml_total_energy_counter"
    assert metadata["native_unit"] == "millijoule"
    assert metadata["normalized_unit"] == "joule"
    assert metadata["device_index"] == 0
    assert metadata["device_name"] == "Fake NVIDIA GPU"
    assert metadata["device_uuid"] == "GPU-FAKE-UUID"


def test_nvml_rejects_invalid_device_index():
    with pytest.raises(ValueError, match="device_index"):
        NvmlEnergyBackend(device_index=-1, nvml_module=FakeNvml())

    with pytest.raises(ValueError, match="device_index"):
        NvmlEnergyBackend(device_index=True, nvml_module=FakeNvml())


def test_nvml_invalid_device_is_unavailable():
    backend = NvmlEnergyBackend(device_index=1, nvml_module=FakeNvml())

    assert backend.available() is False
    assert "Could not initialize NVML device 1" in backend.unavailable_reason()
