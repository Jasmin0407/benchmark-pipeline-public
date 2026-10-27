from __future__ import annotations

import struct

import pytest

from bench.core.metrics.energy_backend import EnergyBackendError
from bench.core.metrics.energy_meter import EnergyMeter
from bench.core.metrics.windows_emi_energy_backend import (
    DEFAULT_PACKAGE_CHANNEL,
    EMI_MEASUREMENT_UNIT_PICOWATT_HOURS,
    IOCTL_EMI_GET_MEASUREMENT,
    IOCTL_EMI_GET_METADATA,
    IOCTL_EMI_GET_METADATA_SIZE,
    IOCTL_EMI_GET_VERSION,
    PWH_TO_JOULE,
    EmiChannel,
    EmiDevice,
    EmiMeasurement,
    WindowsEmiEnergyBackend,
    parse_emi_measurements,
    parse_emi_metadata,
)


def _utf16_fixed(value: str, chars: int = 16) -> bytes:
    encoded = (value + "\x00").encode("utf-16-le")
    return encoded[: chars * 2].ljust(chars * 2, b"\x00")


def _utf16_sized(value: str) -> bytes:
    return (value + "\x00").encode("utf-16-le")


def _device(*channels: EmiChannel, path: str = "device0") -> EmiDevice:
    return EmiDevice(
        path=path,
        version=2,
        metadata_size=128,
        hardware_oem="Microsoft",
        hardware_model="PPM",
        hardware_revision=1,
        channels=tuple(channels),
    )


class FakeEmiApi:
    def __init__(self, devices, measurement_sets=()):
        self.devices = tuple(devices)
        self.measurement_sets = iter(measurement_sets)
        self.discover_calls = 0
        self.read_calls = 0

    def discover_devices(self):
        self.discover_calls += 1
        return self.devices

    def read_measurements(self, device):
        self.read_calls += 1
        return next(self.measurement_sets)


def test_ioctl_values_match_emi_header_layout():
    assert IOCTL_EMI_GET_VERSION == 0x00224000
    assert IOCTL_EMI_GET_METADATA_SIZE == 0x00224004
    assert IOCTL_EMI_GET_METADATA == 0x00224008
    assert IOCTL_EMI_GET_MEASUREMENT == 0x0022400C


def test_parse_v2_metadata_with_variable_channel_names():
    channels = [
        (0, "RAPL_Package0_PKG"),
        (0, "RAPL_Package0_PP0"),
    ]
    payload = bytearray()
    payload += _utf16_fixed("Microsoft")
    payload += _utf16_fixed("PPM")
    payload += struct.pack("<HH", 1, len(channels))
    for unit, name in channels:
        encoded = _utf16_sized(name)
        payload += struct.pack("<IH", unit, len(encoded))
        payload += encoded

    oem, model, revision, parsed = parse_emi_metadata(2, bytes(payload))

    assert oem == "Microsoft"
    assert model == "PPM"
    assert revision == 1
    assert [channel.name for channel in parsed] == [name for _, name in channels]
    assert all(channel.measurement_unit == 0 for channel in parsed)


def test_parse_measurements_reads_energy_and_timestamp_pairs():
    raw = struct.pack("<QQQQ", 1234, 10_000_000, 5678, 20_000_000)

    parsed = parse_emi_measurements(raw, 2)

    assert parsed[0] == EmiMeasurement(1234, 10_000_000)
    assert parsed[1] == EmiMeasurement(5678, 20_000_000)


def test_backend_selects_exact_package_channel_and_converts_to_joules():
    api = FakeEmiApi(
        [
            _device(
                EmiChannel("RAPL_Package0_PP0", 0),
                EmiChannel(DEFAULT_PACKAGE_CHANNEL, 0),
            )
        ],
        [
            (
                EmiMeasurement(10, 10_000_000),
                EmiMeasurement(1_000_000_000, 10_000_000),
            )
        ],
    )
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)

    assert backend.available() is True
    assert backend.read_energy() == pytest.approx(1_000_000_000 * PWH_TO_JOULE)
    assert backend.metadata()["primary_domain"] == DEFAULT_PACKAGE_CHANNEL
    assert backend.metadata()["native_unit"] == "picowatt_hours"


def test_backend_uses_emi_absolute_time_for_energy_meter_duration():
    device = _device(EmiChannel(DEFAULT_PACKAGE_CHANNEL, 0))
    api = FakeEmiApi(
        [device],
        [
            (EmiMeasurement(1_000_000_000, 100_000_000),),
            (EmiMeasurement(2_000_000_000, 120_000_000),),
        ],
    )
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)
    clock = iter((50.0, 53.0))

    result = EnergyMeter(backend=backend, clock=lambda: next(clock)).measure(
        lambda: None,
        repeats=10,
        batch_size=2,
    )

    assert result["available"] is True
    assert result["backend"] == "windows_emi"
    assert result["primary_domain"] == DEFAULT_PACKAGE_CHANNEL
    assert result["total_energy_j"] == pytest.approx(3.6)
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["duration_source"] == "backend_counter"
    assert result["wall_clock_duration_s"] == pytest.approx(3.0)
    assert result["average_power_w"] == pytest.approx(1.8)
    assert result["domains"][DEFAULT_PACKAGE_CHANNEL]["interval_energy_j"] == pytest.approx(3.6)


def test_backend_rejects_wrong_measurement_unit():
    api = FakeEmiApi([_device(EmiChannel(DEFAULT_PACKAGE_CHANNEL, 99))])
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)

    assert backend.available() is False
    assert "unsupported measurement unit" in backend.unavailable_reason()


def test_backend_rejects_missing_package_channel():
    api = FakeEmiApi([_device(EmiChannel("RAPL_Package0_PP0", 0))])
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)

    assert backend.available() is False
    assert DEFAULT_PACKAGE_CHANNEL in backend.unavailable_reason()
    assert "RAPL_Package0_PP0" in backend.unavailable_reason()


def test_backend_does_not_silently_aggregate_multiple_matching_devices():
    channel = EmiChannel(DEFAULT_PACKAGE_CHANNEL, EMI_MEASUREMENT_UNIT_PICOWATT_HOURS)
    api = FakeEmiApi([_device(channel, path="a"), _device(channel, path="b")])
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)

    assert backend.available() is False
    assert "multiple devices" in backend.unavailable_reason()


def test_backend_rejects_decreasing_absolute_energy():
    device = _device(EmiChannel(DEFAULT_PACKAGE_CHANNEL, 0))
    api = FakeEmiApi(
        [device],
        [
            (EmiMeasurement(1000, 10_000_000),),
            (EmiMeasurement(900, 20_000_000),),
        ],
    )
    backend = WindowsEmiEnergyBackend(system_name="Windows", api=api)

    backend.read_energy()
    with pytest.raises(EnergyBackendError, match="decreased"):
        backend.read_energy()


def test_backend_rejects_non_windows_without_touching_api():
    class FailingApi:
        def discover_devices(self):
            raise AssertionError("API must not be called")

        def read_measurements(self, device):
            raise AssertionError("API must not be called")

    backend = WindowsEmiEnergyBackend(system_name="Linux", api=FailingApi())

    assert backend.available() is False
    assert "requires Windows" in backend.unavailable_reason()
