from __future__ import annotations

import platform
import time

import pytest

from bench.core.metrics.windows_emi_energy_backend import (
    DEFAULT_PACKAGE_CHANNEL,
    WindowsEmiEnergyBackend,
)


@pytest.mark.hardware
@pytest.mark.energy
def test_windows_emi_reports_monotonic_package_energy():
    if platform.system().lower() != "windows":
        pytest.skip("Windows EMI hardware test requires Windows.")

    backend = WindowsEmiEnergyBackend()
    if not backend.available():
        pytest.skip(backend.unavailable_reason())

    first = backend.read_energy()
    first_time = backend.last_sample_time_s()
    time.sleep(0.25)
    second = backend.read_energy()
    second_time = backend.last_sample_time_s()

    assert first >= 0.0
    assert second > first
    assert first_time is not None
    assert second_time is not None
    assert second_time > first_time
    assert backend.metadata()["primary_domain"] == DEFAULT_PACKAGE_CHANNEL
