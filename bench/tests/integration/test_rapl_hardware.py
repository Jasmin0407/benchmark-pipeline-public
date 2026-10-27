from __future__ import annotations

import platform

import pytest

from bench.core.metrics.rapl_energy_backend import RaplEnergyBackend


@pytest.mark.hardware
@pytest.mark.energy
def test_rapl_energy_counter_on_real_cpu():
    """P5-T5: sanity-check a real Linux RAPL counter when hardware is available."""
    if platform.system() != "Linux":
        pytest.skip("Intel RAPL sysfs hardware test requires Linux")

    backend = RaplEnergyBackend()
    if not backend.available():
        pytest.skip(backend.unavailable_reason())

    first_j = backend.read_energy()
    second_j = backend.read_energy()

    assert first_j >= 0.0
    assert second_j >= first_j
    assert backend.metadata()["normalized_unit"] == "joule"
