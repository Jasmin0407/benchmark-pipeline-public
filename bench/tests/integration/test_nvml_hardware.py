from __future__ import annotations

import pytest

from bench.core.metrics.nvml_energy_backend import NvmlEnergyBackend


@pytest.mark.hardware
@pytest.mark.energy
def test_nvml_total_energy_counter_on_real_gpu():
    """P6-T4: read a real cumulative NVML energy counter when supported."""
    backend = NvmlEnergyBackend(device_index=0)

    if not backend.available():
        pytest.skip(backend.unavailable_reason())

    first_j = backend.read_energy()
    second_j = backend.read_energy()

    assert first_j >= 0.0
    assert second_j >= first_j
