from __future__ import annotations

import platform
import time

import pytest

from bench.core.metrics.jetson_energy_backend import JetsonEnergyBackend


@pytest.mark.hardware
@pytest.mark.energy
def test_jetson_ina3221_on_real_hardware():
    """P7-T5/P7-T6: integrate real Jetson rails and expose DLA-relevant domains."""
    if platform.system() != "Linux":
        pytest.skip("NVIDIA Jetson INA3221 hardware test requires Linux")

    backend = JetsonEnergyBackend(sample_interval_s=0.02)
    if not backend.available():
        pytest.skip(backend.unavailable_reason())

    start_j = backend.read_energy()
    time.sleep(0.25)
    end_j = backend.read_energy()

    assert start_j == pytest.approx(0.0)
    assert end_j > 0.0

    metadata = backend.metadata()
    domains = backend.domains()
    assert metadata["primary_domain"] in domains
    assert any("energy_j" in value for value in domains.values())

    # On Orin/Xavier variants exposing CV rails, keep them visible for later DLA validation.
    for domain in metadata["dla_relevant_domains"]:
        assert domain in domains
        assert domains[domain]["dla_relevant"] is True
