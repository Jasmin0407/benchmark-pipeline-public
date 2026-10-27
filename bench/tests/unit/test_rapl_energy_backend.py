from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.core.metrics.energy_backend import EnergyBackendError
from bench.core.metrics.energy_meter import EnergyMeter
from bench.core.metrics.rapl_energy_backend import (
    RaplEnergyBackend,
    _RaplDomain,
    _TOP_LEVEL_RAPL_RE,
)


def _write_domain(
    root: Path,
    key: str = "intel-rapl:0",
    *,
    directory_name: str = "rapl_package_0",
    energy_uj: int = 1_000_000,
    max_energy_range_uj: int = 10_000_000,
    label: str = "package-0",
) -> _RaplDomain:
    """Create a platform-neutral fake RAPL domain for unit tests.

    Windows does not allow ':' in file or directory names, while real Linux
    RAPL sysfs domains use names such as ``intel-rapl:0``. The logical RAPL
    key is therefore kept separate from the temporary directory name.
    """
    domain_path = root / directory_name
    domain_path.mkdir(parents=True)

    energy_path = domain_path / "energy_uj"
    max_range_path = domain_path / "max_energy_range_uj"
    energy_path.write_text(str(energy_uj), encoding="utf-8")
    max_range_path.write_text(str(max_energy_range_uj), encoding="utf-8")
    (domain_path / "name").write_text(label, encoding="utf-8")

    return _RaplDomain(
        key=key,
        path=domain_path,
        energy_path=energy_path,
        max_range_path=max_range_path,
        label=label,
    )


def _backend_with_domains(tmp_path: Path, *domains: _RaplDomain) -> RaplEnergyBackend:
    """Build a Linux-mode backend with pre-discovered fake domains."""
    backend = RaplEnergyBackend(tmp_path, system_name="Linux")
    backend._domains_cache = list(domains)
    return backend


def test_rapl_real_domain_name_pattern_accepts_packages_and_rejects_subdomains():
    assert _TOP_LEVEL_RAPL_RE.fullmatch("intel-rapl:0") is not None
    assert _TOP_LEVEL_RAPL_RE.fullmatch("intel-rapl:12") is not None
    assert _TOP_LEVEL_RAPL_RE.fullmatch("intel-rapl:0:0") is None
    assert _TOP_LEVEL_RAPL_RE.fullmatch("rapl_package_0") is None


def test_rapl_converts_microjoules_to_joules(tmp_path):
    """P5-T1: native RAPL microjoules are normalized to joules."""
    domain = _write_domain(tmp_path, energy_uj=2_500_000)
    backend = _backend_with_domains(tmp_path, domain)

    assert backend.available() is True
    assert backend.read_energy() == pytest.approx(2.5)


def test_rapl_wraparound_is_reconstructed_as_monotonic_energy(tmp_path):
    """P5-T2: max_energy_range_uj is used to correct counter wraparound."""
    domain = _write_domain(
        tmp_path,
        energy_uj=90,
        max_energy_range_uj=100,
    )
    backend = _backend_with_domains(tmp_path, domain)

    start_j = backend.read_energy()
    domain.energy_path.write_text("20", encoding="utf-8")
    end_j = backend.read_energy()

    assert start_j == pytest.approx(90e-6)
    assert end_j == pytest.approx(120e-6)
    assert end_j - start_j == pytest.approx(30e-6)


def test_rapl_wraparound_integrates_cleanly_with_energy_meter(tmp_path):
    domain = _write_domain(
        tmp_path,
        energy_uj=90,
        max_energy_range_uj=100,
    )
    backend = _backend_with_domains(tmp_path, domain)
    clock = iter((5.0, 6.0))

    def infer() -> None:
        domain.energy_path.write_text("20", encoding="utf-8")

    result = EnergyMeter(backend=backend, clock=lambda: next(clock)).measure(
        infer,
        repeats=1,
    )

    assert result["available"] is True
    assert result["backend"] == "rapl"
    assert result["total_energy_j"] == pytest.approx(30e-6)
    assert result["average_power_w"] == pytest.approx(30e-6)


def test_rapl_missing_powercap_path_is_unavailable(tmp_path):
    """P5-T3: a missing powercap path is reported without pretending availability."""
    missing = tmp_path / "does-not-exist"
    backend = RaplEnergyBackend(missing, system_name="Linux")

    assert backend.available() is False
    assert "does not exist" in backend.unavailable_reason()

    with pytest.raises(EnergyBackendError, match="does not exist"):
        backend.read_energy()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows cannot create real Linux RAPL directory names containing ':'.",
)
def test_rapl_missing_counter_files_are_not_discovered(tmp_path):
    """Discovery requires both energy_uj and max_energy_range_uj files."""
    domain = tmp_path / "intel-rapl:0"
    domain.mkdir()
    (domain / "energy_uj").write_text("1", encoding="utf-8")

    backend = RaplEnergyBackend(tmp_path, system_name="Linux")

    assert backend.available() is False
    assert "No top-level Intel RAPL" in backend.unavailable_reason()


def test_rapl_permission_error_becomes_backend_error(tmp_path, monkeypatch):
    """P5-T4: missing read permission becomes a recoverable backend error."""
    domain = _write_domain(tmp_path)
    backend = _backend_with_domains(tmp_path, domain)

    def deny_read(path: Path) -> int:
        raise PermissionError(f"permission denied: {path}")

    monkeypatch.setattr(backend, "_read_nonnegative_integer", deny_read)

    with pytest.raises(EnergyBackendError, match="permission denied"):
        backend.read_energy()


def test_rapl_energy_meter_turns_permission_error_into_unavailable_result(
    tmp_path,
    monkeypatch,
):
    domain = _write_domain(tmp_path)
    backend = _backend_with_domains(tmp_path, domain)

    def deny_read(path: Path) -> int:
        raise PermissionError(f"permission denied: {path}")

    monkeypatch.setattr(backend, "_read_nonnegative_integer", deny_read)
    result = EnergyMeter(backend=backend).measure(lambda: None, repeats=1)

    assert result["available"] is False
    assert result["backend"] == "rapl"
    assert "Energy counter could not be read" in result["reason"]


def test_rapl_rejects_non_linux_even_if_domains_are_injected(tmp_path):
    domain = _write_domain(tmp_path)
    backend = RaplEnergyBackend(tmp_path, system_name="Windows")
    backend._domains_cache = [domain]

    assert backend.available() is False
    assert "only supported on Linux" in backend.unavailable_reason()

    with pytest.raises(EnergyBackendError, match="only supported on Linux"):
        backend.read_energy()


def test_rapl_sums_multiple_packages_and_corrects_wrap_per_package(tmp_path):
    package0 = _write_domain(
        tmp_path,
        "intel-rapl:0",
        directory_name="rapl_package_0",
        energy_uj=90,
        max_energy_range_uj=100,
        label="package-0",
    )
    package1 = _write_domain(
        tmp_path,
        "intel-rapl:1",
        directory_name="rapl_package_1",
        energy_uj=40,
        max_energy_range_uj=100,
        label="package-1",
    )
    backend = _backend_with_domains(tmp_path, package0, package1)

    start_j = backend.read_energy()
    package0.energy_path.write_text("20", encoding="utf-8")
    package1.energy_path.write_text("70", encoding="utf-8")
    end_j = backend.read_energy()

    assert start_j == pytest.approx(130e-6)
    assert end_j == pytest.approx(190e-6)
    assert end_j - start_j == pytest.approx(60e-6)


def test_rapl_ignores_subdomains_to_avoid_double_counting(tmp_path):
    package = _write_domain(
        tmp_path,
        "intel-rapl:0",
        directory_name="rapl_package_0",
        energy_uj=1_000_000,
    )
    backend = _backend_with_domains(tmp_path, package)

    assert backend.read_energy() == pytest.approx(1.0)
    assert list(backend.domains()) == ["intel-rapl:0"]
    assert _TOP_LEVEL_RAPL_RE.fullmatch("intel-rapl:0:0") is None


def test_rapl_metadata_describes_normalized_units(tmp_path):
    domain = _write_domain(tmp_path)
    backend = _backend_with_domains(tmp_path, domain)

    assert backend.metadata() == {
        "source": "linux_powercap_sysfs",
        "native_unit": "microjoule",
        "normalized_unit": "joule",
        "package_count": 1,
    }
    assert backend.domains()["intel-rapl:0"]["label"] == "package-0"
