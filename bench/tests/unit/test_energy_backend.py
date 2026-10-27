from __future__ import annotations

from collections.abc import Iterable

import pytest

from bench.core.metrics.energy_backend import EnergyBackend
from bench.core.metrics.energy_meter import EnergyMeter


class SequenceBackend(EnergyBackend):
    """Deterministic fake backend used to test the Phase 4 contract."""

    name = "fake"

    def __init__(
        self,
        values: Iterable[float] = (1.0, 3.0),
        *,
        is_available: bool = True,
        reason: str = "Fake backend unavailable.",
        read_error: Exception | None = None,
    ) -> None:
        self._values = iter(values)
        self._is_available = is_available
        self._reason = reason
        self._read_error = read_error
        self.read_calls = 0

    def available(self) -> bool:
        return self._is_available

    def read_energy(self) -> float:
        self.read_calls += 1
        if self._read_error is not None:
            raise self._read_error
        return float(next(self._values))

    def unavailable_reason(self) -> str:
        return self._reason

    def metadata(self):
        return {"source": "fake_counter"}

    def domains(self):
        return {"package": {"unit": "J"}}


class FailingAvailabilityBackend(SequenceBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def available(self) -> bool:
        raise self._error


class SequenceClock:
    def __init__(self, values: Iterable[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return float(next(self._values))


def test_backend_contract_available():
    """P4-T1: an available backend can be selected by the EnergyMeter."""
    backend = SequenceBackend(is_available=True)
    meter = EnergyMeter(backend=backend)

    assert backend.available() is True
    assert meter.available() is True


def test_unavailable_backend_returns_non_fatal_result_without_inference():
    """P4-T2: unavailable hardware must not abort or execute inference."""
    backend = SequenceBackend(
        is_available=False,
        reason="No supported energy measurement backend available.",
    )
    meter = EnergyMeter(backend=backend)
    inference_calls = 0

    def infer() -> None:
        nonlocal inference_calls
        inference_calls += 1

    result = meter.measure(infer, repeats=5)

    assert result == {
        "available": False,
        "backend": "fake",
        "reason": "No supported energy measurement backend available.",
    }
    assert inference_calls == 0
    assert backend.read_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("permission denied"),
        FileNotFoundError("counter missing"),
    ],
)
def test_backend_read_errors_are_reported_without_benchmark_abort(error):
    """P4-T3: recoverable hardware-read errors become unavailable results."""
    backend = SequenceBackend(read_error=error)
    meter = EnergyMeter(backend=backend, clock=SequenceClock((1.0, 2.0)))

    result = meter.measure(lambda: None, repeats=3)

    assert result["available"] is False
    assert result["backend"] == "fake"
    assert "Energy counter could not be read" in result["reason"]
    assert backend.read_calls == 1


def test_backend_availability_errors_are_reported_without_abort():
    backend = FailingAvailabilityBackend(PermissionError("blocked"))
    meter = EnergyMeter(backend=backend)

    result = meter.measure(lambda: None, repeats=1)

    assert result["available"] is False
    assert result["backend"] == "fake"
    assert "availability check failed" in result["reason"]


def test_available_backend_produces_standardized_energy_result():
    backend = SequenceBackend(values=(10.0, 14.0))
    meter = EnergyMeter(backend=backend, clock=SequenceClock((5.0, 7.0)))
    inference_calls = 0

    def infer() -> None:
        nonlocal inference_calls
        inference_calls += 1

    result = meter.measure(infer, repeats=4, batch_size=2)

    assert result["available"] is True
    assert result["backend"] == "fake"
    assert result["measurement_scope"] == "dedicated_inference_window"
    assert result["total_energy_j"] == pytest.approx(4.0)
    assert result["energy_per_inference_j"] == pytest.approx(1.0)
    assert result["energy_per_sample_j"] == pytest.approx(0.5)
    assert result["average_power_w"] == pytest.approx(2.0)
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["inference_count"] == 4
    assert result["metadata"] == {"source": "fake_counter"}
    assert result["domains"] == {"package": {"unit": "J"}}
    assert inference_calls == 4
    assert backend.read_calls == 2


def test_backend_and_legacy_reader_are_mutually_exclusive():
    backend = SequenceBackend()

    with pytest.raises(ValueError, match="either energy_reader or backend"):
        EnergyMeter(energy_reader=lambda: 1.0, backend=backend)


def test_phase3_legacy_reader_api_remains_supported():
    values = iter((1.0, 2.0))
    clock = SequenceClock((10.0, 11.0))
    meter = EnergyMeter(energy_reader=lambda: next(values), clock=clock)

    result = meter.measure(lambda: None, repeats=2)

    assert "available" not in result
    assert result["total_energy_j"] == pytest.approx(1.0)
    assert result["energy_per_inference_j"] == pytest.approx(0.5)
