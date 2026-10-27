from __future__ import annotations

from collections.abc import Iterable

import pytest

from bench.core.metrics.energy_meter import EnergyMeter


class SequenceReader:
    """Return deterministic values from a finite sequence."""

    def __init__(self, values: Iterable[float]):
        self._values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return float(next(self._values))


class CountingInference:
    """Count how often the synthetic inference function is called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def make_meter(
    energy_values: Iterable[float],
    clock_values: Iterable[float] = (10.0, 12.0),
) -> tuple[EnergyMeter, SequenceReader, SequenceReader]:
    energy_reader = SequenceReader(energy_values)
    clock = SequenceReader(clock_values)
    meter = EnergyMeter(energy_reader=energy_reader, clock=clock)
    return meter, energy_reader, clock


def test_total_energy_is_counter_difference():
    """P3-T1: 1.0 J to 3.5 J must yield 2.5 J total energy."""
    meter, _, _ = make_meter((1.0, 3.5))

    result = meter.measure(lambda: None, repeats=1)

    assert result["total_energy_j"] == pytest.approx(2.5)


def test_energy_per_inference_is_normalized_by_repeat_count():
    """P3-T2: 10 J across 100 calls must yield 0.1 J per inference."""
    meter, _, _ = make_meter((5.0, 15.0))

    result = meter.measure(lambda: None, repeats=100)

    assert result["energy_per_inference_j"] == pytest.approx(0.1)
    assert result["inference_count"] == 100


def test_energy_per_sample_is_batch_normalized():
    """P3-T3: 0.16 J per inference at batch 16 is 0.01 J per sample."""
    meter, _, _ = make_meter((2.0, 2.16))

    result = meter.measure(lambda: None, repeats=1, batch_size=16)

    assert result["energy_per_inference_j"] == pytest.approx(0.16)
    assert result["energy_per_sample_j"] == pytest.approx(0.01)


def test_average_power_uses_measured_window_duration():
    """P3-T4: 10 J over 2 seconds must yield 5 W average power."""
    meter, _, _ = make_meter(
        energy_values=(100.0, 110.0),
        clock_values=(20.0, 22.0),
    )

    result = meter.measure(lambda: None, repeats=10)

    assert result["duration_s"] == pytest.approx(2.0)
    assert result["average_power_w"] == pytest.approx(5.0)


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_invalid_repeat_count_is_rejected_before_measurement(repeats):
    """P3-T5: invalid repeat counts must fail before any inference starts."""
    inference = CountingInference()
    meter, energy_reader, _ = make_meter((0.0, 1.0))

    with pytest.raises(ValueError, match="repeats"):
        meter.measure(inference, repeats=repeats)

    assert inference.calls == 0
    assert energy_reader.calls == 0


def test_measure_executes_exactly_the_requested_number_of_inferences():
    """The dedicated energy window must execute exactly ``repeats`` calls."""
    inference = CountingInference()
    meter, energy_reader, clock = make_meter((4.0, 5.0))

    result = meter.measure(inference, repeats=37, batch_size=4)

    assert inference.calls == 37
    assert result["inference_count"] == 37
    assert energy_reader.calls == 2
    assert clock.calls == 2


def test_optional_signal_duration_reports_energy_per_signal_second():
    """Energy per signal-second is sample energy divided by signal duration."""
    meter, _, _ = make_meter((1.0, 2.0))

    result = meter.measure(
        lambda: None,
        repeats=10,
        batch_size=2,
        signal_duration_s=5.0,
    )

    # 1 J / 10 inferences / 2 samples / 5 s = 0.01 J per signal-second.
    assert result["energy_per_signal_s_j"] == pytest.approx(0.01)


def test_signal_metric_is_omitted_when_duration_is_unknown():
    """Do not invent a signal-normalized metric without signal duration."""
    meter, _, _ = make_meter((1.0, 2.0))

    result = meter.measure(lambda: None, repeats=10, batch_size=2)

    assert "energy_per_signal_s_j" not in result


def test_unconfigured_meter_reports_unavailable_and_cannot_measure():
    """Availability reflects whether a cumulative joule reader was injected."""
    meter = EnergyMeter()

    assert meter.available() is False
    with pytest.raises(RuntimeError, match="No energy reader"):
        meter.measure(lambda: None, repeats=1)


def test_configured_meter_reports_available():
    meter, _, _ = make_meter((0.0, 1.0))

    assert meter.available() is True


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected(batch_size):
    meter, energy_reader, _ = make_meter((0.0, 1.0))

    with pytest.raises(ValueError, match="batch_size"):
        meter.measure(lambda: None, repeats=1, batch_size=batch_size)

    assert energy_reader.calls == 0


@pytest.mark.parametrize(
    "signal_duration_s",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_signal_duration_is_rejected(signal_duration_s):
    meter, energy_reader, _ = make_meter((0.0, 1.0))

    with pytest.raises(ValueError, match="signal_duration_s"):
        meter.measure(
            lambda: None,
            repeats=1,
            signal_duration_s=signal_duration_s,
        )

    assert energy_reader.calls == 0


@pytest.mark.parametrize(
    "energy_values",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (-1.0, 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, -1.0),
    ],
)
def test_invalid_energy_counter_values_are_rejected(energy_values):
    meter, _, _ = make_meter(energy_values)

    with pytest.raises(ValueError, match="energy counter"):
        meter.measure(lambda: None, repeats=1)


def test_decreasing_counter_is_rejected_for_backend_wraparound_handling():
    """Generic meter must not guess how a hardware counter wraps around."""
    meter, _, _ = make_meter((10.0, 2.0))

    with pytest.raises(ValueError, match="counter decreased"):
        meter.measure(lambda: None, repeats=1)


@pytest.mark.parametrize(
    "clock_values",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 2.0),
        (1.0, float("inf")),
    ],
)
def test_invalid_measurement_duration_is_rejected(clock_values):
    meter, _, _ = make_meter((0.0, 1.0), clock_values=clock_values)

    with pytest.raises(ValueError, match="clock|duration"):
        meter.measure(lambda: None, repeats=1)


def test_inference_exception_is_not_swallowed():
    """A model failure must remain visible instead of producing fake metrics."""
    meter, _, _ = make_meter((0.0, 1.0))

    def failing_inference() -> None:
        raise RuntimeError("synthetic inference failure")

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        meter.measure(failing_inference, repeats=3)


def test_measure_idle_reports_energy_and_power_without_inference():
    meter, energy_reader, clock = make_meter(
        energy_values=(10.0, 14.0),
        clock_values=(100.0, 102.0),
    )
    slept = []

    result = meter.measure_idle(2.0, sleep_fn=lambda seconds: slept.append(seconds))

    assert slept == [2.0]
    assert energy_reader.calls == 2
    assert clock.calls == 2
    assert result["total_energy_j"] == pytest.approx(4.0)
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["average_power_w"] == pytest.approx(2.0)


@pytest.mark.parametrize("duration_s", [0, -1.0, True, float("nan")])
def test_measure_idle_rejects_invalid_duration(duration_s):
    meter, energy_reader, _ = make_meter((0.0, 1.0))

    with pytest.raises(ValueError, match="duration_s"):
        meter.measure_idle(duration_s, sleep_fn=lambda _: None)

    assert energy_reader.calls == 0


def test_measure_idle_prefers_backend_counter_duration():
    class FakeBackend:
        name = "fake_backend"

        def __init__(self):
            self.energy = iter([100.0, 106.0])
            self.times = iter([20.0, 22.0])
            self.last_time = None

        def available(self):
            return True

        def unavailable_reason(self):
            return ""

        def read_energy(self):
            self.last_time = next(self.times)
            return next(self.energy)

        def last_sample_time_s(self):
            return self.last_time

        def metadata(self):
            return {"primary_domain": "package"}

        def domains(self):
            return {}

    clock = SequenceReader((10.0, 13.0))
    meter = EnergyMeter(backend=FakeBackend(), clock=clock)

    result = meter.measure_idle(2.0, sleep_fn=lambda _: None)

    assert result["duration_source"] == "backend_counter"
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["wall_clock_duration_s"] == pytest.approx(3.0)
    assert result["average_power_w"] == pytest.approx(3.0)
    assert result["primary_domain"] == "package"
