from __future__ import annotations

import pytest

from bench.core.metrics.energy_window import determine_energy_repeats


def test_energy_repeats_cover_minimum_duration():
    """P2-T1: 10 ms latency and a 2 s window require 200 repetitions."""
    repeats = determine_energy_repeats(
        mean_inference_ms=10.0,
        min_duration_s=2.0,
        run_repeats=10,
        max_repeats=10000,
    )

    assert repeats == 200


def test_energy_repeats_respect_regular_run_repeats():
    """P2-T2: the energy window must not use fewer regular repetitions."""
    repeats = determine_energy_repeats(
        mean_inference_ms=10.0,
        min_duration_s=2.0,
        run_repeats=500,
        max_repeats=10000,
    )

    assert repeats == 500


def test_energy_repeats_apply_hard_maximum():
    """P2-T3: max_repeats must cap very large repetition estimates."""
    repeats = determine_energy_repeats(
        mean_inference_ms=0.01,
        min_duration_s=2.0,
        run_repeats=100,
        max_repeats=10000,
    )

    assert repeats == 10000


def test_energy_repeats_round_up_fractional_requirement():
    """The duration estimate must round up so the target is not undershot."""
    repeats = determine_energy_repeats(
        mean_inference_ms=7.0,
        min_duration_s=2.0,
        run_repeats=1,
        max_repeats=10000,
    )

    assert repeats == 286


def test_max_repeats_is_a_hard_safety_cap():
    """The configured cap takes precedence when it is below run_repeats."""
    repeats = determine_energy_repeats(
        mean_inference_ms=100.0,
        min_duration_s=1.0,
        run_repeats=500,
        max_repeats=250,
    )

    assert repeats == 250


@pytest.mark.parametrize(
    "mean_inference_ms",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_mean_inference_latency_is_rejected(mean_inference_ms):
    """Invalid latency values must fail before any energy measurement starts."""
    with pytest.raises(ValueError, match="mean_inference_ms"):
        determine_energy_repeats(
            mean_inference_ms=mean_inference_ms,
            min_duration_s=2.0,
            run_repeats=10,
            max_repeats=10000,
        )


@pytest.mark.parametrize(
    "min_duration_s",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_minimum_duration_is_rejected(min_duration_s):
    """Invalid duration values must fail deterministically."""
    with pytest.raises(ValueError, match="min_duration_s"):
        determine_energy_repeats(
            mean_inference_ms=10.0,
            min_duration_s=min_duration_s,
            run_repeats=10,
            max_repeats=10000,
        )


@pytest.mark.parametrize("run_repeats", [0, -1, True, 1.5])
def test_invalid_run_repeats_is_rejected(run_repeats):
    """The regular benchmark repeat count must be a positive integer."""
    with pytest.raises(ValueError, match="run_repeats"):
        determine_energy_repeats(
            mean_inference_ms=10.0,
            min_duration_s=2.0,
            run_repeats=run_repeats,
            max_repeats=10000,
        )


@pytest.mark.parametrize("max_repeats", [0, -1, True, 1.5])
def test_invalid_max_repeats_is_rejected(max_repeats):
    """The energy safety cap must be a positive integer."""
    with pytest.raises(ValueError, match="max_repeats"):
        determine_energy_repeats(
            mean_inference_ms=10.0,
            min_duration_s=2.0,
            run_repeats=10,
            max_repeats=max_repeats,
        )
