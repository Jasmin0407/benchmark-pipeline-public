"""Optional Phase 12 validation against repeated real hardware results."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.tools.energy_repeatability import (
    DEFAULT_SCHEDULE_ID,
    analyze_repeatability_results,
    discover_repeatability_results,
)


@pytest.mark.hardware
@pytest.mark.energy
def test_phase12_real_energy_results_are_repeatable():
    """Validate user-supplied repeated bench.json files from one target system.

    Set ``BENCH_ENERGY_REPEATABILITY_RESULTS`` to the 18 Phase 12 bench.json
    paths separated by the platform path separator (``;`` on Windows, ``:`` on
    Linux). Optionally set ``BENCH_ENERGY_REPEATABILITY_MAX_CV_PCT`` to a
    project-specific acceptance limit for the sample CV of energy per inference.

    The test is intentionally opt-in because normal CI/development machines do
    not possess the real Dell hardware result files.
    """

    raw_paths = os.environ.get("BENCH_ENERGY_REPEATABILITY_RESULTS")
    results_root = os.environ.get("BENCH_ENERGY_REPEATABILITY_ROOT")
    schedule_id = os.environ.get(
        "BENCH_ENERGY_REPEATABILITY_SCHEDULE_ID",
        DEFAULT_SCHEDULE_ID,
    )

    if raw_paths:
        paths = [Path(value) for value in raw_paths.split(os.pathsep) if value.strip()]
    elif results_root:
        paths = discover_repeatability_results(
            Path(results_root),
            schedule_id=schedule_id,
        )
    else:
        pytest.skip(
            "Optional Phase 12 hardware gate: set BENCH_ENERGY_REPEATABILITY_ROOT "
            "or BENCH_ENERGY_REPEATABILITY_RESULTS to real hardware results"
        )
    threshold_raw = os.environ.get("BENCH_ENERGY_REPEATABILITY_MAX_CV_PCT")
    threshold = None if threshold_raw in (None, "") else float(threshold_raw)

    report = analyze_repeatability_results(
        paths,
        expected_repetitions=6,
        expected_counts=(100, 500, 1000),
        max_cv_pct=threshold,
    )

    assert report.comparable is True
    assert len(report.observations) == 18
    if threshold is not None:
        assert report.cv_acceptance_passed is True
        assert report.overall_status == "passed"
