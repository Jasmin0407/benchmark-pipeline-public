"""Optional Phase 11 validation against real hardware benchmark results."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.tools.energy_plausibility import analyze_plausibility_results


@pytest.mark.hardware
@pytest.mark.energy
def test_phase11_real_energy_results_are_physically_plausible():
    """Validate user-supplied real bench.json files from one target system.

    Set ``BENCH_ENERGY_PLAUSIBILITY_RESULTS`` to paths separated by the platform
    path separator (``;`` on Windows, ``:`` on Linux). Optionally set
    ``BENCH_ENERGY_PLAUSIBILITY_MAX_SPREAD_PCT`` to enforce a laboratory-specific
    normalized-energy tolerance.
    """

    raw_paths = os.environ.get("BENCH_ENERGY_PLAUSIBILITY_RESULTS")
    if not raw_paths:
        pytest.skip(
            "Optional Phase 11 hardware gate: set BENCH_ENERGY_PLAUSIBILITY_RESULTS "
            "to the real 100/500/1000 bench.json files"
        )

    paths = [Path(value) for value in raw_paths.split(os.pathsep) if value.strip()]
    if len(paths) < 2:
        pytest.fail("BENCH_ENERGY_PLAUSIBILITY_RESULTS must contain at least two bench.json files")

    tolerance_raw = os.environ.get("BENCH_ENERGY_PLAUSIBILITY_MAX_SPREAD_PCT")
    tolerance = None if tolerance_raw in (None, "") else float(tolerance_raw)

    report = analyze_plausibility_results(paths, max_relative_spread_pct=tolerance)

    assert report.total_energy_monotonic is True
    if tolerance is not None:
        assert report.normalized_stability_passed is True
        assert report.overall_status == "passed"
