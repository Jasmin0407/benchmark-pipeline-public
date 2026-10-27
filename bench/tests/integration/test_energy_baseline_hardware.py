"""Optional Phase 13 validation against real baseline-corrected results."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.tools.energy_baseline_validation import (
    analyze_baseline_results,
    discover_baseline_results,
)


@pytest.mark.hardware
@pytest.mark.energy
def test_phase13_real_baseline_correction_is_plausible():
    """Validate user-supplied Phase 13 bench.json files from one target system."""

    results_root = os.environ.get("BENCH_ENERGY_BASELINE_ROOT")
    if not results_root:
        pytest.skip(
            "Optional Phase 13 hardware gate: set BENCH_ENERGY_BASELINE_ROOT "
            "to the root containing the real baseline-corrected benchmark results"
        )

    paths = discover_baseline_results(Path(results_root))
    report = analyze_baseline_results(
        paths,
        expected_counts=(100, 500, 1000),
    )

    assert report.comparable is True
    assert report.gross_total_energy_monotonic is True
    assert report.adjusted_total_energy_monotonic is True
    assert len(report.points) == 3
