"""Utilities for planning a stable energy-measurement inference window.

This module intentionally contains no hardware access. It converts the already
measured mean inference latency and the configured energy-window constraints
into a deterministic number of inference repetitions.
"""

from __future__ import annotations

import math


def determine_energy_repeats(
    mean_inference_ms: float,
    min_duration_s: float,
    run_repeats: int,
    max_repeats: int,
) -> int:
    """Return the number of inference calls for the dedicated energy window.

    The calculation follows the Phase 2 specification:

    1. Estimate the repetitions needed to cover ``min_duration_s`` using the
       already measured mean inference latency.
    2. Never request fewer repetitions than the regular benchmark uses.
    3. Apply ``max_repeats`` as a hard safety cap.

    Parameters
    ----------
    mean_inference_ms:
        Mean latency of one inference in milliseconds. Must be finite and > 0.
    min_duration_s:
        Desired minimum duration of the energy window in seconds. Must be
        finite and > 0.
    run_repeats:
        Number of measured repetitions used by the regular benchmark. Must be
        an integer >= 1.
    max_repeats:
        Hard upper bound for the energy-measurement repetitions. Must be an
        integer >= 1.

    Returns
    -------
    int
        Deterministic number of energy-window inference repetitions.

    Notes
    -----
    If ``max_repeats`` is smaller than the number required to satisfy either
    the minimum duration or ``run_repeats``, the hard cap takes precedence.
    In that case the requested minimum duration may not be reached. This is
    intentional: ``max_repeats`` protects the benchmark from unexpectedly
    long energy-measurement runs.
    """
    mean_ms = float(mean_inference_ms)
    min_s = float(min_duration_s)

    if not math.isfinite(mean_ms) or mean_ms <= 0.0:
        raise ValueError("mean_inference_ms must be a finite number > 0")
    if not math.isfinite(min_s) or min_s <= 0.0:
        raise ValueError("min_duration_s must be a finite number > 0")
    if (
        isinstance(run_repeats, bool)
        or not isinstance(run_repeats, int)
        or run_repeats < 1
    ):
        raise ValueError("run_repeats must be an integer >= 1")
    if (
        isinstance(max_repeats, bool)
        or not isinstance(max_repeats, int)
        or max_repeats < 1
    ):
        raise ValueError("max_repeats must be an integer >= 1")

    mean_inference_s = mean_ms / 1000.0
    duration_repeats = math.ceil(min_s / mean_inference_s)
    requested_repeats = max(run_repeats, duration_repeats)

    return min(requested_repeats, max_repeats)
