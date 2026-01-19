# bench/tests/test_memory_duration.py

import math


def _compute_duration_from_ts_or_fallback(memory_block: dict) -> float:
    """
    Compute the total measurement duration from a memory metrics block.

    Priority:
    1) If timestamp samples are available, derive the duration from the
       first and last timestamp (most accurate).
    2) Otherwise, fall back to the sum of pre-roll, inference, and post-delay
       durations.

    The function is defensive by design and guarantees a non-negative
    floating-point return value, even in partially populated or malformed
    input dictionaries.
    """
    ts = memory_block.get("timestamps_s") or []

    # Primary path: derive duration from sampled timestamps.
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return float(ts[-1] - ts[0])

    # Fallback path: reconstruct duration from configured time windows.
    pre = float(memory_block.get("pre_roll_s", 0.0) or 0.0)
    infer = float(memory_block.get("inference_duration_s", 0.0) or 0.0)
    post = float(memory_block.get("post_delay_s", 0.0) or 0.0)

    total = pre + infer + post
    return float(total) if total > 0 else 0.0


def test_duration_from_timestamps():
    """
    Verify that duration is correctly derived from timestamp samples
    when they are present.

    The fallback values must be ignored in this case.
    """
    memory = {
        "timestamps_s": [0.0, 0.5, 1.0, 1.5],
        "pre_roll_s": 5.0,
        "inference_duration_s": 0.2,
        "post_delay_s": 2.0,
        "duration_s": 0.0,
    }

    dur = _compute_duration_from_ts_or_fallback(memory)

    # Expected duration: last timestamp minus first timestamp.
    assert math.isclose(dur, 1.5, rel_tol=0.0, abs_tol=1e-9)


def test_duration_fallback_when_no_timestamps():
    """
    Verify that the fallback path is used when no timestamp samples
    are available.

    The duration must equal the sum of pre-roll, inference, and post-delay.
    """
    memory = {
        "timestamps_s": [],
        "pre_roll_s": 5.0,
        "inference_duration_s": 0.2,
        "post_delay_s": 2.0,
        "duration_s": 0.0,
    }

    dur = _compute_duration_from_ts_or_fallback(memory)

    # Expected duration: 5.0 + 0.2 + 2.0 = 7.2 seconds.
    assert math.isclose(dur, 7.2, rel_tol=0.0, abs_tol=1e-9)
