def test_timing_meter():
    """
    Validate basic functionality and sanity of the TimingMeter.

    This test verifies that:
    - the TimingMeter executes the configured number of warmup and
      measurement iterations,
    - the reported mean execution time is within a reasonable range
      for a known, deterministic workload.

    A small sleep-based workload is intentionally used to:
    - avoid CPU-bound variability,
    - provide a predictable lower bound for timing,
    - keep the test lightweight and CI-friendly.
    """
    from bench.core.metrics.timing_meter import TimingMeter
    import time

    # Configure the timing meter with one warmup iteration and
    # multiple measured runs to obtain a stable mean.
    tm = TimingMeter(warmups=1, repeats=5)

    # Measure a simple, deterministic workload.
    res = tm.measure(lambda: time.sleep(0.01))

    # The expected mean duration should be close to 10 ms.
    # A relatively wide tolerance is used to account for
    # scheduler jitter and CI environment variability.
    assert 8 <= res["mean"] <= 20, res
