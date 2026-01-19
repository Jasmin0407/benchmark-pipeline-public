# tests/unit/test_measure_controller.py

import torch

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.torch_runner import TorchRunner
from bench.tests.helpers.create_models import create_tiny_torch_model


def test_measure_controller_torch(tmp_path):
    """
    End-to-end unit test for MeasureController using the TorchRunner backend.

    This test validates that:
    - a minimal Torch model can be executed through the full measurement pipeline,
    - timing, throughput, memory, and CPU utilization metrics are produced,
    - all reported metrics are internally consistent and physically meaningful
      (e.g. positive durations, monotonic percentiles).

    The test intentionally uses a tiny model and CPU execution to ensure
    determinism and fast runtime in CI environments.
    """

    # Create a minimal Torch model on disk to simulate a real inference workload.
    model_path = create_tiny_torch_model(tmp_path / "tiny_measure_model.pt")

    # Measurement configuration:
    # - warmups: iterations excluded from measurement
    # - repeats: number of measured inference runs
    # - sampler_hz: sampling frequency for CPU and memory metrics
    # - pre/post timing windows: capture system behavior around inference
    cfg = {
        "warmups": 3,
        "repeats": 5,
        "sampler_hz": 50,
        "pre_roll_s": 1.0,
        "post_delay_s": 1.0,
    }

    # Initialize the Torch runner in CPU mode and load the model.
    runner = TorchRunner(model_path=str(model_path), device="cpu")
    runner.load()

    # Prepare a dummy input tensor matching the expected input signature.
    dummy = runner.prepare({"shape": [1, 8], "dtype": "float32"})

    # Execute the benchmark via the MeasureController.
    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy)

    # Ensure proper cleanup of runner resources.
    runner.teardown()

    # Extract result sections for readability.
    timing = result["timing_ms"]
    memory = result["memory"]
    cpu = result["cpu_utilization_pct"]

    # --- Timing assertions ---
    # Mean inference time must be positive.
    assert timing["mean"] > 0

    # Percentiles must be monotonic.
    assert timing["p50"] <= timing["p90"] <= timing["p99"]

    # Throughput (samples per second) must be strictly positive.
    assert result["throughput_sps"] > 0

    # --- Memory assertions ---
    # Peak RSS must be greater than or equal to the initial RSS.
    assert memory["rss_peak_bytes"] >= memory["rss_start_bytes"]

    # Memory usage must be non-zero.
    assert memory["rss_peak_bytes"] > 0

    # Inference duration must be strictly positive.
    # This guards against zero-duration measurement bugs.
    assert memory["inference_duration_s"] > 0
