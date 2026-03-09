import torch

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.torch_runner import TorchRunner
from bench.tests.helpers.create_models import create_tiny_torch_model


def test_measure_controller_torch(tmp_path):
    """Validate enriched CPU, memory, and thread exports for a minimal Torch benchmark."""
    model_path = create_tiny_torch_model(tmp_path / "tiny_measure_model.pt")

    cfg = {
        "warmups": 3,
        "repeats": 5,
        "sampler_hz": 50,
        "pre_roll_s": 1.0,
        "post_delay_s": 1.0,
    }

    runner = TorchRunner(model_path=str(model_path), device="cpu")
    runner.load()
    dummy = runner.prepare({"shape": [1, 8], "dtype": "float32"})

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy)
    runner.teardown()

    timing = result["timing_ms"]
    memory = result["memory"]
    cpu = result["cpu_utilization_pct"]
    thread_config = result["thread_config"]

    assert timing["mean"] > 0
    assert timing["p50"] <= timing["p90"] <= timing["p99"]
    assert result["throughput_sps"] > 0

    assert memory["rss_peak_bytes"] >= memory["rss_start_bytes"]
    assert memory["rss_peak_bytes"] > 0
    assert memory["inference_duration_s"] > 0
    assert memory["inference_duration_source"] == "perf_counter"
    assert memory["duration_source"] == "perf_counter"
    assert memory["infer_start_time_s"] == 0.0
    assert memory["infer_end_time_s"] == memory["inference_duration_s"]
    assert "timestamps_raw_s" in memory
    assert memory["memory_recommendation_scope"] == "observed_process_runtime"
    assert "runtime_overhead_estimate_bytes" in memory
    assert "memory_interpretation_note" in memory

    assert "scope" in cpu and cpu["scope"] == "inference_window"
    assert "scope_label" in cpu and cpu["scope_label"] == "cpu_inference_window"
    assert "core_util_mean_cores" in cpu
    assert "core_util_p95_cores" in cpu
    assert "cpu_time" in cpu
    assert cpu["cpu_audit_consistency_note"] in {
        "plausible_agreement",
        "mild_deviation",
        "investigate_mismatch",
        "insufficient_data",
    }
    assert cpu["valid"] in (True, False)

    assert "requested_threads" in thread_config
    assert "backend_thread_control_supported" in thread_config
    assert "thread_env" in thread_config
