import pytest
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


def test_energy_disabled_config_does_not_change_measurement_path(monkeypatch):
    """Phase 1 energy config must not introduce inference or hardware side effects."""

    class CountingRunner:
        def __init__(self):
            self.infer_calls = 0
            self.warmup_calls = 0

        def prepare(self, input_spec):
            return torch.zeros(tuple(input_spec["shape"]), dtype=torch.float32)

        def warmup(self, iterations, input_spec):
            self.warmup_calls += 1

        def infer(self, dummy_input):
            self.infer_calls += 1

        def get_thread_audit(self):
            return {}

    class FakeCpuMeter:
        def start(self):
            return None

        def stop(self):
            return None

        def summary(self, **kwargs):
            return {
                "scope": kwargs.get("scope"),
                "scope_label": kwargs.get("scope_label"),
                "cpu_time": {},
                "core_util_mean_cores": None,
            }

    class FakeGpuMeter:
        def available(self):
            return False

    class FakeMemoryMeter:
        def measure(self, fn, **kwargs):
            fn()
            return {
                "rss_start_bytes": 100.0,
                "rss_end_bytes": 100.0,
                "rss_peak_bytes": 100.0,
                "rss_delta_bytes": 0.0,
                "rss_peak_inference_bytes": 100.0,
                "rss_delta_inference_bytes": 0.0,
                "rss_samples": [100.0, 100.0],
                "timestamps_s": [0.0, 0.01],
                "timestamps_raw_s": [1.0, 1.01],
                "infer_start_idx": 0,
                "infer_end_idx": 1,
                "infer_start_time_s": 0.0,
                "infer_end_time_s": 0.01,
                "infer_start_time_raw_s": 1.0,
                "infer_end_time_raw_s": 1.01,
                "inference_duration_s": 0.01,
                "inference_duration_source": "perf_counter",
                "duration_s": 0.01,
                "duration_source": "perf_counter",
                "pre_roll_s": 0.0,
                "post_delay_s": 0.0,
                "sample_hz": 50.0,
                "mode": "static",
            }

    monkeypatch.setattr("bench.core.measure.measure_controller.sleep", lambda _: None)

    base_cfg = {
        "input": {"shape": [1, 8]},
        "run": {"warmups": 0, "repeats": 3},
        "metrics": {
            "sampler_hz": 50,
            "pre_roll_s": 0.0,
            "post_delay_s": 0.0,
            "memory_mode": "static",
        },
    }
    energy_cfg = {
        **base_cfg,
        "metrics": {
            **base_cfg["metrics"],
            "energy": {
                "enabled": False,
                "min_duration_s": 2.0,
                "max_repeats": 10000,
            },
        },
    }

    def run_and_count(cfg):
        runner = CountingRunner()
        controller = MeasureController(cfg)
        controller.cpu_meter = FakeCpuMeter()
        controller.gpu_meter = FakeGpuMeter()
        controller.mem_meter = FakeMemoryMeter()
        controller.run_benchmark(runner)
        return runner.infer_calls, runner.warmup_calls

    legacy_counts = run_and_count(base_cfg)
    disabled_energy_counts = run_and_count(energy_cfg)

    assert disabled_energy_counts == legacy_counts


def test_energy_enabled_runs_after_timing_and_memory_without_changing_timing_samples(monkeypatch):
    """P8-T1/P8-T2/P8-T4: energy is a dedicated final inference pass."""
    events = []

    class CountingRunner:
        def __init__(self):
            self.infer_calls = 0

        def prepare(self, input_spec):
            return torch.zeros(tuple(input_spec["shape"]), dtype=torch.float32)

        def warmup(self, iterations, input_spec):
            events.append("warmup")

        def infer(self, dummy_input):
            self.infer_calls += 1

        def get_thread_audit(self):
            return {}

    class FakeTimingMeter:
        def __init__(self, repeats):
            self.repeats = repeats

        def reset(self):
            return None

        def measure(self, fn):
            events.append("timing")
            for _ in range(self.repeats):
                fn()
            samples = [10.0] * self.repeats
            return {
                "mean": 10.0,
                "p50": 10.0,
                "p90": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "samples": samples,
            }

    class FakeCpuMeter:
        def start(self):
            return None

        def stop(self):
            return None

        def summary(self, **kwargs):
            return {
                "scope": kwargs.get("scope"),
                "scope_label": kwargs.get("scope_label"),
                "cpu_time": {},
                "core_util_mean_cores": None,
            }

    class FakeGpuMeter:
        def available(self):
            return False

    class FakeMemoryMeter:
        def measure(self, fn, **kwargs):
            events.append("memory")
            fn()
            return {
                "rss_start_bytes": 100.0,
                "rss_end_bytes": 100.0,
                "rss_peak_bytes": 100.0,
                "rss_delta_bytes": 0.0,
                "rss_peak_inference_bytes": 100.0,
                "rss_delta_inference_bytes": 0.0,
                "rss_samples": [100.0, 100.0],
                "timestamps_s": [0.0, 0.01],
                "timestamps_raw_s": [1.0, 1.01],
                "infer_start_idx": 0,
                "infer_end_idx": 1,
                "infer_start_time_s": 0.0,
                "infer_end_time_s": 0.01,
                "infer_start_time_raw_s": 1.0,
                "infer_end_time_raw_s": 1.01,
                "inference_duration_s": 0.01,
                "inference_duration_source": "perf_counter",
                "duration_s": 0.01,
                "duration_source": "perf_counter",
                "pre_roll_s": 0.0,
                "post_delay_s": 0.0,
                "sample_hz": 50.0,
                "mode": "static",
            }

    class FakeEnergyMeter:
        def __init__(self):
            self.repeats = None
            self.batch_size = None
            self.signal_duration_s = None

        def measure(self, fn, repeats, batch_size=1, signal_duration_s=None):
            events.append("energy")
            self.repeats = repeats
            self.batch_size = batch_size
            self.signal_duration_s = signal_duration_s
            for _ in range(repeats):
                fn()
            return {
                "available": True,
                "backend": "fake",
                "measurement_scope": "dedicated_inference_window",
                "duration_s": 2.0,
                "inference_count": repeats,
                "total_energy_j": 4.0,
                "energy_per_inference_j": 4.0 / repeats,
                "energy_per_sample_j": 4.0 / repeats,
                "average_power_w": 2.0,
            }

    monkeypatch.setattr("bench.core.measure.measure_controller.sleep", lambda _: None)

    cfg = {
        "input": {"shape": [1, 8], "fs_hz": 4.0},
        "device": {"target": "cpu"},
        "run": {"warmups": 0, "repeats": 3},
        "metrics": {
            "sampler_hz": 50,
            "pre_roll_s": 0.0,
            "post_delay_s": 0.0,
            "memory_mode": "static",
            "energy": {
                "enabled": True,
                "min_duration_s": 2.0,
                "max_repeats": 10000,
            },
        },
    }

    runner = CountingRunner()
    controller = MeasureController(cfg)
    controller.timing_meter = FakeTimingMeter(repeats=3)
    controller.cpu_meter = FakeCpuMeter()
    controller.gpu_meter = FakeGpuMeter()
    controller.mem_meter = FakeMemoryMeter()
    controller.energy_meter = FakeEnergyMeter()

    result = controller.run_benchmark(runner)

    assert events == ["warmup", "timing", "memory", "energy"]
    assert len(result["timing_ms"]["samples"]) == 3
    assert controller.energy_meter.repeats == 200
    assert result["metrics"]["energy"]["inference_count"] == 200
    assert result["metrics"]["energy"]["measurement_scope"] == "dedicated_inference_window"
    # 3 timing calls + 1 existing memory call + 200 dedicated energy calls.
    assert runner.infer_calls == 204


def test_energy_disabled_does_not_construct_or_probe_backend(monkeypatch):
    """P8-T3: disabled energy keeps the pre-energy controller path untouched."""
    calls = []

    def fail_if_called(cfg):
        calls.append(cfg)
        raise AssertionError("Energy backend must not be created when disabled")

    monkeypatch.setattr(
        "bench.core.measure.measure_controller.create_energy_backend",
        fail_if_called,
    )

    controller = MeasureController(
        {
            "run": {"warmups": 0, "repeats": 1},
            "metrics": {"energy": {"enabled": False}},
        }
    )

    assert controller.energy_meter is None
    assert calls == []


def test_phase13_baseline_runs_after_gross_energy_without_extra_inferences(monkeypatch):
    """Phase 13 adds idle windows after gross energy but no extra model calls."""
    events = []

    class CountingRunner:
        def __init__(self):
            self.infer_calls = 0

        def prepare(self, input_spec):
            return torch.zeros(tuple(input_spec["shape"]), dtype=torch.float32)

        def warmup(self, iterations, input_spec):
            events.append("warmup")

        def infer(self, dummy_input):
            self.infer_calls += 1

        def get_thread_audit(self):
            return {}

    class FakeTimingMeter:
        def reset(self):
            return None

        def measure(self, fn):
            events.append("timing")
            for _ in range(3):
                fn()
            return {
                "mean": 10.0,
                "p50": 10.0,
                "p90": 10.0,
                "p95": 10.0,
                "p99": 10.0,
                "samples": [10.0, 10.0, 10.0],
            }

    class FakeCpuMeter:
        def start(self):
            return None

        def stop(self):
            return None

        def summary(self, **kwargs):
            return {
                "scope": kwargs.get("scope"),
                "scope_label": kwargs.get("scope_label"),
                "cpu_time": {},
                "core_util_mean_cores": None,
            }

    class FakeGpuMeter:
        def available(self):
            return False

    class FakeMemoryMeter:
        def measure(self, fn, **kwargs):
            events.append("memory")
            fn()
            return {
                "rss_start_bytes": 100.0,
                "rss_end_bytes": 100.0,
                "rss_peak_bytes": 100.0,
                "rss_delta_bytes": 0.0,
                "rss_peak_inference_bytes": 100.0,
                "rss_delta_inference_bytes": 0.0,
                "rss_samples": [100.0, 100.0],
                "timestamps_s": [0.0, 0.01],
                "timestamps_raw_s": [1.0, 1.01],
                "infer_start_idx": 0,
                "infer_end_idx": 1,
                "infer_start_time_s": 0.0,
                "infer_end_time_s": 0.01,
                "infer_start_time_raw_s": 1.0,
                "infer_end_time_raw_s": 1.01,
                "inference_duration_s": 0.01,
                "inference_duration_source": "perf_counter",
                "duration_s": 0.01,
                "duration_source": "perf_counter",
                "pre_roll_s": 0.0,
                "post_delay_s": 0.0,
                "sample_hz": 50.0,
                "mode": "static",
            }

    class FakeEnergyMeter:
        def __init__(self):
            self.idle_calls = 0

        def measure(self, fn, repeats, batch_size=1, signal_duration_s=None):
            events.append("energy")
            for _ in range(repeats):
                fn()
            return {
                "available": True,
                "backend": "windows_emi",
                "measurement_scope": "dedicated_inference_window",
                "duration_s": 2.0,
                "inference_count": repeats,
                "total_energy_j": 40.0,
                "energy_per_inference_j": 40.0 / repeats,
                "energy_per_sample_j": 40.0 / repeats / batch_size,
                "average_power_w": 20.0,
                "primary_domain": "RAPL_Package0_PKG",
            }

        def measure_idle(self, duration_s):
            self.idle_calls += 1
            events.append("idle_baseline")
            return {
                "available": True,
                "backend": "windows_emi",
                "measurement_scope": "idle_baseline_window",
                "duration_s": duration_s,
                "duration_source": "backend_counter",
                "wall_clock_duration_s": duration_s,
                "total_energy_j": 16.0,
                "average_power_w": 8.0,
                "primary_domain": "RAPL_Package0_PKG",
            }

    monkeypatch.setattr("bench.core.measure.measure_controller.sleep", lambda _: None)

    cfg = {
        "input": {"shape": [1, 8], "fs_hz": 4.0},
        "device": {"target": "cpu"},
        "run": {"warmups": 0, "repeats": 3},
        "metrics": {
            "sampler_hz": 50,
            "pre_roll_s": 0.0,
            "post_delay_s": 0.0,
            "memory_mode": "static",
            "energy": {
                "enabled": True,
                "min_duration_s": 2.0,
                "max_repeats": 10000,
                "baseline_correction": {
                    "enabled": True,
                    "duration_s": 2.0,
                    "repetitions": 3,
                    "method": "post_energy_idle",
                },
            },
        },
    }

    runner = CountingRunner()
    controller = MeasureController(cfg)
    controller.timing_meter = FakeTimingMeter()
    controller.cpu_meter = FakeCpuMeter()
    controller.gpu_meter = FakeGpuMeter()
    controller.mem_meter = FakeMemoryMeter()
    controller.energy_meter = FakeEnergyMeter()

    result = controller.run_benchmark(runner)

    assert events == [
        "warmup",
        "timing",
        "memory",
        "energy",
        "idle_baseline",
        "idle_baseline",
        "idle_baseline",
    ]
    assert runner.infer_calls == 204
    energy = result["metrics"]["energy"]
    assert energy["gross_energy_j"] == pytest.approx(40.0)
    assert energy["baseline_adjusted_energy_j"] == pytest.approx(24.0)
    assert energy["baseline_correction"]["baseline_power_w"]["mean"] == pytest.approx(8.0)
