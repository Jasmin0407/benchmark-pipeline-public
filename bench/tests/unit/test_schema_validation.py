# tests/unit/test_schema_validation.py

import json
import tempfile

from bench.core.schemas.run_schema import RunSchema


def test_schema_validation_minimal():
    """
    Validate that a minimal, but structurally complete, JSON document
    conforms to the RunSchema.

    This test ensures that:
    - all mandatory top-level sections are present,
    - the schema accepts a minimal configuration without optional metrics,
    - deserialization from JSON into a typed RunSchema model succeeds.

    The goal is to guard against accidental schema regressions that would
    break backward compatibility for existing benchmark result files.
    """
    data = {
        "meta": {
            "run_id": "dummy",
            "framework": "torch",
            "framework_version": "0.0",
            "device_target": "cpu",
            "timestamp": "2025-01-01T00:00:00",
        },
        "model": {
            "path": "inline",
            "input_shape": [1, 1, 360],
            "dtype": "float32",
            "parameters": 0,
            "dtype_breakdown": {"fp32": 0},
            "size_on_disk_bytes": 0,
        },
        "hardware_table": {
            "Systemname": "Test",
            "CPU (Modell + Takt)": "Dummy CPU",
            "GPU (Modell)": "—",
            "NPU (Modell)": "—",
            "RAM (GB)": 16,
            "Speichertyp": "—",
            "OS / Version": "DummyOS",
        },
        "hardware": {
            "table": {
                "cpu": "Dummy CPU",
                "gpu": "—",
                "npu": "—",
                "ram_gb": 16,
                "storage_gb": 128,
                "os": "DummyOS",
            },
            "detail": {},
            "capabilities": {},
            "fingerprint": "dummy",
        },
        "metrics": {
            "macs": {
                "total": 0,
                "per_sample": 0,
                "parameters_total": 0,
            },
            "inference_time_ms": {
                "mean": 1.0,
                "p50": 1.0,
                "p90": 1.0,
                "p95": 1.0,
                "p99": 1.1,
                "batch_size": 32,
                "mean_per_sample": 1.0 / 32.0,
            },
            "throughput_sps": 1000,
            "cpu_utilization_pct": {
                "mean": 10.0,
                "p95": 20.0,
            },
            "memory": {
                "weights_bytes": 0,
                "activations_bytes": 0,
                "total_theoretical_bytes": 0,
                "peak_ram_process_bytes": 0,
                "peak_gpu_bytes": 0,

                "rss_baseline_mean_bytes": 680_000_000,
                "rss_mean_inference_bytes": 720_000_000,
                "model_specific_runtime_memory_bytes": 40_000_000,
                "rss_peak_inference_bytes": 760_000_000,
                "memory_safety_margin_pct": 15.0,
                "minimal_required_ram_bytes": 874_000_000,
                "memory_recommendation": "Recommended minimum RAM: 0.81 GiB (peak_inference_rss + 15.0% safety margin).",
            },

        },
        "env": {},
        "config": {"backend": "torch_cpu"},
    }

    # Serialize to JSON and validate against the RunSchema.
    json_str = json.dumps(data)
    run = RunSchema.model_validate_json(json_str)

    # Sanity check on a representative field.
    assert run.meta.framework == "torch"
