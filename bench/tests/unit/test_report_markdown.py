from pathlib import Path

from bench.core.analysis.viz.report_markdown import create_markdown_report


def test_markdown_report_includes_cpu_audit_memory_scope_and_threads(tmp_path):
    run_dir = Path(tmp_path)
    payload = {
        "meta": {"framework": "onnx", "device_target": "cpu", "timestamp": "2026-03-09T10:00:00"},
        "model": {
            "path": "model.onnx",
            "input_shape": [1, 360, 1],
            "dtype": "fp32",
            "size_on_disk_bytes": 1024,
        },
        "metrics": {
            "inference_time_ms": {"mean": 1.0, "p50": 1.0, "p90": 1.2, "p95": 1.3, "p99": 1.5, "batch_size": 1, "mean_per_sample": 1.0},
            "throughput_sps": 1000.0,
            "cpu_utilization_pct": {
                "mean": 250.0,
                "p95": 320.0,
                "core_util_mean_cores": 2.5,
                "core_util_p95_cores": 3.2,
                "cpu_audit_consistency_note": "plausible_agreement",
                "cpu_time": {"wall_s": 1.0, "cpu_s": 2.2, "cpu_core_util": 2.2},
            },
            "memory": {
                "rss_peak_bytes": 500000000,
                "rss_peak_inference_bytes": 420000000,
                "rss_baseline_mean_bytes": 350000000,
                "rss_mean_inference_bytes": 370000000,
                "model_specific_runtime_memory_bytes": 20000000,
                "runtime_overhead_estimate_bytes": 480000000,
                "memory_safety_margin_pct": 15.0,
                "minimal_required_ram_bytes": 483000000,
                "memory_recommendation_scope": "observed_process_runtime",
                "memory_recommendation": "Empirical recommendation.",
                "memory_interpretation_note": "Process RSS includes runtime overhead.",
            },
        },
        "thread_config": {
            "requested_threads": 4,
            "backend_thread_control_supported": True,
            "applied_intra_op_threads": 4,
            "applied_inter_op_threads": 1,
            "execution_mode": "sequential",
            "thread_env": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None, "NUMEXPR_NUM_THREADS": None},
        },
        "config": {"run": {"threads": 4}},
        "hardware": {},
        "env": {},
    }

    create_markdown_report(run_dir, payload)
    md = (run_dir / "bench.md").read_text(encoding="utf-8")

    assert "CPU wall time (audit, s)" in md
    assert "CPU core utilization (time-based audit)" in md
    assert "Memory recommendation scope" in md
    assert "Requested threads" in md
    assert "Runtime overhead estimate" in md
