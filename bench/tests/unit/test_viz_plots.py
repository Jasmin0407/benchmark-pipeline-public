import os

from bench.core.analysis.viz.plots import (
    plot_inference_boxplot,
    plot_memory_over_time,
    plot_cpu_utilization,
)
from bench.core.analysis.viz.plots_cpu_single import plot_cpu_util_single


def test_viz_generates_png(tmp_path):
    """
    Verify that visualization helpers generate valid PNG output files.

    This test ensures that:
    - plotting functions execute without raising exceptions,
    - PNG files are written to disk,
    - generated files are non-empty, indicating successful rendering.

    The test intentionally uses a minimal synthetic metrics dictionary to
    avoid dependencies on real benchmark results or hardware.
    """

    # Minimal synthetic benchmark output resembling the real JSON structure.
    fake_json = {
        "meta": {"framework": "torch"},
        "metrics": {
            "inference_time_ms": {"samples": [1.2, 1.3, 1.1]},
            "cpu_utilization_pct": {"mean": 55.0, "p95": 70.0},
            "memory": {
                "rss_samples": [100000000, 100500000, 100800000],
                "timestamps_s": [0.0, 1.0, 2.0],
                "infer_start_idx": 1,
                "infer_end_idx": 2,
                "sample_hz": 1,
            },
        },
    }

    # Output directory for generated plots.
    out_dir = tmp_path / "plots"
    os.makedirs(out_dir, exist_ok=True)

    # Generate inference latency boxplot.
    p1 = plot_inference_boxplot(fake_json, out_dir / "latency_boxplot.png")

    # Generate memory usage over time plot.
    p2 = plot_memory_over_time(
        fake_json["metrics"]["memory"]["rss_samples"],
        timestamps_s=fake_json["metrics"]["memory"]["timestamps_s"],
        infer_start_idx=1,
        infer_end_idx=2,
        sample_hz=1,
        out_dir=str(out_dir),
    )

    # Generate CPU utilization plot (single-run variant).
    p3 = plot_cpu_util_single(
        fake_json["metrics"],
        out_dir / "cpu_utilization.png",
    )

    # Verify that all plots were created and contain data.
    for path in [p1, p2, p3]:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
