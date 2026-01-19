"""ONNX Runtime end-to-end smoke test.

Dependency-gated:
- Skips if `onnx`, `onnxruntime`, or `torch` are not installed.

The test exports a tiny ONNX model into a temporary directory and validates that
the OnnxRunner + MeasureController can produce a serializable metrics structure.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.onnx_runner import OnnxRunner
from bench.core.schemas.run_schema import MetaSchema, MetricsSchema, ModelSchema, RunSchema
from bench.tests.helpers.create_models import create_tiny_onnx_model


def test_onnx_e2e(tmp_path):
    onnx_path = create_tiny_onnx_model(tmp_path / "tiny_model.onnx", input_shape=(1, 8))

    runner = OnnxRunner(onnx_path.as_posix(), device="cpu")
    runner.load()

    dummy = runner.prepare({"shape": [1, 8]})

    cfg = {
        "run": {"warmups": 2, "repeats": 5},
        "metrics": {"sampler_hz": 10},
        "model": {"backend": "onnx", "path": onnx_path.as_posix()},
    }

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy_input=dummy)

    metrics_dict = result.get("metrics", result)

    run = RunSchema(
        meta=MetaSchema(framework="onnx", device_target="cpu"),
        model=ModelSchema(path=onnx_path.as_posix(), input_shape=[1, 8], dtype="float32"),
        metrics=MetricsSchema(**metrics_dict),
        config=cfg,
    )

    json_str = run.model_dump_json(indent=2)
    assert "inference_time_ms" in json_str

    out_file = tmp_path / "run_onnx.json"
    out_file.write_text(json_str, encoding="utf-8")

    assert out_file.exists()
    assert out_file.stat().st_size > 10

    runner.teardown()
