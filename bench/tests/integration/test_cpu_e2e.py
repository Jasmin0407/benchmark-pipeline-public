"""CPU end-to-end smoke test for the TorchRunner + MeasureController stack.

This test is designed to be portable (CI-friendly). It validates that:
- a tiny Torch model can be loaded on CPU,
- a dummy input can be prepared,
- the benchmark controller produces a serializable metrics structure,
- a RunSchema JSON artifact can be generated.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.torch_runner import TorchRunner
from bench.core.schemas.run_schema import MetaSchema, MetricsSchema, ModelSchema, RunSchema
from bench.tests.helpers.create_models import create_tiny_torch_model


def test_cpu_e2e(tmp_path):
    model_file = create_tiny_torch_model(tmp_path / "tiny_model.pt")

    runner = TorchRunner(model_file.as_posix(), device="cpu")
    runner.load()

    dummy = runner.prepare({"shape": [1, 8]})

    cfg = {
        "run": {"warmups": 2, "repeats": 5},
        "metrics": {"sampler_hz": 10},
        "model": {"backend": "torch", "path": model_file.as_posix()},
    }

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy_input=dummy)

    # The controller may return either a nested structure with a top-level "metrics"
    # or a flattened dict, depending on adapter mode.
    metrics_dict = result.get("metrics", result)

    run = RunSchema(
        meta=MetaSchema(framework="torch", device_target="cpu"),
        model=ModelSchema(path=model_file.as_posix(), input_shape=[1, 8], dtype="float32"),
        metrics=MetricsSchema(**metrics_dict),
        config=cfg,
    )

    json_str = run.model_dump_json(indent=2)
    assert "inference_time_ms" in json_str

    out_file = tmp_path / "run.json"
    out_file.write_text(json_str, encoding="utf-8")

    assert out_file.exists()
    assert out_file.stat().st_size > 10

    runner.teardown()
