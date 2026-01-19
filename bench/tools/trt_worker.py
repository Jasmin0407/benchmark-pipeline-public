# bench/tools/trt_worker.py
# Standalone TensorRT worker for Jetson (Python 3.8 / JetPack TensorRT bindings)
# - Builds TRT engine from ONNX (EXPLICIT_BATCH) with Optimization Profile if dynamic
# - Runs warmup + timed iterations
# - Writes results to JSON

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

def _parse_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse a comma-separated shape string (e.g. '1,361,1').

    Validates that all dimensions are positive integers.
    """
    try:
        dims = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip() != "")
    except Exception as e:
        raise ValueError(f"Invalid --shape '{shape_str}': {e}") from e
    if not dims:
        raise ValueError("--shape must contain at least one dimension.")
    if any(d <= 0 for d in dims):
        raise ValueError(f"--shape must contain only positive integers: {dims}")
    return dims


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically to avoid partial files on crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)




class TRTWorker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        self.cuda_ctx = None
        self.shape_override: Optional[Tuple[int, ...]] = None

        self.inputs = []
        self.outputs = []
        self.bindings: List[int] = []

        self.host_in: Dict[str, np.ndarray] = {}
        self.dev_in: Dict[str, Any] = {}
        self.host_out: Dict[str, np.ndarray] = {}
        self.dev_out: Dict[str, Any] = {}

    def load(self):
        cuda.init()
        dev = cuda.Device(0)
        self.cuda_ctx = dev.make_context()
        self.stream = cuda.Stream()

        if self.model_path.endswith(".onnx"):
            self._build_from_onnx()
        elif self.model_path.endswith(".engine"):
            self._load_engine()
        else:
            raise RuntimeError("Unsupported model format")

        self._analyze_bindings()

    def _load_engine(self):
        runtime = trt.Runtime(self.logger)
        with open(self.model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _build_from_onnx(self):
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30

        with open(self.model_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("ONNX parse failed")

        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            t = network.get_input(i)
            name = t.name
            shape = tuple(t.shape)

            if any(d == -1 for d in shape):
                if self.shape_override is None:
                    opt = tuple(1 if d == -1 else int(d) for d in shape)
                else:
                    opt = self.shape_override

                for j, d in enumerate(shape):
                    if d != -1 and int(opt[j]) != int(d):
                        raise RuntimeError(
                            f"Shape mismatch for '{name}': expected {shape}, got {opt}"
                        )

                min_s = list(opt)
                max_s = list(opt)
                min_s[0] = 1
                max_s[0] = opt[0]

                profile.set_shape(name, tuple(min_s), tuple(opt), tuple(max_s))
            else:
                profile.set_shape(name, shape, shape, shape)

        config.add_optimization_profile(profile)

        self.engine = builder.build_engine(network, config)
        if self.engine is None:
            raise RuntimeError("Engine build failed")

        self.context = self.engine.create_execution_context()

    def _analyze_bindings(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if self.engine.binding_is_input(i):
                self.inputs.append((name, i, dtype))
            else:
                self.outputs.append((name, i, dtype))

    def prepare(self):
        feed = {}
        for name, idx, dtype in self.inputs:
            shape = self.context.get_binding_shape(idx)
            shape = tuple(1 if d == -1 else d for d in shape)
            feed[name] = np.random.randn(*shape).astype(dtype)
        return feed

    def infer(self, feed):
        self.bindings = [0] * self.engine.num_bindings

        for name, idx, _ in self.inputs:
            host = feed[name]
            dev = cuda.mem_alloc(host.nbytes)
            cuda.memcpy_htod_async(dev, host, self.stream)
            self.bindings[idx] = int(dev)
            self.dev_in[name] = dev

        for name, idx, dtype in self.outputs:
            shape = self.context.get_binding_shape(idx)
            shape = tuple(1 if d == -1 else d for d in shape)
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings[idx] = int(dev)
            self.host_out[name] = host
            self.dev_out[name] = dev

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        for name in self.host_out:
            cuda.memcpy_dtoh_async(self.host_out[name], self.dev_out[name], self.stream)

        self.stream.synchronize()
        return self.host_out

    def teardown(self):
        if self.cuda_ctx:
            self.cuda_ctx.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shape", required=True, help="e.g. 1,361,1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    worker: Optional[TRTWorker] = None
    worker = TRTWorker(args.model)
    worker.shape_override = shape

    try:
        # --- Setup / Load ---
        worker.load()

        feed = worker.prepare()

        # --- Warmup ---
        for _ in range(args.warmup):
            worker.infer(feed)

        # --- Timed Benchmark ---
        times_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            worker.infer(feed)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms, dtype=np.float64)

        result = {
            "backend": "tensorrt",
            "model": str(Path(args.model).resolve()),
            "shape": list(shape),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "latency_ms": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            },
        }

        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(outp, json.dumps(result, indent=2))

    finally:
        # Deterministic cleanup is critical on Jetson/embedded devices to avoid CUDA context leaks.
        if worker is not None:
            worker.teardown()






if __name__ == "__main__":
    main()


