# bench/core/runner/tensorrt_runner.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

from .base_runner import BaseRunner


@dataclass
class TRTIO:
    """TensorRT binding metadata (name, binding index, numpy dtype)."""

    name: str
    idx: int
    dtype: Any


class TensorRTRunner(BaseRunner):
    """
    TensorRT runner for the benchmark pipeline.

    Capabilities:
        - Build a TensorRT engine from an ONNX model (EXPLICIT_BATCH) or load a prebuilt .engine.
        - Creates an optimization profile to support dynamic input shapes.
        - prepare(input_spec) creates model inputs (dummy data) and ensures device buffers are allocated.
        - warmup(n, input_spec) runs n warmup iterations.
        - infer(feed) executes the network using execute_async_v2.

    Input specification (compatible with OnnxRunner-style configs):
        input_spec = {
            "shape": [ ... ],              # global shape for all inputs
            "shape_map": {
                "input_name_1": [ ... ],   # per-input shape override
            }
        }

    Notes:
        - This runner uses PyCUDA to manage a CUDA context and stream. The context is created on load()
          and released on teardown(). Ensure teardown() is called to avoid context leaks in long runs.
        - For simplicity, the CUDA device is currently fixed to device 0. If multi-GPU support is needed,
          promote device selection into config and validate availability.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "trt:gpu",
        input_shape: Sequence[int] = (1, 1, 1),
        fp16: bool = False,
        dla_core: Optional[int] = None,
        workspace_bytes: int = 1 << 30,
        trt_log_level: int = trt.Logger.ERROR,
    ) -> None:
        super().__init__(model_path=model_path, device=device)

        self.model_path = str(Path(model_path).expanduser())
        self.input_shape = tuple(int(x) for x in input_shape)

        self.fp16 = bool(fp16)
        self.dla_core = dla_core
        self.workspace_bytes = int(workspace_bytes)

        self.logger = trt.Logger(trt_log_level)
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None

        self.cuda_context: Optional[cuda.Context] = None
        self.stream: Optional[cuda.Stream] = None

        # TensorRT binding pointers (device addresses) in engine binding order
        self.bindings: List[int] = []

        # Binding metadata (inputs/outputs)
        self.inputs: List[TRTIO] = []
        self.outputs: List[TRTIO] = []

        # Host/device buffers keyed by binding name
        self.host_in: Dict[str, np.ndarray] = {}
        self.dev_in: Dict[str, Any] = {}
        self.host_out: Dict[str, np.ndarray] = {}
        self.dev_out: Dict[str, Any] = {}

        # Buffer state
        self._buffers_ready: bool = False
        self._last_shapes: Dict[str, Tuple[int, ...]] = {}

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def load(self) -> None:
        """
        Initialize CUDA, create a CUDA context + stream, and create TensorRT engine/context.

        Supported model inputs:
            - .engine  : deserialize a prebuilt TensorRT engine
            - .onnx    : parse ONNX and build a TensorRT engine

        Raises:
            FileNotFoundError: if model file does not exist.
            RuntimeError: if engine/context creation fails.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        cuda.init()
        dev = cuda.Device(0)
        self.cuda_context = dev.make_context()
        self.stream = cuda.Stream()

        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".engine":
            self._load_engine()
        elif ext == ".onnx":
            self._build_engine_from_onnx()
        else:
            self.teardown()
            raise RuntimeError(f"Unknown model type: {ext}")

        self._analyze_bindings()

        if self.engine is None or self.context is None:
            self.teardown()
            raise RuntimeError("TensorRT engine/context not initialized.")

    def teardown(self) -> None:
        """
        Release resources (buffers and CUDA context).

        Note:
            - Device buffers are released by dropping references; PyCUDA will free memory when objects
              are garbage collected. Clearing dicts makes this deterministic in long-running processes.
            - The CUDA context must be popped to avoid leaking contexts across runs.
        """
        self.bindings = []
        self.host_in.clear()
        self.dev_in.clear()
        self.host_out.clear()
        self.dev_out.clear()
        self._last_shapes.clear()
        self._buffers_ready = False

        self.context = None
        self.engine = None
        self.stream = None

        if self.cuda_context is not None:
            try:
                self.cuda_context.pop()
            except Exception:
                pass
            finally:
                self.cuda_context = None

    # ---------------------------
    # Engine build / load
    # ---------------------------
    def _load_engine(self) -> None:
        """Deserialize a TensorRT engine (.engine) and create an execution context."""
        runtime = trt.Runtime(self.logger)
        with open(self.model_path, "rb") as f:
            blob = f.read()

        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        self.engine = engine

        ctx = self.engine.create_execution_context()
        if ctx is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self.context = ctx

    def _build_engine_from_onnx(self) -> None:
        """
        Parse an ONNX model and build a TensorRT engine.

        Key points:
            - Uses EXPLICIT_BATCH network.
            - Sets workspace memory pool size (compatibility with TRT versions).
            - Enables FP16 if requested and supported.
            - If DLA is selected, uses GPU fallback.
            - Creates an optimization profile:
                - If the ONNX input has dynamic dims, uses self.input_shape for min/opt/max.
                - Otherwise uses the static ONNX shape.
        """
        builder = trt.Builder(self.logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)

        with open(self.model_path, "rb") as f:
            if not parser.parse(f.read()):
                msgs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parse failed:\n" + "\n".join(msgs))

        config = builder.create_builder_config()

        # Workspace / memory pool: TRT API differs across versions
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)
        except Exception:
            config.max_workspace_size = self.workspace_bytes

        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if self.dla_core is not None:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(self.dla_core)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Optimization profile is required if there are dynamic dimensions
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            in_tensor = network.get_input(i)
            name = in_tensor.name
            shp = tuple(int(d) if d is not None else -1 for d in in_tensor.shape)

            if any(d == -1 for d in shp):
                fixed = tuple(self.input_shape)
                profile.set_shape(name, min=fixed, opt=fixed, max=fixed)
            else:
                profile.set_shape(name, min=shp, opt=shp, max=shp)

        config.add_optimization_profile(profile)

        # TRT 8.6+ supports build_serialized_network; older versions may require build_engine
        try:
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Engine build failed (serialized is None).")
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(serialized)
        except Exception:
            engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("TensorRT engine build failed.")
        self.engine = engine

        ctx = self.engine.create_execution_context()
        if ctx is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self.context = ctx

    # ---------------------------
    # I/O + buffers
    # ---------------------------
    def _analyze_bindings(self) -> None:
        """Populate input/output binding metadata from the engine."""
        assert self.engine is not None
        self.inputs = []
        self.outputs = []

        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            trt_dtype = self.engine.get_binding_dtype(idx)
            dtype = self._trt_dtype_to_numpy(trt_dtype)

            if self.engine.binding_is_input(idx):
                self.inputs.append(TRTIO(name=name, idx=idx, dtype=dtype))
            else:
                self.outputs.append(TRTIO(name=name, idx=idx, dtype=dtype))

    @staticmethod
    def _trt_dtype_to_numpy(dt: trt.DataType) -> Any:
        """Map TensorRT DataType to numpy dtype."""
        if dt == trt.DataType.FLOAT:
            return np.float32
        if dt == trt.DataType.HALF:
            return np.float16
        if dt == trt.DataType.INT8:
            return np.int8
        if dt == trt.DataType.INT32:
            return np.int32
        if dt == trt.DataType.BOOL:
            return np.bool_
        raise TypeError(f"Unsupported TensorRT dtype: {dt}")

    def _resolve_shapes_from_spec(self, input_spec: Optional[Dict[str, Any]]) -> Dict[str, Tuple[int, ...]]:
        """
        Resolve input shapes from an optional input_spec.

        Supported fields (OnnxRunner-compatible):
            - input_spec["shape"]: global shape for all inputs
            - input_spec["shape_map"][input_name]: per-input shape override

        Fallback:
            - self.input_shape (runner default)
        """
        shapes: Dict[str, Tuple[int, ...]] = {}

        global_shape = None
        shape_map: Dict[str, Any] = {}

        if input_spec is not None:
            global_shape = input_spec.get("shape")
            shape_map = input_spec.get("shape_map", {}) or {}

        for io in self.inputs:
            if io.name in shape_map:
                shp_raw = shape_map[io.name]
                shp = tuple(int(x) for x in shp_raw)
            elif global_shape is not None:
                shp = tuple(int(x) for x in global_shape)
            else:
                shp = tuple(int(x) for x in self.input_shape)

            shapes[io.name] = shp

        return shapes

    def prepare(self, input_spec: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Create dummy inputs and ensure buffers are allocated for the resolved shapes.

        Buffer allocation is re-done if the shape set changes across calls.
        """
        if self.engine is None or self.context is None or self.stream is None:
            raise RuntimeError("TensorRT: engine/context/stream not initialized. Call load() first.")

        shapes = self._resolve_shapes_from_spec(input_spec)

        feed: Dict[str, np.ndarray] = {}
        for io in self.inputs:
            shp = shapes[io.name]
            feed[io.name] = np.random.randn(*shp).astype(io.dtype)

        # Re-allocate buffers if shapes changed
        if (not self._buffers_ready) or (shapes != self._last_shapes):
            self._free_device_buffers()
            self._ensure_buffers(feed)
            self._last_shapes = dict(shapes)

        return feed

    def _free_device_buffers(self) -> None:
        """
        Drop references to host/device buffers.

        Note:
            - Device memory is freed when PyCUDA objects are garbage collected.
              Clearing references makes memory release more deterministic in long runs.
        """
        self.bindings = []
        self.dev_in.clear()
        self.dev_out.clear()
        self.host_in.clear()
        self.host_out.clear()
        self._buffers_ready = False

    def _ensure_buffers(self, feed: Dict[str, np.ndarray]) -> None:
        """
        Allocate device buffers for current input shapes and set binding shapes in the execution context.
        """
        assert self.engine is not None
        assert self.context is not None
        assert self.stream is not None

        # Set binding shapes for inputs (required for dynamic shapes)
        for io in self.inputs:
            shp = tuple(int(x) for x in feed[io.name].shape)
            self.context.set_binding_shape(io.idx, shp)

        # Binding pointers must have length == num_bindings
        self.bindings = [0] * self.engine.num_bindings

        # Inputs: allocate device memory and store host copies
        for io in self.inputs:
            host = np.ascontiguousarray(feed[io.name]).astype(io.dtype, copy=False)
            self.host_in[io.name] = host
            devmem = cuda.mem_alloc(host.nbytes)
            self.dev_in[io.name] = devmem
            self.bindings[io.idx] = int(devmem)

        # Outputs: allocate based on resolved output binding shapes
        for io in self.outputs:
            out_shape = tuple(int(d) for d in self.context.get_binding_shape(io.idx))
            out_shape = tuple(1 if d == -1 else d for d in out_shape)
            host = np.empty(out_shape, dtype=io.dtype)
            self.host_out[io.name] = host
            devmem = cuda.mem_alloc(host.nbytes)
            self.dev_out[io.name] = devmem
            self.bindings[io.idx] = int(devmem)

        self._buffers_ready = True

    # ---------------------------
    # Warmup / inference
    # ---------------------------
    def warmup(self, n: int = 10, input_spec: Optional[Dict[str, Any]] = None) -> None:
        """Run warmup iterations (prepare() is called internally)."""
        feed = self.prepare(input_spec)
        for _ in range(int(n)):
            _ = self.infer(feed)

    def infer(self, feed: Dict[str, np.ndarray]) -> Any:
        """
        Execute a single inference.

        Steps:
            - Ensure buffers are allocated (in case infer() is called without prepare()).
            - Copy host inputs -> device (HtoD) asynchronously.
            - Execute execute_async_v2 on the current CUDA stream.
            - Copy outputs device -> host (DtoH) asynchronously.
            - Synchronize stream for timing correctness.
        """
        assert self.context is not None
        assert self.stream is not None

        # Safety: allow infer() without explicit prepare()
        if not self._buffers_ready:
            self._ensure_buffers(feed)

        # HtoD copies
        for name, host in self.host_in.items():
            cuda.memcpy_htod_async(self.dev_in[name], host, self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed.")

        # DtoH copies
        for name, host in self.host_out.items():
            cuda.memcpy_dtoh_async(host, self.dev_out[name], self.stream)

        self.stream.synchronize()

        if len(self.host_out) == 1:
            return next(iter(self.host_out.values()))
        return dict(self.host_out)

    # ---------------------------
    # Convenience benchmark (optional)
    # ---------------------------
    def benchmark_ms(self, iters: int, warmup: int) -> Dict[str, float]:
        """
        Small helper to benchmark mean and percentiles (ms) for the current input_shape.

        This method is optional and not required by the main pipeline; it is useful for
        quick local sanity checks during development.
        """
        feed = self.prepare({"shape": list(self.input_shape)})

        for _ in range(int(warmup)):
            _ = self.infer(feed)

        times: List[float] = []
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            _ = self.infer(feed)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        arr = np.array(times, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
