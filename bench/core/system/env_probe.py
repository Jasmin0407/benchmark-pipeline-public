"""bench.core.system.env_probe

Environment probe – version & capability scanning.

Design principles:
 - Best-effort: missing optional dependencies must never crash a benchmark run.
 - Transparency: report both *available* and *active* ONNX Runtime providers.
 - Platform safety:
   - On Windows, do not attempt to initialize TensorRT Execution Provider by default
     (frequent DLL load issues on developer machines).
 - Hybrid systems: NVIDIA detection should consider all GPUs, not only GPU[0].
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import subprocess
from typing import Any, Dict, List, Optional

from bench.core.schemas.env_schema import EnvSchema


def _safe_import(module_name: str, attr: Optional[str] = None):
    """
    Import helper that never raises.
    attr can be "submodule.attr" to drill down.
    """
    try:
        module = __import__(module_name)
        if attr:
            for part in attr.split("."):
                module = getattr(module, part)
        return module
    except Exception:
        return None


def _run_cmd(cmd: List[str]) -> Optional[str]:
    """Run a command and return decoded output; else None."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def _has_nvidia_gpu(hardware: Optional[Dict[str, Any]]) -> bool:
    """Robust NVIDIA detection for hybrid systems."""
    if not hardware or not isinstance(hardware.get("gpus"), list):
        return False
    return any("nvidia" in ((g.get("name") or "").lower()) for g in hardware["gpus"])

def _collect_numpy_env() -> Optional[Dict[str, Any]]:
    """Collect NumPy build configuration in a structured way.

    We capture the output of ``numpy.show_config()`` and parse key sections.
    The parsed structure is intended for reporting/debugging only.
    """
    try:
        import numpy as np
        import sys
        from io import StringIO

        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        np.show_config()
        sys.stdout = old_stdout

        raw = buf.getvalue().splitlines()

        blas = {}
        lapack = {}
        simd_found = []
        simd_missing = []
        compiler = {}

        section = None
        for line in raw:
            l = line.strip()

            if l.startswith("blas"):
                section = "blas"
            elif l.startswith("lapack"):
                section = "lapack"
            elif l.startswith("SIMD Extensions"):
                section = "simd"
            elif l.startswith("Compilers"):
                section = "compiler"

            elif section == "blas" and "name:" in l:
                blas["library"] = l.split(":", 1)[1].strip()
            elif section == "lapack" and "name:" in l:
                lapack["library"] = l.split(":", 1)[1].strip()

            elif section == "simd":
                if l.startswith("-"):
                    simd_found.append(l.lstrip("- ").strip())
                elif l.startswith("not found"):
                    simd_missing = []

            elif section == "compiler" and "version:" in l:
                compiler["version"] = l.split(":", 1)[1].strip()

        return {
            "version": np.__version__,
            "blas": blas or None,
            "lapack": lapack or None,
            "simd": {
                "available": simd_found,
                "missing": simd_missing,
            },
            "compiler": compiler or None,
        }

    except Exception:
        return None


def collect_env_info(model_path: Optional[str] = None, hardware: Optional[Dict[str, Any]] = None) -> EnvSchema:
    versions: Dict[str, Any] = {}

    # ------------------------------------------------------------
    # Python / Compiler
    # ------------------------------------------------------------
    versions["python"] = platform.python_version()
    versions["python_compiler"] = platform.python_compiler()

    # Threading-related environment variables are important for reproducibility.
    versions["thread_env"] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }

    # ------------------------------------------------------------
    # Torch
    # ------------------------------------------------------------
    torch = _safe_import("torch")
    if torch:
        versions["torch"] = getattr(torch, "__version__", None)
        versions["torch_cuda"] = torch.version.cuda if hasattr(torch, "version") else None
        try:
            versions["torch_cudnn"] = torch.backends.cudnn.version()
        except Exception:
            versions["torch_cudnn"] = None
    else:
        versions.update({"torch": None, "torch_cuda": None, "torch_cudnn": None})

    # ------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------
    onnx = _safe_import("onnx")
    versions["onnx"] = getattr(onnx, "__version__", None) if onnx else None
    versions["onnx_opset"] = None
    if onnx:
        try:
            versions["onnx_opset"] = onnx.defs.onnx_opset_version()
        except Exception:
            pass

    # ------------------------------------------------------------
    # ONNX Runtime
    # ------------------------------------------------------------
    ort = _safe_import("onnxruntime")
    if ort:
        versions["onnxruntime"] = getattr(ort, "__version__", None)

        try:
            providers_available = ort.get_available_providers()
        except Exception:
            providers_available = []

        # Reporting: keep provider list transparent (no filtering for reporting)
        versions["onnxruntime_providers"] = list(providers_available)

        # Runtime init policy:
        # - Windows: do not attempt TRT EP initialization (DLL load errors)
        providers_for_session = list(providers_available)
        if platform.system().lower() == "windows":
            providers_for_session = [p for p in providers_for_session if p != "TensorrtExecutionProvider"]

        # Optional: if you ever want to suppress CUDA when no NVIDIA exists, do it here.
        _ = _has_nvidia_gpu(hardware)

        # Determine active providers by creating a session with a minimal model.
        try:
            model = _make_minimal_onnx_model()
            sess = ort.InferenceSession(model.SerializeToString(), providers=providers_for_session)
            versions["onnxruntime_providers_active"] = sess.get_providers()
        except Exception:
            versions["onnxruntime_providers_active"] = providers_for_session

        versions["onnxruntime_cuda_available"] = "CUDAExecutionProvider" in (versions["onnxruntime_providers_active"] or [])
    else:
        versions.update(
            {
                "onnxruntime": None,
                "onnxruntime_providers": None,
                "onnxruntime_providers_active": None,
                "onnxruntime_cuda_available": None,
            }
        )

    # ------------------------------------------------------------
    # OpenVINO + Devices/Capabilities
    # ------------------------------------------------------------
    try:
        import openvino.runtime as ov  # type: ignore
        from openvino.runtime import get_version as ov_get_version  # type: ignore

        core = ov.Core()
        versions["openvino"] = ov_get_version()
        versions["openvino_devices"] = core.available_devices

        caps: Dict[str, Any] = {}
        for dev in core.available_devices:
            try:
                caps[dev] = core.get_property(dev, "OPTIMIZATION_CAPABILITIES")
            except Exception:
                caps[dev] = None
        versions["openvino_capabilities"] = caps
    except Exception:
        versions["openvino"] = None
        versions["openvino_devices"] = None
        versions["openvino_capabilities"] = None

    # OpenVINO IR version (relevant for IR/XML models only)
    versions["openvino_ir_version"] = 11
    if model_path and isinstance(model_path, str) and model_path.endswith(".xml"):
        try:
            import openvino.runtime as ov  # type: ignore
            core = ov.Core()
            im = core.read_model(model_path)
            versions["openvino_ir_version"] = im.get_ir_version()
        except Exception:
            pass

    # ------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------
    trt = _safe_import("tensorrt")
    versions["tensorrt"] = getattr(trt, "__version__", None) if trt else None

    # ------------------------------------------------------------
    # CUDA Toolkit / cuDNN
    # ------------------------------------------------------------
    versions["cuda_toolkit"] = _run_cmd(["nvcc", "--version"])
    versions["cudnn"] = versions.get("torch_cudnn") if versions.get("torch_cudnn") else None

    # ------------------------------------------------------------
    # MKL / NumPy / BLAS (structured)
    # ------------------------------------------------------------
    try:
        import mkl  # type: ignore
        versions["mkl"] = mkl.get_version_string()
    except Exception:
        versions["mkl"] = None

    numpy_env = _collect_numpy_env()
    if numpy_env:
        versions["numpy"] = numpy_env.get("version")
        # Keep both a concise summary and a structured record for diagnostics.
        versions["numpy_blas"] = (numpy_env.get("blas") or {}).get("library")
        versions["numpy_lapack"] = (numpy_env.get("lapack") or {}).get("library")
        versions["numpy_simd"] = numpy_env.get("simd")
        versions["numpy_env"] = numpy_env
    else:
        versions["numpy"] = None
        versions["numpy_blas"] = None
        versions["numpy_lapack"] = None
        versions["numpy_simd"] = None
        versions["numpy_env"] = None


    # ------------------------------------------------------------
    # Numba
    # ------------------------------------------------------------
    numba = _safe_import("numba")
    versions["numba"] = getattr(numba, "__version__", None) if numba else None

    return EnvSchema(**versions)


def _make_minimal_onnx_model():
    """
    Create a minimal valid ONNX model for provider activation probing.
    Avoids file I/O and works cross-platform.
    """
    import onnx  # type: ignore
    from onnx import TensorProto, helper  # type: ignore

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([node], "MinimalGraph", [X], [Y])
    model = helper.make_model(graph)
    return model
