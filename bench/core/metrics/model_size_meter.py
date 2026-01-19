"""
ModelSizeMeter

Estimates model size characteristics:
- Total parameter count
- Parameter count by dtype
- Estimated weight storage in bytes (weights_bytes)
- On-disk file size (total_on_disk_bytes)

Supports:
- Torch: reads tensors from state_dict (or module.state_dict())
- ONNX: reads initializers and estimates bytes per element

Notes
-----
- For public repositories, avoid printing absolute local paths in warnings/errors.
  This implementation logs only the file name when a file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np


class ModelSizeMeter:
    def __init__(self, framework: str = "onnx"):
        self.framework = str(framework).lower()

    def analyze(self, model_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "parameters_total": None,
            "by_dtype": {},
            "weights_bytes": None,
            "total_on_disk_bytes": None,
        }

        p = Path(model_path)

        # 1) On-disk size
        if p.exists():
            try:
                result["total_on_disk_bytes"] = int(p.stat().st_size)
            except Exception:
                result["total_on_disk_bytes"] = None
        else:
            # Avoid leaking full local paths in public logs
            print(f"[WARN] Model file not found: {p.name}")
            return result

        # 2) Framework-specific analysis
        if self.framework == "torch":
            return self._analyze_torch(model_path, result)

        if self.framework == "onnx":
            return self._analyze_onnx(model_path, result)

        print(f"[WARN] Unsupported framework for ModelSizeMeter: {self.framework}")
        return result

    def _analyze_torch(self, model_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch  # optional dependency
        except Exception as e:
            print(f"[WARN] PyTorch not available; skipping torch model size analysis: {e}")
            return result

        try:
            obj = torch.load(model_path, map_location="cpu")

            if hasattr(obj, "state_dict"):
                state_dict = obj.state_dict()
            elif isinstance(obj, dict):
                state_dict = obj
            else:
                print("[WARN] Torch file is neither nn.Module nor a state_dict-like dict; skipping.")
                return result

            total_params = 0
            dtype_counter: Dict[str, int] = {}
            total_bytes = 0

            for _, tensor in state_dict.items():
                if not hasattr(tensor, "numel"):
                    continue
                n = int(tensor.numel())
                if n <= 0:
                    continue

                total_params += n
                dtype_str = str(getattr(tensor, "dtype", "unknown")).replace("torch.", "")
                dtype_counter[dtype_str] = dtype_counter.get(dtype_str, 0) + n

                try:
                    elem_size = int(tensor.element_size())
                except Exception:
                    elem_size = 4
                total_bytes += n * elem_size

            result["parameters_total"] = int(total_params)
            result["by_dtype"] = dtype_counter
            result["weights_bytes"] = int(total_bytes)

            return result

        except Exception as e:
            print(f"[WARN] Torch model analysis failed: {e}")
            return result

    def _analyze_onnx(self, model_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import onnx
        except Exception as e:
            print(f"[WARN] ONNX not available; skipping ONNX model size analysis: {e}")
            return result

        # Conservative dtype -> bytes mapping
        bytes_per_elem = {
            "float": 4,
            "float16": 2,
            "double": 8,
            "bfloat16": 2,
            "int8": 1,
            "uint8": 1,
            "int16": 2,
            "uint16": 2,
            "int32": 4,
            "uint32": 4,
            "int64": 8,
            "uint64": 8,
            "bool": 1,
        }

        try:
            model = onnx.load(model_path)

            total_params = 0
            dtype_counter: Dict[str, int] = {}
            total_bytes = 0

            for tensor in model.graph.initializer:
                n = int(np.prod(tensor.dims)) if tensor.dims else 0
                if n <= 0:
                    continue

                total_params += n

                # ONNX returns names like "FLOAT", "FLOAT16", "INT64"
                dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type).lower()

                dtype_counter[dtype_name] = dtype_counter.get(dtype_name, 0) + n
                total_bytes += n * int(bytes_per_elem.get(dtype_name, 4))

            result["parameters_total"] = int(total_params)
            result["by_dtype"] = dtype_counter
            result["weights_bytes"] = int(total_bytes)

            return result

        except Exception as e:
            print(f"[WARN] ONNX model analysis failed: {e}")
            return result
