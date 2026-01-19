# bench/core/metrics/macs_meter.py
"""
MacsMeter

Estimates Multiply-Accumulate operations (MACs) and parameter counts.

Backends
--------
- torch:
    Uses ptflops if available. Requires a torch.nn.Module instance.
    State-dict-only files are intentionally not reconstructed here (public repo safety & generality).

- onnx:
    Best-effort static graph estimation for common ops (Conv, MatMul, Gemm).
    Uses ONNX shape inference to retrieve tensor shapes.

Notes
-----
MAC accounting conventions vary across tools/papers (MACs vs FLOPs).
This meter reports MACs (multiply-accumulate pairs) under a consistent definition per op.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Any

# Torch + ptflops (optional)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None

# ONNX (required for ONNX MAC estimation)
try:
    import onnx
    import onnx.numpy_helper as numpy_helper
    from onnx import shape_inference
except Exception:
    onnx = None
    numpy_helper = None
    shape_inference = None


class MacsMeter:
    def __init__(self, framework: str = "torch"):
        self.framework = str(framework).lower()

    def analyze(self, model_path: str, input_shape: List[int]) -> Optional[Dict[str, float]]:
        """
        Analyze model MACs.

        Parameters
        ----------
        model_path:
            Path to the model file (.pt/.pth for torch, .onnx for onnx).
        input_shape:
            Full input tensor shape including batch dimension.

        Returns
        -------
        Dict[str, float] or None
        """
        try:
            if self.framework == "torch":
                return self._analyze_torch(model_path, input_shape)
            if self.framework == "onnx":
                return self._analyze_onnx(model_path, input_shape)

            print(f"[WARN] Unsupported MACs framework: {self.framework}")
            return None
        except Exception as e:
            print(f"[WARN] MACs analysis failed: {e}")
            return None

    def _analyze_torch(self, model_path: str, input_shape: List[int]) -> Dict[str, float]:
        """
        Torch MAC estimation via ptflops.

        Requires:
        - torch installed
        - ptflops installed
        - model file loads into a torch.nn.Module
        """
        if torch is None or nn is None:
            print("[WARN] PyTorch not installed; skipping torch MACs.")
            return {}
        if get_model_complexity_info is None:
            print("[WARN] ptflops not installed; skipping torch MACs.")
            return {}

        device = "cpu"

        model = torch.load(model_path, map_location=device)

        if not isinstance(model, nn.Module):
            # Public-repo safe default: do not reconstruct unknown models from state_dict.
            print(
                "[WARN] Torch model file did not contain a torch.nn.Module instance. "
                "State-dict-only files are not reconstructed by default. "
                "Export to ONNX for MAC estimation or provide a model factory hook."
            )
            return {}

        model.eval()

        # ptflops expects input shape without batch dimension
        macs, params = get_model_complexity_info(
            model,
            tuple(input_shape[1:]),
            as_strings=False,
            print_per_layer_stat=False,
        )

        batch = float(input_shape[0]) if input_shape and input_shape[0] else 1.0

        return {
            "macs_total": float(macs),
            "macs_per_sample": float(macs / batch) if batch > 0 else float(macs),
            "parameters_total": float(params),
        }

    def _analyze_onnx(self, model_path: str, input_shape: List[int]) -> Dict[str, float]:
        """
        ONNX MAC estimation for common ops.

        Best-effort approach:
        - Run ONNX shape inference to access intermediate tensor shapes.
        - Estimate MACs for Conv / MatMul / Gemm where shapes allow.
        """
        if onnx is None or numpy_helper is None or shape_inference is None:
            print("[WARN] ONNX dependencies not installed; skipping ONNX MACs.")
            return {}

        model = onnx.load(model_path)
        model = shape_inference.infer_shapes(model)

        # Map initializers (weights) by name
        initializer_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

        # Helper: get tensor shape from value_info / inputs / outputs
        def get_shape(name: str) -> Optional[List[int]]:
            candidates = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
            for vi in candidates:
                if vi.name == name:
                    dims = []
                    for d in vi.type.tensor_type.shape.dim:
                        # dim_value can be 0 if unknown; treat as None-like
                        dims.append(int(d.dim_value) if d.dim_value else 0)
                    return dims
            return None

        def get_attr_int(node: Any, key: str, default: int) -> int:
            for a in node.attribute:
                if a.name == key:
                    # ONNX stores ints in .i
                    try:
                        return int(a.i)
                    except Exception:
                        return default
            return default

        macs_total = 0.0

        for node in model.graph.node:
            op = str(node.op_type).lower()

            # -------------------------
            # Conv: MACs = N*Hout*Wout*Cout*(Cin/group)*Kh*Kw
            # Weight shape: [Cout, Cin/group, Kh, Kw] (2D conv)
            # -------------------------
            if op == "conv":
                if len(node.input) < 2:
                    continue

                W = initializer_map.get(node.input[1])
                if W is None:
                    continue

                out_shape = get_shape(node.output[0])
                in_shape = get_shape(node.input[0])

                if not out_shape or not in_shape:
                    continue

                # Expect NCHW in many ONNX convs; if NHWC appears, estimation will be off.
                # We use output shape directly for N*Hout*Wout*Cout.
                if len(out_shape) < 4:
                    continue

                N = float(out_shape[0] or 1)
                Cout = float(out_shape[1] or 0)
                Hout = float(out_shape[2] or 0)
                Wout = float(out_shape[3] or 0)

                if Cout == 0 or Hout == 0 or Wout == 0:
                    continue

                groups = get_attr_int(node, "group", 1)
                groups = max(1, int(groups))

                # W: [Cout, Cin_per_group, Kh, Kw]
                if W.ndim < 4:
                    continue
                Cin_per_group = float(W.shape[1])
                Kh = float(W.shape[2])
                Kw = float(W.shape[3])

                macs = N * Hout * Wout * Cout * Cin_per_group * Kh * Kw
                macs_total += macs
                continue

            # -------------------------
            # MatMul: MACs = M*N*K for (M,K) x (K,N)
            # -------------------------
            if op == "matmul":
                if len(node.input) < 2:
                    continue
                A = get_shape(node.input[0])
                B = get_shape(node.input[1])
                if not A or not B:
                    continue
                if len(A) != 2 or len(B) != 2:
                    continue
                M, K = float(A[0] or 0), float(A[1] or 0)
                K2, N = float(B[0] or 0), float(B[1] or 0)
                if M == 0 or K == 0 or N == 0 or K2 == 0:
                    continue
                if K != K2:
                    continue
                macs_total += M * N * K
                continue

            # -------------------------
            # Gemm: like MatMul with optional bias; MACs ~ M*N*K
            # Output shape can help; use input shapes when available.
            # -------------------------
            if op == "gemm":
                if len(node.input) < 2:
                    continue
                A = get_shape(node.input[0])
                B = get_shape(node.input[1])
                if not A or not B:
                    continue
                if len(A) != 2 or len(B) != 2:
                    continue
                M, K = float(A[0] or 0), float(A[1] or 0)
                K2, N = float(B[0] or 0), float(B[1] or 0)
                if M == 0 or K == 0 or N == 0 or K2 == 0:
                    continue
                if K != K2:
                    continue
                macs_total += M * N * K
                continue

        batch = float(input_shape[0]) if input_shape and input_shape[0] else 1.0

        return {
            "macs_total": float(macs_total),
            "macs_per_sample": float(macs_total / batch) if batch > 0 else float(macs_total),
        }
