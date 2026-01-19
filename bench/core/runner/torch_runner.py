# C:\Users\freud\Documents\benchmark-pipeline\bench\core\runner\torch_runner.py
"""
TorchRunner – Loads and executes PyTorch models (CPU or CUDA).

Supported model file formats:
    1) A serialized torch.nn.Module object via torch.save(model, path)
    2) A state_dict (raw or wrapped) via torch.save(state_dict_or_checkpoint, path)

State-dict reconstruction:
    - If a state_dict is detected, this runner attempts to reconstruct a known model
      architecture (currently: DualTransformer) and then loads the weights.

Design goals:
    - Robust loading with clear error messages for typical checkpoint variants.
    - Deterministic inference behavior (eval mode + inference_mode).
    - Correct device placement and optional CUDA synchronization for timing correctness.

Security note:
    - torch.load uses pickle under the hood. Never load untrusted model files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_runner import BaseRunner


class TorchRunner(BaseRunner):
    """
    Runner for PyTorch models (CPU/CUDA).

    Args:
        model_path: Path to a .pt/.pth file created by torch.save(...).
        device: "cpu" or "cuda" (CUDA is used only if available).
        threads: Number of intra-op threads for CPU execution (torch.set_num_threads).
    """

    def __init__(self, model_path: str, device: str = "cpu", threads: int = 1) -> None:
        super().__init__(model_path, device)
        self.model: Optional[torch.nn.Module] = None
        self.threads = int(threads)

        # Determine target device. CUDA is used only if explicitly requested and available.
        self._torch_device = torch.device(
            "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

    # -----------------------------------------------------------
    # Load model
    # -----------------------------------------------------------
    def load(self) -> None:
        """
        Load a PyTorch model from disk.

        Handles:
            - Full model object (torch.nn.Module)
            - state_dict (raw dict of tensors)
            - common checkpoint wrappers: {"state_dict": ...}, {"model": ...}, {"net": ...}

        If a state_dict is found, attempts to reconstruct a known architecture and load weights.
        """
        torch.set_num_threads(self.threads)
        map_loc = "cuda" if self._torch_device.type == "cuda" else "cpu"

        obj = torch.load(self.model_path, map_location=map_loc)

        # -------------------------------------------------------
        # Case 1: Full serialized model object
        # -------------------------------------------------------
        if isinstance(obj, torch.nn.Module):
            self.model = obj.to(self._torch_device)
            self.model.eval()
            print(f"[TorchRunner] Loaded full model object on {self._torch_device}.")
            return

        # -------------------------------------------------------
        # Case 2: state_dict (raw or wrapped checkpoint)
        # -------------------------------------------------------
        if isinstance(obj, dict):
            # a) Pure state_dict? (all values are tensors)
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                state_dict = obj

            # b) Typical wrappers
            elif "state_dict" in obj:
                state_dict = obj["state_dict"]

            elif "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]

            elif "net" in obj and isinstance(obj["net"], dict):
                state_dict = obj["net"]

            else:
                raise RuntimeError(
                    "[TorchRunner] Detected dict checkpoint, but no state_dict was found. "
                    f"Keys: {list(obj.keys())}"
                )

            # ---------------------------------------------------
            # Reconstruct a known model definition and load weights
            # ---------------------------------------------------
            try:
                # Keep the architecture reference explicit and local to avoid import-time coupling.
                from internal.dual_transformer_exportable import DualTransformer


                model_def = DualTransformer(
                    num_classes=4,
                    num_leads=2,
                    dim_model=64,
                    segmented_trans_num_heads=4,
                    segmented_trans_num_encoder_layers=2,
                    global_trans_num_heads=4,
                    global_trans_num_encoder_layers=2,
                    dim_feedforward=64,
                )
                model_def.load_state_dict(state_dict, strict=False)
                model_def.to(self._torch_device)
                model_def.eval()

                self.model = model_def
                print("[TorchRunner] Reconstructed DualTransformer from state_dict.")
                return

            except Exception as e:
                raise RuntimeError(
                    f"[TorchRunner] Failed to reconstruct model from state_dict: {e}"
                ) from e

        # -------------------------------------------------------
        # Case 3: Unknown format
        # -------------------------------------------------------
        raise RuntimeError(f"[TorchRunner] Unknown model format: {type(obj)}")

    # -----------------------------------------------------------
    # Prepare dummy input
    # -----------------------------------------------------------
    def prepare(self, input_spec: Dict[str, Any]) -> Any:
        """
        Create a dummy input tensor.

        input_spec fields:
            - shape: list[int], default [1, 1, 256]
            - dtype: string, default "float32" (also accepts "torch.float32" style)
        """
        shape = input_spec.get("shape", [1, 1, 256])
        dtype_str = input_spec.get("dtype", "float32")

        # Accept "torch.float32" and normalize to "float32"
        if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
            dtype_str = dtype_str.replace("torch.", "")

        try:
            dtype = getattr(torch, dtype_str)
        except AttributeError as exc:
            raise ValueError(f"[TorchRunner] Unsupported dtype '{dtype_str}'.") from exc

        dummy_input = torch.randn(*shape, dtype=dtype).to(self._torch_device)
        print(
            f"[TorchRunner] Dummy input created: shape={shape}, dtype={dtype_str}, device={self._torch_device}"
        )
        return dummy_input

    # -----------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------
    @torch.inference_mode()
    def warmup(self, n: int = 10, input_spec: Optional[Dict[str, Any]] = None) -> None:
        """
        Run warmup iterations.

        CUDA synchronization is performed to ensure timing correctness if used during measurement.
        """
        if input_spec is None:
            input_spec = {"shape": [1, 1, 256], "dtype": "float32"}

        if self.model is None:
            raise RuntimeError("[TorchRunner] Model not loaded. Call load() first.")

        dummy = self.prepare(input_spec)

        for _ in range(int(n)):
            _ = self.model(dummy)
            if self._torch_device.type == "cuda":
                torch.cuda.synchronize()

        print(f"[TorchRunner] Warmup completed ({n} iterations).")

    # -----------------------------------------------------------
    # Inference
    # -----------------------------------------------------------
    @torch.inference_mode()
    def infer(self, dummy_input: Any) -> Any:
        """Run a single inference call."""
        if self.model is None:
            raise RuntimeError("[TorchRunner] Model not loaded. Call load() first.")

        out = self.model(dummy_input)
        if self._torch_device.type == "cuda":
            torch.cuda.synchronize()
        return out

    # -----------------------------------------------------------
    # Release resources
    # -----------------------------------------------------------
    def teardown(self) -> None:
        """Release model references and (optionally) clear CUDA cache."""
        self.model = None
        if self._torch_device.type == "cuda":
            torch.cuda.empty_cache()
