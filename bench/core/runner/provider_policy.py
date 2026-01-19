# bench/core/runner/provider_policy.py
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ProviderType = Union[str, Tuple[str, Dict[str, Any]]]


@dataclass(frozen=True)
class ProviderDecision:
    requested_device: str
    platform_system: str
    provider_chain: List[ProviderType]
    fail_fast: bool
    notes: str = ""


def _sys() -> str:
    return platform.system().lower()


def _normalize_device(device: str) -> str:
    return (device or "").strip().lower()


def resolve_onnx_providers(
    device: str,
    *,
    platform_system: Optional[str] = None,
    allow_fallback: bool = False,
    yaml_override: Optional[Sequence[ProviderType]] = None,
) -> ProviderDecision:
    """Resolve the ONNX Runtime provider chain for a requested device.

    The policy is intentionally conservative to avoid common "works on my machine"
    issues:
      - Windows: explicit CUDA+CPU for GPU requests; do not default to TensorRT/OpenVINO
        Execution Providers due to frequent DLL/runtime compatibility problems.
      - Linux/Jetson: TensorRT may be primary, CUDA secondary, CPU as last fallback.
      - Fallback occurs only when ``allow_fallback=True``.
      - A YAML override can be supplied, but it is validated against platform hard rules.
    """
    sysname = (platform_system or _sys()).lower()
    dev = _normalize_device(device)

    # YAML override (optional)
    if yaml_override:
        chain = list(yaml_override)
        _validate_override(sysname, dev, chain)
        return ProviderDecision(
            requested_device=dev,
            platform_system=sysname,
            provider_chain=chain,
            fail_fast=not allow_fallback,
            notes="YAML override",
        )

    # Windows policy
    if sysname == "windows":
        if dev in ("ort:cuda", "ort_cuda", "onnx_cuda", "cuda"):
            # CUDA + CPU only to avoid TRT/OV EP pitfalls on Windows.
            return ProviderDecision(
                requested_device=dev,
                platform_system=sysname,
                provider_chain=["CUDAExecutionProvider", "CPUExecutionProvider"],
                fail_fast=True,  # GPU requests should be fail-fast
                notes="Windows CUDA policy (no TensorRT/OpenVINO).",
            )
        if dev in ("ort:cpu", "ort_cpu", "onnx_cpu", "cpu"):
            return ProviderDecision(
                requested_device=dev,
                platform_system=sysname,
                provider_chain=["CPUExecutionProvider"],
                fail_fast=not allow_fallback,
                notes="Windows ORT-CPU policy.",
            )

        # OpenVINO via ORT is intentionally not a default on Windows.
        return ProviderDecision(
            requested_device=dev,
            platform_system=sysname,
            provider_chain=["CPUExecutionProvider"],
            fail_fast=not allow_fallback,
            notes="Windows unknown device -> CPU.",
        )

    # Linux / Jetson policy
    if sysname in ("linux", "darwin"):  # Jetson runs Linux
        # TensorRT can be used on Linux; on Jetson it is typically primary.
        if dev in ("ort:cuda", "ort_cuda", "onnx_cuda", "cuda"):
            return ProviderDecision(
                requested_device=dev,
                platform_system=sysname,
                provider_chain=["CUDAExecutionProvider", "CPUExecutionProvider"],
                fail_fast=True,
                notes="Linux CUDA policy.",
            )

        if dev in ("ort:trt", "ort_trt", "tensorrt", "trt"):
            # ORT-TRT only works when TRT libraries are present; allowing CUDA as backup.
            return ProviderDecision(
                requested_device=dev,
                platform_system=sysname,
                provider_chain=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                fail_fast=not allow_fallback,
                notes="Linux TensorRT policy (TRT->CUDA->CPU).",
            )

        if dev in ("ort:cpu", "ort_cpu", "onnx_cpu", "cpu"):
            return ProviderDecision(
                requested_device=dev,
                platform_system=sysname,
                provider_chain=["CPUExecutionProvider"],
                fail_fast=not allow_fallback,
                notes="Linux ORT-CPU policy.",
            )

        return ProviderDecision(
            requested_device=dev,
            platform_system=sysname,
            provider_chain=["CPUExecutionProvider"],
            fail_fast=not allow_fallback,
            notes="Linux unknown device -> CPU.",
        )

    # Fallback for uncommon platforms
    return ProviderDecision(
        requested_device=dev,
        platform_system=sysname,
        provider_chain=["CPUExecutionProvider"],
        fail_fast=not allow_fallback,
        notes="Unknown platform -> CPU.",
    )


def _validate_override(sysname: str, dev: str, chain: List[ProviderType]) -> None:
    """Validate YAML overrides against hard platform rules."""

    # On Windows we explicitly forbid TRT/OV EP for CUDA requests because it commonly
    # triggers DLL-load and runtime issues.
    if sysname == "windows" and dev in ("ort:cuda", "ort_cuda", "onnx_cuda", "cuda"):
        names = [_provider_name(p) for p in chain]
        if "TensorrtExecutionProvider" in names:
            raise ValueError("Windows override invalid: TensorRT provider is not allowed for ort:cuda.")
        if "OpenVINOExecutionProvider" in names:
            raise ValueError("Windows override invalid: OpenVINO provider is not allowed for ort:cuda.")


def _provider_name(p: ProviderType) -> str:
    if isinstance(p, tuple):
        return p[0]
    return p
