import platform

from bench.core.runner.provider_policy import resolve_onnx_providers


def test_windows_cuda_policy():
    """
    Verify ONNX Runtime provider resolution on Windows for CUDA devices.

    Expected behavior:
    - 'ort:cuda' must resolve to CUDAExecutionProvider as the primary provider.
    - TensorRT must not be included on Windows, as it is not supported
      or intentionally disabled in this environment.
    - OpenVINO must not be part of the provider chain in this configuration.
    """
    decision = resolve_onnx_providers(
        device="ort:cuda",
        platform_system="Windows",
        allow_fallback=False,
    )

    names = [
        p if isinstance(p, str) else p[0]
        for p in decision.provider_chain
    ]

    assert names[0] == "CUDAExecutionProvider"
    assert "TensorrtExecutionProvider" not in names
    assert "OpenVINOExecutionProvider" not in names


def test_linux_trt_policy():
    """
    Verify ONNX Runtime provider resolution on Linux (e.g. Jetson platforms).

    Expected behavior:
    - 'ort:trt' is allowed to prioritize TensorRTExecutionProvider.
    - CUDAExecutionProvider may be included as a fallback when enabled.
    """
    decision = resolve_onnx_providers(
        device="ort:trt",
        platform_system="Linux",
        allow_fallback=True,
    )

    names = [
        p if isinstance(p, str) else p[0]
        for p in decision.provider_chain
    ]

    assert names[0] == "TensorrtExecutionProvider"
    assert "CUDAExecutionProvider" in names
