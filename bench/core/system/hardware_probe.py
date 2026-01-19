# bench/core/system/hardware_probe.py
"""
HardwareProbe
-------------
Purpose:
- Produce a reproducible snapshot of the system hardware (CPU/RAM/Storage/GPU/NPU).
- Provide capability flags for backend routing (CUDA/TensorRT/OpenVINO/ORT providers).
- Handle hybrid setups correctly (e.g., Intel iGPU + NVIDIA dGPU).

Design principles:
- Best-effort: No import/call in this module should abort the benchmark.
- Multiple sources per device class (Torch/NVML/OpenVINO) are allowed.
  Results are deduplicated and merged (missing fields are filled where possible).
- supports_cuda must NOT depend on gpus[0]; we must consider all GPUs.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
from typing import Any, Dict, List, Optional

import psutil

from bench.core.schemas.hardware_capabilities_schema import HardwareCapabilitiesSchema
from bench.core.schemas.hardware_detail_schema import HardwareDetailSchema
from bench.core.schemas.hardware_schema import HardwareSchema
from bench.core.schemas.hardware_table_schema import HardwareTableSchema
from bench.core.system.env_probe import collect_env_info

# Optional imports (best-effort)
try:
    import cpuinfo  # type: ignore
except Exception:
    cpuinfo = None

try:
    import openvino.runtime as ov  # type: ignore
    _ov_core = ov.Core()
except Exception:
    _ov_core = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None


def _norm_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _safe_run(cmd: str) -> Optional[str]:
    """
    Run a shell command and return stdout stripped, else None.

    We suppress stderr to avoid noisy messages on systems where optional tools
    (e.g., trtexec) are not installed. This is expected and should not confuse users.
    """
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


# -------------------------------------------------------------------------
# NVML (pynvml) lazy import + init
# -------------------------------------------------------------------------
_NVML_MODULE = None
_NVML_INITIALIZED = False


def _get_pynvml():
    """
    Lazy import for NVML (pynvml). Per design, we avoid module-level NVML imports
    to prevent side effects (driver probing) at import time.
    """
    global _NVML_MODULE
    if _NVML_MODULE is not None:
        return _NVML_MODULE

    try:
        import pynvml  # type: ignore  # local import by design
        _NVML_MODULE = pynvml
        return _NVML_MODULE
    except Exception:
        _NVML_MODULE = None
        return None


def _ensure_nvml_initialized() -> bool:
    """
    Initialize NVML exactly once per process.
    Returns True if NVML is ready, otherwise False.
    """
    global _NVML_INITIALIZED
    pynvml = _get_pynvml()
    if pynvml is None:
        return False

    if _NVML_INITIALIZED:
        return True

    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
        return True
    except Exception:
        _NVML_INITIALIZED = False
        return False


def _detect_cpu_instruction_sets() -> Dict[str, bool]:
    """
    CPU ISA flags (useful for benchmark context / paper methodology).
    If cpuinfo is unavailable, fall back to conservative defaults.
    """
    defaults = {"avx": False, "avx2": False, "avx512": False, "fma": False, "bf16": False}
    try:
        if not cpuinfo:
            return defaults
        ci = cpuinfo.get_cpu_info()
        feats = ci.get("flags", []) or []
        return {
            "avx": "avx" in feats,
            "avx2": "avx2" in feats,
            "avx512": any(str(f).startswith("avx512") for f in feats),
            "fma": "fma" in feats,
            "bf16": ("bf16" in feats) or ("bfloat16" in feats),
        }
    except Exception:
        return defaults


def _infer_vendor(name: str) -> str:
    """
    Normalize vendor classification based on the GPU name string.

    Notes:
    - NVIDIA devices often omit the literal 'NVIDIA' and only show product lines
      such as 'GeForce', 'RTX', 'Quadro', 'Tesla'.
    """
    n = (name or "").lower()

    if any(k in n for k in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return "nvidia"
    if "intel" in n or "arc" in n:
        return "intel"
    if any(k in n for k in ("amd", "radeon", "vega", "rx")):
        return "amd"
    return "unknown"


def _detect_gpu_capabilities_per_gpu(gpus: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Per-GPU capability projection (routing-relevant).
    Output matches your Pydantic schema.
    """
    if not gpus:
        return []

    out: List[Dict[str, Any]] = []
    for gpu in gpus:
        raw_name = gpu.get("name") or ""
        vendor = _infer_vendor(raw_name)

        out.append(
            {
                "device": gpu.get("device"),
                "name": raw_name,
                "vendor": vendor,
                "fp16": vendor in ("nvidia", "intel", "amd"),
                "int8": vendor in ("nvidia", "intel"),  # conservative
                "tensor_cores": vendor == "nvidia",
                "compute_capability": gpu.get("compute_capability"),
            }
        )
    return out


def _detect_npu_capabilities(npus: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """NPU capabilities (OpenVINO OPTIMIZATION_CAPABILITIES)."""
    if not npus:
        return {}

    raw = npus[0].get("capabilities", []) or []
    return {
        "optimization_capabilities": raw,
        "supports_fp16": any("FP16" in str(c) for c in raw),
        "supports_int8": any("INT8" in str(c) for c in raw),
    }


def _format_table(
    cpu: Dict[str, Any],
    gpus: Optional[List[Dict[str, Any]]],
    npus: Optional[List[Dict[str, Any]]],
    ram_total: Optional[float],
    storage: Optional[Dict[str, Any]],
    os_str: str,
) -> HardwareTableSchema:
    """
    Compact table view. The table shows only the first GPU/NPU for readability.
    Full lists remain available in the detailed view.
    """
    return HardwareTableSchema(
        cpu=cpu.get("name"),
        gpu=gpus[0]["name"] if gpus else None,
        npu=npus[0]["name"] if npus else None,
        ram_gb=ram_total,
        storage_gb=storage.get("total_gb") if storage else None,
        os=os_str,
    )


def _fingerprint(
    cpu: Dict[str, Any],
    gpus: Optional[List[Dict[str, Any]]],
    npus: Optional[List[Dict[str, Any]]],
    ram_gb: Optional[float],
    os_str: str,
) -> str:
    """
    Stable short fingerprint for experiment grouping.
    Conservative: uses only the first GPU/NPU name (table-level identity).
    """
    gpu_name = gpus[0]["name"] if gpus else "none"
    npu_name = npus[0]["name"] if npus else "none"
    base = f"{cpu.get('name','')}|{gpu_name}|{npu_name}|{ram_gb}|{os_str}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


def _dedupe_gpus(gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Torch/NVML/OpenVINO can report the same GPU multiple times.
    Strategy:
    - Key primarily by normalized name
    - Merge: fill missing fields
    - Device priority: cuda:* > nvml:* > openvino:*
    """

    def key(g: Dict[str, Any]) -> str:
        k = _norm_name(g.get("name"))
        if k:
            return k
        return f"__noname__:{_norm_name(str(g.get('device')))}"

    def score(device: str) -> int:
        d = (device or "").lower()
        if d.startswith("cuda:"):
            return 3
        if d.startswith("nvml:"):
            return 2
        if d.startswith("openvino:"):
            return 1
        return 0

    merged: Dict[str, Dict[str, Any]] = {}

    for g in gpus:
        k = key(g)
        if k not in merged:
            merged[k] = dict(g)
            continue

        # Merge: fill only missing fields
        for field, val in g.items():
            if merged[k].get(field) in (None, "", [], {}):
                merged[k][field] = val

        # Device priority
        cur_dev = str(merged[k].get("device") or "")
        new_dev = str(g.get("device") or "")
        if score(new_dev) > score(cur_dev):
            merged[k]["device"] = g.get("device")

    return list(merged.values())


class HardwareProbe:
    @staticmethod
    def collect() -> HardwareSchema:
        system = platform.system().lower()
        host = socket.gethostname()
        os_str = f"{platform.system()} {platform.release()} ({platform.version()})"

        cpu = HardwareProbe._cpu_info()
        ram_total, ram_modules = HardwareProbe._ram_info(system)
        storage, storage_modules = HardwareProbe._storage_info(system)
        gpus = HardwareProbe._gpu_info()
        npus = HardwareProbe._npu_info()

        # ORT provider list (raw, unfiltered: reporting transparency)
        try:
            providers_raw: List[str] = ort.get_available_providers() if ort else []
        except Exception:
            providers_raw = []

        # Per-GPU capabilities and aggregate flags
        gpu_caps = _detect_gpu_capabilities_per_gpu(gpus)

        # NVIDIA detection should not depend on "GPU[0]" only
        has_nvidia = any(c.get("vendor") == "nvidia" for c in gpu_caps)
        try:
            import torch  # type: ignore
            has_nvidia = has_nvidia or bool(torch.cuda.is_available())
        except Exception:
            pass

        # TensorRT tool availability: do not spam console if missing on Windows.
        # We only probe trtexec if we actually see an NVIDIA GPU.
        trtexec_ok = bool(has_nvidia and _safe_run("trtexec --version"))

        capabilities = HardwareCapabilitiesSchema(
            cpu_isa=_detect_cpu_instruction_sets(),
            gpus=gpu_caps,
            npu=_detect_npu_capabilities(npus),
            supports_cuda=bool(has_nvidia),
            supports_tensorrt=bool(trtexec_ok),
            supports_openvino=bool(_ov_core),
            supports_onnx=bool(ort),
            onnx_providers=list(providers_raw),
        )

        table = _format_table(cpu, gpus, npus, ram_total, storage, os_str)

        detail = HardwareDetailSchema(
            hostname=host,
            os=os_str,
            architecture=platform.machine(),
            cpu=cpu,
            ram_total_gb=ram_total,
            ram_modules=ram_modules,
            storage=storage,
            storage_modules=storage_modules,
            gpus=gpus,
            npus=npus,
        )

        fp = _fingerprint(cpu, gpus, npus, ram_total, os_str)

        return HardwareSchema(table=table, detail=detail, capabilities=capabilities, fingerprint=fp)

    @staticmethod
    def _cpu_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            if cpuinfo:
                ci = cpuinfo.get_cpu_info()
                info["name"] = ci.get("brand_raw", platform.processor())
                info["arch"] = ci.get("arch", platform.machine())
            else:
                info["name"] = platform.processor()
                info["arch"] = platform.machine()

            info["cores_physical"] = psutil.cpu_count(logical=False)
            info["cores_logical"] = psutil.cpu_count(logical=True)

            freq = psutil.cpu_freq()
            if freq:
                info["max_freq_mhz"] = float(freq.max)
        except Exception as e:
            info["error"] = str(e)
        return info

    @staticmethod
    def _ram_info(system: str):
        """Total RAM + optional module list (Windows via WMI)."""
        try:
            total = round(psutil.virtual_memory().total / 1024**3, 2)
        except Exception:
            total = None

        modules = None
        if system == "windows":
            try:
                import wmi  # type: ignore
                w = wmi.WMI()
                modules = [
                    {
                        "capacity_gb": round(int(m.Capacity) / 1024**3, 1),
                        "speed_mhz": getattr(m, "Speed", None),
                        "manufacturer": getattr(m, "Manufacturer", None),
                        "slot": getattr(m, "DeviceLocator", None),
                    }
                    for m in w.Win32_PhysicalMemory()
                ]
            except Exception:
                modules = None

        return total, modules

    @staticmethod
    def _storage_info(system: str):
        """Disk usage + optional disk module list (Windows via WMI)."""
        summary = None
        try:
            if system == "windows":
                drive = os.getenv("SystemDrive", "C:") + "\\"
                d = psutil.disk_usage(drive)
            else:
                d = psutil.disk_usage("/")
            summary = {"total_gb": round(d.total / 1024**3, 2), "free_gb": round(d.free / 1024**3, 2)}
        except Exception:
            summary = None

        modules = None
        if system == "windows":
            try:
                import wmi  # type: ignore
                w = wmi.WMI()
                modules = [
                    {
                        "model": getattr(drv, "Model", None),
                        "interface": getattr(drv, "InterfaceType", None),
                        "size_gb": round(int(getattr(drv, "Size", 0)) / 1024**3, 1),
                    }
                    for drv in w.Win32_DiskDrive()
                ]
            except Exception:
                modules = None

        return summary, modules

    @staticmethod
    def _gpu_info() -> Optional[List[Dict[str, Any]]]:
        """
        GPU inventory (unified, deduplicated).

        Source priority:
        1) Torch CUDA: best for NVIDIA (compute capability, CUDA/cuDNN versions, VRAM)
        2) NVML: NVIDIA driver + VRAM (lazy import + one-time init)
        3) OpenVINO: visible GPU devices (often Intel iGPU)
        """
        gpus: List[Dict[str, Any]] = []

        # 1) Torch CUDA devices
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(idx)
                    gpus.append(
                        {
                            "device": f"cuda:{idx}",
                            "name": prop.name,
                            "vram_gb": round(float(prop.total_memory) / 1024**3, 1),
                            "compute_capability": f"{prop.major}.{prop.minor}",
                            "cuda_version": getattr(torch.version, "cuda", None),
                            "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
                        }
                    )
        except Exception:
            pass

        # 2) NVML (lazy)
        if _ensure_nvml_initialized():
            try:
                pynvml = _NVML_MODULE
                count = pynvml.nvmlDeviceGetCount()

                drv = None
                try:
                    drv = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(drv, bytes):
                        drv = drv.decode(errors="ignore")
                except Exception:
                    drv = None

                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode(errors="ignore")

                    total = None
                    try:
                        total = pynvml.nvmlDeviceGetMemoryInfo(h).total
                    except Exception:
                        total = None

                    gpus.append(
                        {
                            "device": f"nvml:{i}",
                            "name": name,
                            "vram_gb": round(float(total) / 1024**3, 1) if total else None,
                            "driver_version": drv,
                        }
                    )
            except Exception:
                pass

        # 3) OpenVINO GPU devices
        if _ov_core:
            try:
                for dev in _ov_core.available_devices:
                    if "GPU" in dev.upper():
                        try:
                            full_name = _ov_core.get_property(dev, "FULL_DEVICE_NAME")
                        except Exception:
                            full_name = dev
                        gpus.append({"device": f"openvino:{dev}", "name": full_name, "vram_gb": None})
            except Exception:
                pass

        if not gpus:
            return None

        return _dedupe_gpus(gpus)

    @staticmethod
    def _npu_info() -> Optional[List[Dict[str, Any]]]:
        """
        NPU inventory (OpenVINO devices).
        This method must exist because collect() calls it.
        """
        if not _ov_core:
            return None

        npus: List[Dict[str, Any]] = []
        try:
            for dev in _ov_core.available_devices:
                if "NPU" not in dev.upper():
                    continue

                try:
                    name = _ov_core.get_property(dev, "FULL_DEVICE_NAME")
                except Exception:
                    name = dev

                try:
                    caps = _ov_core.get_property(dev, "OPTIMIZATION_CAPABILITIES")
                except Exception:
                    caps = None

                npus.append({"device": dev, "name": name, "capabilities": caps})
        except Exception:
            return None

        return npus if npus else None


def collect_system_snapshot(model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Public API used by bench/core/main.py.
    Returns plain dicts (model_dump) for JSON serialization stability.
    """
    hardware_model = HardwareProbe.collect()
    env_model = collect_env_info(model_path=model_path, hardware=hardware_model.detail.model_dump())
    return {"hardware": hardware_model.model_dump(), "env": env_model.model_dump()}
