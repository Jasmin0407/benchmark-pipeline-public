# bench/core/orchestrator/device_loop.py
from __future__ import annotations

import datetime
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.onnx_runner import OnnxRunner
from bench.core.runner.openvino_runner import OpenVinoRunner
from bench.core.system.env_probe import collect_env_info
from bench.core.system.hardware_probe import HardwareProbe

try:
    import openvino.runtime as ov  # type: ignore
except Exception:
    ov = None


def _to_plain_dict(obj: Any) -> Any:
    """Best-effort conversion of pydantic/dataclass-like objects to plain dict."""
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


class DeviceLoop:
    def __init__(self, model_path: str, backend: str, devices: List[str], cfg: Dict[str, Any]):
        self.model_path = model_path
        self.backend = str(backend).lower()
        self.devices = list(devices)
        self.cfg = cfg
        self.ov_core = ov.Core() if ov else None

        # One run_id shared across all devices in this multi-run (stable reporting).
        self._run_id = str(uuid.uuid4())

        # Collect shared system snapshots once (best-effort, must not crash).
        self._shared_hardware: Optional[Dict[str, Any]] = None
        self._shared_env: Optional[Dict[str, Any]] = None
        try:
            hw = HardwareProbe.collect()
            self._shared_hardware = _to_plain_dict(hw)
            env = collect_env_info(model_path=self.model_path, hardware=self._shared_hardware.get("detail") if isinstance(self._shared_hardware, dict) else None)  # type: ignore[union-attr]
            self._shared_env = _to_plain_dict(env)
        except Exception:
            self._shared_hardware = None
            self._shared_env = None

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Execute the benchmark for all configured devices.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping: device_key -> per-device result dict (serializable).
        """
        results: Dict[str, Dict[str, Any]] = {}

        for dev in self.devices:
            dev_str = str(dev)
            dev_key = dev_str.lower()

            print("\n=================================")
            print(f"  Starting benchmark for device: {dev_str}")
            print("=================================\n")

            # OpenVINO device availability check (best-effort)
            if self.backend in ("openvino", "ov"):
                if self.ov_core is None:
                    print("[WARN] OpenVINO is not available in this environment -> skipping OpenVINO run.")
                    continue
                if not self._device_available(dev_key):
                    print(f"[WARN] Device '{dev_str}' not available -> skipped.")
                    continue

            # Create runner
            try:
                runner = self._make_runner(dev_key)
            except Exception as exc:
                print(f"[ERROR] Could not create runner for '{dev_str}': {exc}")
                continue

            # Load model
            try:
                runner.load()
            except Exception as exc:
                if str(exc) == "NPU_SKIP_DYNAMIC_MODEL":
                    print(f"[INFO] Device '{dev_str}' skipped (dynamic model not supported on NPU).")
                    continue
                print(f"[WARN] Model load failed on '{dev_str}': {exc}")
                continue

            cfg_dev = self._patch_cfg(dev_key)
            controller = MeasureController(cfg_dev)

            try:
                res = controller.run_benchmark(runner)
                # Keep normalized keys for stable downstream processing
                results[dev_key] = self._enrich_device_result(
                    res=res,
                    device_target=dev_key,
                    cfg_dev=cfg_dev,
                )
            except Exception as exc:
                print(f"[ERROR] Benchmark failed on '{dev_str}': {exc}")
            finally:
                runner.teardown()

        return results

    def _enrich_device_result(self, res: Dict[str, Any], device_target: str, cfg_dev: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure per-device multi-run results contain the same top-level blocks as single-run:
        meta, model, metrics, config, hardware, env.
        """
        out: Dict[str, Any] = deepcopy(res) if isinstance(res, dict) else {}

        # Normalize metrics nesting: some controllers may return metric blocks at top-level.
        metrics = out.get("metrics") if isinstance(out.get("metrics"), dict) else {}
        for k in ("macs", "inference_time_ms", "cpu_utilization_pct", "memory"):
            if k in out and k not in metrics and isinstance(out.get(k), (dict, int, float, list)):
                metrics[k] = out.get(k)
        if metrics:
            out["metrics"] = metrics

        # meta
        meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
        meta.setdefault("timestamp", datetime.datetime.now().isoformat())
        meta.setdefault("run_id", self._run_id)
        meta.setdefault("framework", self.backend if self.backend != "ov" else "openvino")
        meta.setdefault("device_target", device_target)
        out["meta"] = meta

        # model
        model = out.get("model") if isinstance(out.get("model"), dict) else {}
        model.setdefault("path", self.model_path)

        # Best-effort fill from config if runner/controller did not provide.
        cfg_model = (cfg_dev.get("model") or {}) if isinstance(cfg_dev.get("model"), dict) else {}
        cfg_input = (cfg_dev.get("input") or {}) if isinstance(cfg_dev.get("input"), dict) else {}

        if "dtype" not in model and cfg_model.get("dtype") is not None:
            model["dtype"] = cfg_model.get("dtype")
        if "input_shape" not in model and cfg_input.get("shape") is not None:
            model["input_shape"] = cfg_input.get("shape")

        out["model"] = model

        # config (effective per-device config view)
        out.setdefault("config", cfg_dev)

        # hardware/env (shared snapshot for this run)
        if self._shared_hardware is not None:
            out.setdefault("hardware", self._shared_hardware)
        if self._shared_env is not None:
            out.setdefault("env", self._shared_env)

        return out

    def _device_available(self, dev: str) -> bool:
        """
        Check whether an OpenVINO device is available.

        Notes:
        - AUTO and HETERO are schedulers/dispatchers and may not appear in available_devices.
        """
        assert self.ov_core is not None
        available = [d.lower() for d in self.ov_core.available_devices]

        if dev.startswith("auto") or dev.startswith("hetero"):
            return True
        return dev in available

    def _patch_cfg(self, dev: str) -> Dict[str, Any]:
        """Create a per-device config view by patching device.target."""
        cfg2 = deepcopy(self.cfg)
        cfg2.setdefault("device", {})
        cfg2["device"]["target"] = dev
        return cfg2

    def _make_runner(self, dev: str):
        """Factory for backend-specific runner instances."""
        if self.backend in ("openvino", "ov"):
            return OpenVinoRunner(self.model_path, device=dev)
        if self.backend == "onnx":
            return OnnxRunner(self.model_path, device=dev)
        raise RuntimeError(f"Unsupported backend: {self.backend}")
