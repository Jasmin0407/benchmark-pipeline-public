from bench.core.metrics.energy_backend import EnergyBackend
from bench.core.metrics.energy_backend_factory import AutoEnergyBackend, create_energy_backend
from bench.core.metrics.jetson_energy_backend import JetsonEnergyBackend
from bench.core.metrics.nvml_energy_backend import NvmlEnergyBackend
from bench.core.metrics.rapl_energy_backend import RaplEnergyBackend
from bench.core.metrics.windows_emi_energy_backend import WindowsEmiEnergyBackend


class FakeBackend(EnergyBackend):
    def __init__(self, name, available, events, sample_time_s=None):
        self.name = name
        self._available = available
        self.events = events
        self._sample_time_s = sample_time_s

    def available(self):
        self.events.append(f"available:{self.name}")
        return self._available

    def unavailable_reason(self):
        return f"{self.name} unavailable"

    def read_energy(self):
        self.events.append(f"read:{self.name}")
        return 1.0

    def last_sample_time_s(self):
        return self._sample_time_s


def test_auto_backend_selects_first_available_candidate():
    events = []
    auto = AutoEnergyBackend(
        [
            FakeBackend("first", False, events),
            FakeBackend("second", True, events),
            FakeBackend("third", True, events),
        ]
    )

    assert auto.name == "auto"
    assert auto.available() is True
    assert auto.name == "second"
    assert events == ["available:first", "available:second"]


def test_auto_backend_unavailable_result_explains_candidates():
    events = []
    auto = AutoEnergyBackend(
        [FakeBackend("a", False, events), FakeBackend("b", False, events)]
    )

    assert auto.available() is False
    reason = auto.unavailable_reason()
    assert "a unavailable" in reason
    assert "b unavailable" in reason


def test_factory_explicit_backend_selection_does_not_probe_hardware():
    assert isinstance(
        create_energy_backend(
            {"device": {"target": "cpu"}, "metrics": {"energy": {"backend": "rapl"}}}
        ),
        RaplEnergyBackend,
    )
    assert isinstance(
        create_energy_backend(
            {"device": {"target": "ort:cuda"}, "metrics": {"energy": {"backend": "nvml"}}}
        ),
        NvmlEnergyBackend,
    )
    assert isinstance(
        create_energy_backend(
            {"device": {"target": "dla:0"}, "metrics": {"energy": {"backend": "jetson_ina3221"}}}
        ),
        JetsonEnergyBackend,
    )


def test_factory_auto_linux_gpu_uses_target_aware_fallback_chain(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Linux",
    )
    backend = create_energy_backend(
        {"device": {"target": "ort:cuda"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == ["jetson_ina3221", "nvml"]


def test_factory_auto_linux_cpu_uses_jetson_then_rapl(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Linux",
    )
    backend = create_energy_backend(
        {"device": {"target": "ort:cpu"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == ["jetson_ina3221", "rapl"]


def test_factory_auto_linux_npu_does_not_pretend_rapl_measures_npu_energy(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Linux",
    )
    backend = create_energy_backend(
        {"device": {"target": "ov:npu"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == ["jetson_ina3221"]


def test_factory_explicit_windows_emi_selection_does_not_probe_hardware():
    backend = create_energy_backend(
        {
            "device": {"target": "ort:cpu"},
            "metrics": {
                "energy": {
                    "backend": "windows_emi",
                    "emi_channel": "RAPL_Package0_PKG",
                }
            },
        }
    )
    assert isinstance(backend, WindowsEmiEnergyBackend)


def test_factory_auto_windows_cpu_uses_emi(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Windows",
    )
    backend = create_energy_backend(
        {"device": {"target": "ort:cpu"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == ["windows_emi"]


def test_factory_auto_windows_gpu_uses_nvml_without_jetson(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Windows",
    )
    backend = create_energy_backend(
        {"device": {"target": "ort:cuda"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == ["nvml"]


def test_factory_auto_windows_npu_has_no_false_energy_backend(monkeypatch):
    monkeypatch.setattr(
        "bench.core.metrics.energy_backend_factory.platform.system",
        lambda: "Windows",
    )
    backend = create_energy_backend(
        {"device": {"target": "ov:npu"}, "metrics": {"energy": {"enabled": True}}}
    )
    assert isinstance(backend, AutoEnergyBackend)
    names = [candidate.name for candidate in backend._candidates]
    assert names == []

def test_auto_backend_delegates_native_sample_time():
    events = []
    auto = AutoEnergyBackend([FakeBackend("timed", True, events, sample_time_s=12.5)])

    assert auto.last_sample_time_s() == 12.5
    assert auto.name == "timed"

