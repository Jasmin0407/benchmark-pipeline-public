"""Abstract base class for all backend runners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseRunner(ABC):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def prepare(self, input_spec: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def warmup(self, n: int = 10, input_spec: Optional[Dict[str, Any]] = None) -> None:
        """Run `n` warm-up iterations (optionally using an input specification)."""
        pass

    @abstractmethod
    def infer(self, dummy_input: Any) -> Any:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass


