from abc import ABC, abstractmethod
from typing import Dict

class IPredictionService(ABC):
    @abstractmethod
    def status(self) -> Dict:
        pass

    @abstractmethod
    def test(self) -> Dict:
        pass