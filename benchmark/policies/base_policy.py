"""Shared policy runner interface for benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class PolicyResult:
    action: np.ndarray
    inference_time_ms: float


class PolicyRunner(Protocol):
    name: str

    def train(self, train_episodes: list) -> None:
        ...

    def predict(self, observation: dict) -> PolicyResult:
        ...


class NotConnectedPolicy:
    """Placeholder runner used until a real method is connected."""

    name = "not_connected"

    def train(self, train_episodes: list) -> None:
        return None

    def predict(self, observation: dict) -> PolicyResult:
        raise NotImplementedError(f"{self.name} runner is not connected yet.")
