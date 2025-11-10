from __future__ import annotations

from typing import List, Optional
import numpy as np


class SequenceBuffer:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self._buf: List[np.ndarray] = []

    def push(self, landmarks: np.ndarray) -> None:
        self._buf.append(landmarks)
        if len(self._buf) > self.max_len:
            self._buf.pop(0)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def as_array(self) -> np.ndarray:
        return np.array(self._buf)


