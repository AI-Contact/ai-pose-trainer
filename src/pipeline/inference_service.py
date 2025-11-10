from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from src.model.models import ExerciseModel


class InferenceService:
    def __init__(self, ckpt_path: str, device: Optional[str] = "cpu") -> None:
        self.model = ExerciseModel.load_from_checkpoint(ckpt_path, map_location=device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs


