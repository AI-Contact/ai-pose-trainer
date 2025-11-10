from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_thresholds(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    thresholds: Dict[str, float] = {}
    for _, row in df.iterrows():
        thresholds[row["condition"]] = row["recommended_threshold"]
    return thresholds


def match_thresholds(probs, condition_names: List[str], thresholds: Dict[str, float], filter_names=None):
    results = []
    for i, cond in enumerate(condition_names):
        if i >= len(probs):
            break
        if filter_names is None or cond in filter_names:
            prob = float(probs[i])
            th = float(thresholds.get(cond, 0.5))
            is_true = prob >= th
            results.append((cond, prob, th, is_true))
    return results


