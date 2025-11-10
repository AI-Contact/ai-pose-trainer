from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json


def load_condition_names(thresholds_exist: bool, thresholds_keys: List[str], label_json: Path | None) -> List[str]:
    if thresholds_exist and thresholds_keys:
        return list(thresholds_keys)
    if label_json and label_json.exists():
        data = json.loads(label_json.read_text(encoding="utf-8"))
        conditions = data.get("type_info", {}).get("conditions", [])
        return [c.get("condition", f"cond{i}") for i, c in enumerate(conditions)]
    return []


