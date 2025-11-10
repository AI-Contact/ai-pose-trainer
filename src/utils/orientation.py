from __future__ import annotations

from typing import Literal
import numpy as np
from src.utils.landmarks import NOSE, L_ANKLE, R_ANKLE


Side = Literal["left", "right"]
PoseType = Literal["lying", "prone"]


def determine_profile_side(
    landmarks: np.ndarray,
    pose_type: PoseType,
) -> Side:
    """옆모습이 왼/오 방향인지 판별.

    Args:
        landmarks: Mediapipe 랜드마크 배열
        pose_type: 자세 타입 ("lying": 누운 자세, "prone": 엎드린 자세)

    기준:
    - 엎드린 자세(prone): 코 좌표가 발목 중간 좌표보다 오른쪽에 있으면 "right", 아니면 "left"
    - 누운 자세(lying): 코 좌표가 발목 중간 좌표보다 오른쪽에 있으면 "left", 아니면 "right" (반대)
    """
    if landmarks is None or len(landmarks) < max(R_ANKLE, NOSE) + 1:
        return "left"

    nose_x = float(landmarks[NOSE][0])
    ankle_mid_x = float((landmarks[L_ANKLE][0] + landmarks[R_ANKLE][0]) * 0.5)

    # 엎드린 자세: nose_x > ankle_mid_x -> "right", 아니면 "left"
    # 누운 자세: nose_x > ankle_mid_x -> "left", 아니면 "right" (반대)
    if pose_type == "prone":
        return "right" if nose_x > ankle_mid_x else "left"
    else:  # lying
        return "left" if nose_x > ankle_mid_x else "right"


