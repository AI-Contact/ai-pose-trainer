from __future__ import annotations

from typing import Dict, Callable
import numpy as np
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP, L_KNEE, R_KNEE


def spine_neutral_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_SH, R_HIP, R_KNEE):
        return 0.0

    side = determine_profile_side(landmarks, pose_type="prone")
    
    if side == "left":
        shoulder = landmarks[L_SH][:2]  # x, y만 사용
        hip = landmarks[L_HIP][:2]
        knee = landmarks[L_KNEE][:2]
    else:  # right
        shoulder = landmarks[R_SH][:2]
        hip = landmarks[R_HIP][:2]
        knee = landmarks[R_KNEE][:2]
    
    # 어깨-힙-무릎 각도 계산
    spine_angle = calculate_angle_2d(shoulder, hip, knee)

    knee_to_shoulder = shoulder - knee
    knee_to_hip = hip - knee
    
    cross_product = knee_to_shoulder[0] * knee_to_hip[1] - knee_to_shoulder[1] * knee_to_hip[0]

    # 엉덩이가 선보다 아래에 있는 경우: 150~180도 구간에 대해 선형 증가
    if cross_product > 0:  # 엉덩이가 선보다 아래
        print(f"spine_angle: {spine_angle}")
        if spine_angle < 160.0:
            return 0.0
        if spine_angle >= 180.0:
            return 1.0
        # 160도에서 0.0, 180도에서 1.0
        score = (spine_angle - 160.0) / 20.0
        return float(max(0.0, min(1.0, score)))
    
    # 엉덩이가 선보다 위에 있는 경우
    # Map 130~180 deg to 0~1 (higher is better)
    return max(0.0, min(1.0, (spine_angle - 130.0) / 50.0))


def upper_body_height_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_SH, R_EL):
        return 0.0
    
    l_angle = calculate_angle_with_horizontal_2d(landmarks[L_SH], landmarks[L_EL])
    r_angle = calculate_angle_with_horizontal_2d(landmarks[R_SH], landmarks[R_EL])
    
    side = determine_profile_side(landmarks, pose_type="prone")
    angle = l_angle if side == "left" else r_angle
    
    MAX_DEVIATION = 25.0  # 25° 이상 벗어나면 0점

    deviation = abs(90.0 - angle)  # 90도에서 벗어난 정도 (0이면 완전 수직)
    if deviation >= MAX_DEVIATION:
        return 0.0
    
    score = 1.0 - (deviation / MAX_DEVIATION)
    return float(max(0.0, min(1.0, score)))


def rule_functions() -> Dict[str, Callable[[np.ndarray], float]]:
    return {
        "몸통과 엉덩이의 정렬 유지": spine_neutral_score,
        "상체의 지면으로부터 충분한 거리 유지": upper_body_height_score,
    }


def rule_model_weights_by_state() -> Dict[str, Dict[str, float]]:
    """플랭크 상태는 hold 하나만 사용."""
    return {
        "hold": {
            "몸통과 엉덩이의 정렬 유지": 0.5,
            "상체의 지면으로부터 충분한 거리 유지": 0.5,
        }
    }


