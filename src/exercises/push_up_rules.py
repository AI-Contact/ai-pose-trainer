from __future__ import annotations

from typing import Dict, Callable
import numpy as np
from src.utils.angles import calculate_angle_3d, calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import (
    L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
    L_HIP, R_HIP, L_KNEE, R_KNEE,
    L_EAR, R_EAR, L_MOUTH, R_MOUTH
)


def spine_neutral_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_SH, R_HIP, R_KNEE):
        return 0.0

    l_spine = calculate_angle_2d(landmarks[L_SH], landmarks[L_HIP], landmarks[L_KNEE])
    r_spine = calculate_angle_2d(landmarks[R_SH], landmarks[R_HIP], landmarks[R_KNEE])

    side = determine_profile_side(landmarks, pose_type="prone")
    spine_angle = l_spine if side == "left" else r_spine

    # Map 130~180 deg to 0~1 (higher is better)
    return max(0.0, min(1.0, (spine_angle - 130.0) / 50.0))


def head_neutral_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_EAR, R_MOUTH):
        return 0.0

    l_head = calculate_angle_with_horizontal_2d(landmarks[L_EAR], landmarks[L_MOUTH])
    r_head = calculate_angle_with_horizontal_2d(landmarks[R_EAR], landmarks[R_MOUTH])

    side = determine_profile_side(landmarks, pose_type="prone")
    head_angle = l_head if side == "left" else r_head

    # 75도일 때 1, 75도에서 멀어질수록 0에 가까워짐
    PEAK_ANGLE = 75.0
    MAX_DEVIATION = 15.0  # 75도 ± 15도 범위
    
    deviation = abs(head_angle - PEAK_ANGLE)
    if deviation >= MAX_DEVIATION:
        return 0.0
    
    score = 1.0 - (deviation / MAX_DEVIATION)
    return float(max(0.0, min(1.0, score)))


def hand_position_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_SH, R_WR):
        return 0.0

    side = determine_profile_side(landmarks, pose_type="prone")
    
    if side == "left":
        shoulder = landmarks[L_SH]
        wrist = landmarks[L_WR]
        # 손이 어깨보다 좌측에 있으면 감점
        if wrist[0] < shoulder[0]:  # 손이 어깨보다 왼쪽
            angle = calculate_angle_with_horizontal_2d(shoulder, wrist)
            # 45도에서 0점, 90도에서 1.0 (45도와 90도 사이에서 선형 증가)
            if angle <= 45.0:
                return 0.0
            if angle >= 90.0:
                return 1.0
            score = (angle - 45.0) / (90.0 - 45.0)
            return float(max(0.0, min(1.0, score)))
    else:  # right
        shoulder = landmarks[R_SH]
        wrist = landmarks[R_WR]
        # 손이 어깨보다 우측에 있으면 감점
        if wrist[0] > shoulder[0]:  # 손이 어깨보다 오른쪽
            angle = calculate_angle_with_horizontal_2d(shoulder, wrist)
            # 45도에서 0점, 90도에서 1.0 (45도와 90도 사이에서 선형 증가)
            if angle <= 45.0:
                return 0.0
            if angle >= 90.0:
                return 1.0
            score = (angle - 45.0) / (90.0 - 45.0)
            return float(max(0.0, min(1.0, score)))
    
    # 손이 어깨보다 안쪽에 있으면 만점
    return 1.0


def chest_movement_score(landmarks: np.ndarray) -> float:
    """어깨가 팔꿈치보다 충분히 아래(화면 기준) 위치하는지 평가."""
    if landmarks is None or len(landmarks) <= max(R_SH, R_EL):
        return 0.0

    side = determine_profile_side(landmarks, pose_type="prone")
    if side == "left":
        shoulder = landmarks[L_SH]
        elbow = landmarks[L_EL]
    else:
        shoulder = landmarks[R_SH]
        elbow = landmarks[R_EL]

    diff = shoulder[1] - elbow[1]  # 양수면 어깨가 더 아래(화면 기준)
    if diff >= 0.0:
        return 1.0

    # 음수면: 어깨-팔꿈치를 이은 선과 수평선 사이의 각도가 커질수록 감점
    angle = calculate_angle_with_horizontal_2d(shoulder, elbow)

    # 0도(수평)에서 1.0, 45도에서 0.0
    MAX_ANGLE = 45.0
    score = 1.0 - (angle / MAX_ANGLE)
    return float(max(0.0, min(1.0, score)))


def rule_functions() -> Dict[str, Callable[[np.ndarray], float]]:
    return {
        "척추의 중립": spine_neutral_score,
        "고개 젖힘/숙임 여부": head_neutral_score,
        "손의 위치 가슴 중앙 여부": hand_position_score,
        "가슴의 충분한 이동": chest_movement_score,
    }


def rule_model_weights_by_state() -> Dict[str, Dict[str, float]]:
    """Return per-state model weight for each condition name.
    The blended prob = w_model * model_prob + (1-w_model) * rule_score
    """
    return {
        "move": {
            "척추의 중립": 0.3,
            "고개 젖힘/숙임 여부": 0.3,
        },
        "up": {
            "척추의 중립": 0.5,
            "손의 위치 가슴 중앙 여부": 0.5,
            "고개 젖힘/숙임 여부": 0.5,
        },
        "down": {
            "척추의 중립": 0.5,
            "고개 젖힘/숙임 여부": 0.5,
            "가슴의 충분한 이동": 0.5,
        },
    }


