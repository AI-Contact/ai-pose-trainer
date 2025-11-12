from __future__ import annotations

from typing import Dict, Callable
import numpy as np
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import (
    L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_EAR, R_EAR
)


def lower_back_fixed_score(landmarks: np.ndarray) -> float:
    """
    허리 지면 고정: 골반 기울기 안정성 프록시.
    좌/우 엉덩이 y 좌표 차이가 작을수록 만점.
    """
    left_hip_y = landmarks[L_HIP][1]
    right_hip_y = landmarks[R_HIP][1]
    y_diff = abs(left_hip_y - right_hip_y)

    MAX_DIFF = 20.0  # 픽셀 스케일 환경에 맞게 조정
    if y_diff >= MAX_DIFF:
        return 0.0
    return float(1.0 - (y_diff / MAX_DIFF))


def knee_angle_fixed_score(landmarks: np.ndarray) -> float:
    """
    허벅지와 종아리 각도 고정: 무릎 각도(엉덩이-무릎-발목)가 클수록(펴질수록) 만점.
    120도 이하 0점, 180도 이상 1점으로 선형 매핑.
    """
    l = calculate_angle_2d(landmarks[L_HIP], landmarks[L_KNEE], landmarks[L_ANKLE])
    r = calculate_angle_2d(landmarks[R_HIP], landmarks[R_KNEE], landmarks[R_ANKLE])
    knee = (l + r) / 2.0
    return float(max(0.0, min(1.0, (knee - 120.0) / (180.0 - 120.0))))


def tension_maintained_score(landmarks: np.ndarray) -> float:
    """
    이완 시 다리 긴장 유지: 발목 높이를 힙 기준으로 평가.
    힙 대비 발목 높이 비율이 높을수록 긴장 유지가 잘된 것으로 가정.
    0.05 이하 0점, 0.20 이상 1점 선형 매핑.
    """
    mean_hip_y = (landmarks[L_HIP][1] + landmarks[R_HIP][1]) / 2.0
    mean_ankle_y = (landmarks[L_ANKLE][1] + landmarks[R_ANKLE][1]) / 2.0
    mean_shoulder_y = (landmarks[L_SH][1] + landmarks[R_SH][1]) / 2.0
    upper_len = abs(mean_hip_y - mean_shoulder_y) + 1e-6
    height_ratio = (mean_hip_y - mean_ankle_y) / upper_len  # 클수록 다리 높음

    if height_ratio <= 0.05:
        return 0.0
    if height_ratio >= 0.20:
        return 1.0
    return float((height_ratio - 0.05) / (0.20 - 0.05))


def head_tuck_score(landmarks: np.ndarray) -> float:
    """
    고개 숙임 여부: 귀-어깨-엉덩이 각도(작을수록 턱 당김).
    120도 이상에서 1.0, 110도 이하는 0.0, 그 사이는 선형 감점.
    """
    side = determine_profile_side(landmarks, pose_type="lying")
    if side == "left":
        angle = calculate_angle_2d(landmarks[L_EAR], landmarks[L_SH], landmarks[L_HIP])
    else:
        angle = calculate_angle_2d(landmarks[R_EAR], landmarks[R_SH], landmarks[R_HIP])

    if angle >= 120.0:
        return 1.0
    if angle <= 115.0:
        return 0.0
    # 110~120: 선형 증가 (110 -> 0.0, 120 -> 1.0)
    return float((angle - 115.0) / (120.0 - 115.0))


def rule_functions() -> Dict[str, Callable[[np.ndarray], float]]:
    return {
        "허리 지면 고정": lower_back_fixed_score,
        "허벅지와 종아리 각도 고정": knee_angle_fixed_score,
        "이완 시 다리 긴장유지": tension_maintained_score,
        "고개 숙임 여부": head_tuck_score,
    }


def rule_model_weights_by_state() -> Dict[str, Dict[str, float]]:
    """
    leg_raise 상태별 규칙 가중치.
    """
    return {
        "up": {
            "허벅지와 종아리 각도 고정": 1.0,
            "고개 숙임 여부": 0.5,
        },
        "down": {
            "허리 지면 고정": 1.0,
            "이완 시 다리 긴장유지": 1.0,
            "허벅지와 종아리 각도 고정": 1.0,
            "고개 숙임 여부": 0.5,
        },
        "move": {
            "허리 지면 고정": 1.0,
            "허벅지와 종아리 각도 고정": 1.0,
            "고개 숙임 여부": 0.5,
        },
    }