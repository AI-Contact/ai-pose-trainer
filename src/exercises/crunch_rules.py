from __future__ import annotations

from typing import Dict, Callable
import numpy as np
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_EAR, R_EAR, NOSE, L_ANKLE, R_ANKLE


def lower_back_fixed_score(landmarks: np.ndarray) -> float:
    left_hip_y = landmarks[L_HIP][1]
    right_hip_y = landmarks[R_HIP][1]
    y_diff = abs(left_hip_y - right_hip_y)  # 좌우 엉덩이 y 좌표 차이
    
    # 차이가 작을수록 좋음. 기준: 0.02 이하면 좋음, 0.05 이상이면 나쁨
    # 0.0~0.05 범위를 1~0으로 정규화
    MAX_DIFF = 1.0  # 사용자 조정 가능
    if y_diff >= MAX_DIFF:
        return 0.0
    return float(1.0 - (y_diff / MAX_DIFF))


def shoulder_blade_lift_score(landmarks: np.ndarray) -> float:
    l_shoulder = calculate_angle_with_horizontal_2d(landmarks[L_SH], landmarks[L_HIP])
    r_shoulder = calculate_angle_with_horizontal_2d(landmarks[R_SH], landmarks[R_HIP])

    side = determine_profile_side(landmarks, pose_type="lying")
    shoulder_angle = l_shoulder if side == "left" else r_shoulder

    # 25도에 가까울수록 만점, 0도에 가까울수록 0점
    # 25도에서 1.0, 0도에서 0.0 (선형 증가)
    if shoulder_angle >= 25.0:
        return 1.0
    score = shoulder_angle / 25.0
    return float(max(0.0, min(1.0, score)))


def tension_maintained_score(landmarks: np.ndarray) -> float:
    """이완시 긴장 유지: 귀가 어깨보다 더 아래로 내려간 경우에만 각도만큼 감점."""
    if landmarks is None or len(landmarks) <= max(R_SH, R_EAR):
        return 0.0
    
    side = determine_profile_side(landmarks, pose_type="lying")
    if side == "left":
        shoulder = landmarks[L_SH]
        ear = landmarks[L_EAR]
    else:
        shoulder = landmarks[R_SH]
        ear = landmarks[R_EAR]
    
    # 귀가 어깨보다 아래에 있는지 확인 (ear_y > shoulder_y)
    if ear[1] <= shoulder[1]:
        # 귀가 어깨보다 위에 있으면 만점
        return 1.0
    
    # 귀가 어깨보다 아래에 있으면: 귀-어깨 선과 수평선 사이의 각도만큼 감점
    angle = calculate_angle_with_horizontal_2d(shoulder, ear)
    
    # 각도가 클수록 감점 (0도에서 1.0, 10도에서 0.0)
    MAX_ANGLE = 10.0
    score = 1.0 - (angle / MAX_ANGLE)
    return float(max(0.0, min(1.0, score)))


def rule_functions() -> Dict[str, Callable[[np.ndarray], float]]:
    return {
        "허리 지면 고정": lower_back_fixed_score,
        "견갑골이 지면으로부터 충분히 올라옴": shoulder_blade_lift_score,
        "이완시 긴장 유지": tension_maintained_score,
    }


def rule_model_weights_by_state() -> Dict[str, Dict[str, float]]:
    """crunch 상태별 모델 가중치."""
    return {
        "up": {
            "견갑골이 지면으로부터 충분히 올라옴": 0.5,
        },
        "down": {
            "허리 지면 고정": 0.5,
            "이완시 긴장 유지": 0.5,
        },
        "move": {

        },
    }

