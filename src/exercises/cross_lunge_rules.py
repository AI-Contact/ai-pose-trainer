from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.landmarks import L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_SH, R_SH


def _select_front_leg(landmarks: np.ndarray) -> tuple[int, int, int]:
    # 발목-무릎 y 좌표 차이 계산
    left_ankle_knee_y_diff = abs(landmarks[L_ANKLE][1] - landmarks[L_KNEE][1])
    right_ankle_knee_y_diff = abs(landmarks[R_ANKLE][1] - landmarks[R_KNEE][1])
    
    # 무릎-엉덩이 y 좌표 차이 계산
    left_knee_hip_y_diff = abs(landmarks[L_KNEE][1] - landmarks[L_HIP][1])
    right_knee_hip_y_diff = abs(landmarks[R_KNEE][1] - landmarks[R_HIP][1])
    
    left_is_down = left_knee_hip_y_diff <= left_ankle_knee_y_diff
    right_is_down = right_knee_hip_y_diff <= right_ankle_knee_y_diff
    
    # 둘 다 down이거나 둘 다 up인 경우, 더 down인 쪽을 선택 (무릎-엉덩이 y 좌표 차이가 더 작은 쪽)
    if left_is_down and not right_is_down:
        return L_HIP, L_KNEE, L_ANKLE
    elif right_is_down and not left_is_down:
        return R_HIP, R_KNEE, R_ANKLE
    else:
        # 둘 다 down이거나 둘 다 up인 경우, 무릎-엉덩이 y 좌표 차이가 더 작은 쪽을 앞다리로
        if left_knee_hip_y_diff <= right_knee_hip_y_diff:
            return L_HIP, L_KNEE, L_ANKLE
        return R_HIP, R_KNEE, R_ANKLE


def torso_alignment_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(L_HIP, R_HIP, L_SH, R_SH, L_ANKLE, R_ANKLE):
        return 0.0
 
    # 엉덩이 중심과 어깨 중심 계산
    hip_center = (landmarks[L_HIP][:2] + landmarks[R_HIP][:2]) / 2.0
    shoulder_center = (landmarks[L_SH][:2] + landmarks[R_SH][:2]) / 2.0
    
    # 앞다리 발목 인덱스 가져오기
    hip_idx, knee_idx, ankle_idx = _select_front_leg(landmarks)
    front_ankle = landmarks[ankle_idx][:2]
    
    # 앞다리 발목 - 엉덩이 중심 - 어깨 중심 각도 계산 (일직선이면 180도)
    angle = calculate_angle_2d(front_ankle, hip_center, shoulder_center)
    
    # 170도 이상은 만점, 130도에서 0점, 그 사이는 선형 감점
    MIN_ANGLE = 170.0
    MAX_ANGLE = 130.0
    if angle >= MIN_ANGLE:
        return 1.0
    if angle <= MAX_ANGLE:
        return 0.0
    
    # 130도~170도 사이에서 선형 감점 (170도에서 1.0, 130도에서 0.0)
    score = (angle - MAX_ANGLE) / (MIN_ANGLE - MAX_ANGLE)
    return float(max(0.0, min(1.0, score)))


def upper_body_balance_score(landmarks: np.ndarray) -> float:
    if landmarks is None or len(landmarks) <= max(R_SH, R_HIP):
        return 0.0

    shoulder_angle = calculate_angle_with_horizontal_2d(landmarks[L_SH], landmarks[R_SH])
    hip_angle = calculate_angle_with_horizontal_2d(landmarks[L_HIP], landmarks[R_HIP])

    # shoulder 각도 점수 계산: 3도 이하는 만점, 10도에 가까워질수록 0점
    if shoulder_angle <= 3.0:
        shoulder_score = 1.0
    elif shoulder_angle >= 10.0:
        shoulder_score = 0.0
    else:
        # 3도~10도 사이에서 선형 감소 (3도에서 1.0, 10도에서 0.0)
        shoulder_score = 1.0 - ((shoulder_angle - 3.0) / (10.0 - 3.0))
    
    # hip 각도 점수 계산: 3도 이하는 만점, 10도에 가까워질수록 0점
    if hip_angle <= 3.0:
        hip_score = 1.0
    elif hip_angle >= 10.0:
        hip_score = 0.0
    else:
        # 3도~10도 사이에서 선형 감소 (3도에서 1.0, 10도에서 0.0)
        hip_score = 1.0 - ((hip_angle - 3.0) / (10.0 - 3.0))
    
    # 더 낮은 점수 리턴
    return float(min(shoulder_score, hip_score))


def rule_functions() -> Dict[str, Callable[[np.ndarray], float]]:
    return {
        "몸통 앞발 앞무릎 방향 일치여부": torso_alignment_score,
        "상체 정면 균형잡기": upper_body_balance_score,
    }


def rule_model_weights_by_state() -> Dict[str, Dict[str, float]]:
    return {
        "left_down": {
            "몸통 앞발 앞무릎 방향 일치여부": 0.5,
            "상체 정면 균형잡기": 0.5,
        },
        "right_down": {
            "몸통 앞발 앞무릎 방향 일치여부": 0.5,
            "상체 정면 균형잡기": 0.5,
        },
        "up": {
            "상체 정면 균형잡기": 0.5,
        }
    }


