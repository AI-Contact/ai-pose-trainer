from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from .base import Exercise
from src.data.rule_based_feature import extract_crunch_features
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import (
    L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, NOSE, L_ANKLE, R_ANKLE
)

class CrunchExercise(Exercise):
    name = "crunch"

    def __init__(self) -> None:
        self._state = "idle"
        self._reps = 0
        self._was_down = False  # down 상태를 방문했는지 추적
        self._was_up_before_move = False  # move 전에 up이었는지 추적
        self._was_down_before_move = False  # move 전에 down이었는지 추적
        self._feedback_en = None  # 현재 피드백 메시지 (영어)
        self._feedback_ko = None  # 현재 피드백 메시지 (한국어)
        self._down_threshold_angle = 3.0  # 어깨가 힙보다 위에 있을 때, 이보다 작으면 down
        self._up_threshold_angle = 20.0  # 이보다 크면 up

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        return extract_crunch_features(pose_seq).astype(np.float32)

    def update_state(self, landmarks: np.ndarray) -> None:
        # 엉덩이-어깨를 이은 선과 수평선 사이의 각도 계산
        l_angle = calculate_angle_with_horizontal_2d(landmarks[L_HIP], landmarks[L_SH])
        r_angle = calculate_angle_with_horizontal_2d(landmarks[R_HIP], landmarks[R_SH])
        
        side = determine_profile_side(landmarks, pose_type="lying")
        
        if side == "left":
            shoulder_y = landmarks[L_SH][1]
            hip_y = landmarks[L_HIP][1]
            current_angle = l_angle
        else:  # right
            shoulder_y = landmarks[R_SH][1]
            hip_y = landmarks[R_HIP][1]
            current_angle = r_angle
        
        # 어깨가 힙보다 아래에 있으면 down (shoulder_y > hip_y)
        # 또는 어깨가 힙보다 위에 있는데 수평선과의 각도가 5도보다 작으면 down
        is_down = (shoulder_y > hip_y) or (shoulder_y <= hip_y and current_angle < self._down_threshold_angle)
        # 어깨가 힙보다 위에 있고 각도가 20도보다 크면 up
        is_up = (shoulder_y <= hip_y and current_angle > self._up_threshold_angle)

        if self._state == "idle":
            if is_down:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif is_up:
                self._state = "up"
                self._was_down = False
                self._was_up_before_move = False
                self._was_down_before_move = False
            else:
                self._state = "move"
        elif self._state == "down":
            if is_up:
                self._state = "up"
                if self._was_down:
                    self._reps += 1
                    self._was_down = False
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif not is_down:
                self._state = "move"
                self._was_down_before_move = True  # down -> move 전환
        elif self._state == "up":
            if is_down:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif not is_up:
                self._state = "move"
                self._was_up_before_move = True  # up -> move 전환
                self._was_down_before_move = False
        elif self._state == "move":
            if is_down:
                # move -> down: up을 거쳤으면 정상, 거치지 않았으면 피드백
                if self._was_down_before_move:
                    # down -> move -> down (up을 거치지 않음): 피드백 및 카운팅 방지
                    self._feedback_en = "Lift your shoulders more"
                    self._feedback_ko = "조금 더 올라와주세요"
                    self._was_down = False  # 카운팅 방지: up을 거치지 않았으므로
                else:
                    # up -> move -> down: 정상적인 경우
                    self._was_down = True
                self._state = "down"
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif is_up:
                # move -> up: down을 거쳤으면 카운트
                if self._was_down:
                    self._reps += 1
                    self._was_down = False
                self._state = "up"
                self._was_up_before_move = False
                self._was_down_before_move = False

    def get_state(self) -> str:
        return self._state

    def conditions_for_state(self):
        up = ["견갑골이 지면으로부터 충분히 올라옴"]
        down = ["허리 지면 고정", "이완시 긴장 유지"]
        if self._state == "down":
            return down
        if self._state == "up":
            return up

        return []

    def condition_name_map(self):
        return {
            "허리 지면 고정": "Lower Back Fixed",
            "견갑골이 지면으로부터 충분히 올라옴": "Shoulder Blade Lift",
            "이완시 긴장 유지": "Tension Maintained",
        }

    def counters(self):
        return {"reps": self._reps}

    def get_feedback(self) -> tuple[Optional[str], Optional[str]]:
        return (self._feedback_en, self._feedback_ko)

    def get_feedback_from_conditions(self, results: List[Tuple[str, float, float, bool]]) -> tuple[Optional[str], Optional[str]]:
        """
        results: [(condition_name, prob, threshold, is_true), ...]
        조건 평가 결과를 기반으로 피드백 생성
        """
        # 피드백 우선순위: 낮은 confidence부터
        feedback_map = {
            "허리 지면 고정": ("Keep lower back fixed", "허리를 지면에 고정하세요"),
            "견갑골이 지면으로부터 충분히 올라옴": ("Lift your shoulders more", "어깨를 더 올려주세요"),
            "이완시 긴장 유지": ("Maintain tension", "긴장을 유지하세요"),
        }
        
        # threshold 미만인 조건 중 가장 낮은 confidence 찾기
        worst_condition = None
        worst_prob = 1.0
        
        for cond_name, prob, threshold, is_true in results:
            if not is_true and prob < worst_prob:
                worst_prob = prob
                worst_condition = cond_name
        
        if worst_condition and worst_condition in feedback_map:
            return feedback_map[worst_condition]
        
        return (None, None)


