from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from .base import Exercise
from src.data.rule_based_feature import extract_leg_raise_features
from src.utils.angles import calculate_angle_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import (
    L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE
)


class LegRaiseExercise(Exercise):
    name = "leg_raise"

    def __init__(self) -> None:
        self._state = "idle"
        self._reps = 0
        self._was_down = False
        self._was_up_before_move = False
        self._was_down_before_move = False
        self._feedback_en = None
        self._feedback_ko = None
        self._down_hip_angle_thresh = 150.0  # 이보다 크면 down
        self._up_hip_angle_thresh = 100.0     # 이보다 작으면 up

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        return extract_leg_raise_features(pose_seq).astype(np.float32)


    def update_state(self, landmarks: np.ndarray) -> None:

        l_hip_angle = calculate_angle_2d(landmarks[L_SH], landmarks[L_HIP], landmarks[L_ANKLE])
        r_hip_angle = calculate_angle_2d(landmarks[R_SH], landmarks[R_HIP], landmarks[R_ANKLE])
        
        side = determine_profile_side(landmarks, pose_type="lying")
        hip_angle = l_hip_angle if side == "left" else r_hip_angle

        # down: 어깨-엉덩이-발목 각도가 큼 (다리 내려감)
        is_down = (hip_angle > self._down_hip_angle_thresh)
        # up: 각도가 작음 (다리 충분히 올라감)
        is_up = (hip_angle < self._up_hip_angle_thresh)

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
                # idle -> up: 직전에 down을 거치지 않았으므로 피드백
                self._feedback_en = "Lower your legs first before lifting up"
                self._feedback_ko = "다리를 더 내려주세요"
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
                self._was_down_before_move = True
        elif self._state == "up":
            if is_down:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif not is_up:
                self._state = "move"
                self._was_up_before_move = True
                self._was_down_before_move = False
        elif self._state == "move":
            if is_down:
                # move -> down: up을 거치지 않고 다시 down이면 피드백
                if self._was_down_before_move:
                    self._feedback_en = "Lift your legs higher"
                    self._feedback_ko = "다리를 더 올려주세요"
                    self._was_down = False
                else:
                    self._was_down = True
                self._state = "down"
                self._was_up_before_move = False
                self._was_down_before_move = False
            elif is_up:
                if self._was_down:
                    self._reps += 1
                    self._was_down = False
                else:
                    # move -> up: 직전에 down을 거치지 않았으므로 피드백
                    self._feedback_en = "Lower your legs first before lifting up"
                    self._feedback_ko = "다리를 더 내려주세요"
                self._state = "up"
                self._was_up_before_move = False
                self._was_down_before_move = False

    def get_state(self) -> str:
        return self._state

    def conditions_for_state(self):
        up = ["허벅지와 종아리 각도 고정", "고개 숙임 여부"]
        move = ["허리지면 고정", "허벅지와 종아리 각도 고정", "고개 숙임 여부"]
        down = ["허리 지면 고정", "이완 시 다리 긴장유지", "허벅지와 종아리 각도 고정", "고개 숙임 여부"]
        if self._state == "down":
            return down
        if self._state == "up":
            return up
        if self._state == "move":
            return move
        return []

    def condition_name_map(self):
        return {
            "허리 지면 고정": "Lower Back Fixed",
            "허벅지와 종아리 각도 고정": "Knee Angle Fixed",
            "이완 시 다리 긴장유지": "Tension Maintained",
            "고개 숙임 여부": "Head Tuck",
        }

    def counters(self):
        return {"reps": self._reps}

    def get_feedback(self) -> tuple[Optional[str], Optional[str]]:
        return (self._feedback_en, self._feedback_ko)

    def get_feedback_from_conditions(self, results: List[Tuple[str, float, float, bool]]) -> tuple[Optional[str], Optional[str]]:
        """
        조건 평가 결과를 기반으로 피드백 생성
        """
        feedback_map = {
            "허리 지면 고정": ("Keep lower back fixed", "허리를 지면에 고정하세요"),
            "허벅지와 종아리 각도 고정": ("Keep your knees straight", "무릎을 펴주세요"),
            "이완 시 다리 긴장유지": ("Maintain tension while lowering", "내릴 때도 긴장을 유지하세요"),
            "고개 숙임 여부": ("Tuck your chin slightly", "턱을 살짝 당겨주세요"),
        }

        worst_condition = None
        worst_prob = 1.0
        for cond_name, prob, threshold, is_true in results:
            if not is_true and prob < worst_prob:
                worst_prob = prob
                worst_condition = cond_name

        if worst_condition and worst_condition in feedback_map:
            return feedback_map[worst_condition]

        return (None, None)