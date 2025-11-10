from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import Exercise
from src.data.rule_based_feature import extract_cross_lunge_features
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.landmarks import L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE


class CrossLungeExercise(Exercise):
    name = "cross_lunge"

    def __init__(self) -> None:
        self._state = "idle"
        self._feedback_en: Optional[str] = None
        self._feedback_ko: Optional[str] = None
        self._active_leg: Optional[str] = None  # "left" or "right"
        self._left_was_down = False
        self._right_was_down = False
        self._left_reps = 0
        self._right_reps = 0
        self._down_threshold = 70.0  # angle <= down: 굽힘 (down)
        self._up_threshold = 130.0    # angle >= up: 펴짐 (up)
        self._switch_margin = 5.0     # active leg 전환 시 여유각

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        return extract_cross_lunge_features(pose_seq).astype(np.float32)

    def update_state(self, landmarks: np.ndarray) -> None:

        # 발목-무릎 y 좌표 차이 계산
        left_ankle_knee_y_diff = abs(landmarks[L_ANKLE][1] - landmarks[L_KNEE][1])
        right_ankle_knee_y_diff = abs(landmarks[R_ANKLE][1] - landmarks[R_KNEE][1])
        
        # 무릎-엉덩이 y 좌표 차이 계산
        left_knee_hip_y_diff = abs(landmarks[L_KNEE][1] - landmarks[L_HIP][1])
        right_knee_hip_y_diff = abs(landmarks[R_KNEE][1] - landmarks[R_HIP][1])

        # 무릎-엉덩이 y 좌표 차이가 발목-무릎 y 좌표 차이보다 크면 up, 아니면 down
        left_is_up = left_knee_hip_y_diff > left_ankle_knee_y_diff
        right_is_up = right_knee_hip_y_diff > right_ankle_knee_y_diff
        
        # 왼쪽 다리 상태 업데이트
        if not left_is_up:
            # 왼쪽 다리가 down
            if not self._left_was_down:
                self._left_was_down = True
        else:
            # 왼쪽 다리가 up
            if self._left_was_down:
                # down -> up 전환: rep 카운팅
                self._left_reps += 1
                self._left_was_down = False
        
        # 오른쪽 다리 상태 업데이트
        if not right_is_up:
            # 오른쪽 다리가 down
            if not self._right_was_down:
                self._right_was_down = True
        else:
            # 오른쪽 다리가 up
            if self._right_was_down:
                # down -> up 전환: rep 카운팅
                self._right_reps += 1
                self._right_was_down = False
        
        # 전체 상태 업데이트 (표시용)
        if not left_is_up:
            self._state = "left_down"
        elif not right_is_up:
            self._state = "right_down"
        else:
            self._state = "up"


    def get_state(self) -> str:
        return self._state


    def conditions_for_state(self) -> List[str]:
        if self._state == "idle":
            return []
        elif self._state in {"left_down", "right_down"}:
            return [
                "몸통 앞발 앞무릎 방향 일치여부",
                "상체 정면 균형잡기",
            ]
        elif self._state == "up":
            return [
                "상체 정면 균형잡기",
            ]
        return []

    def condition_name_map(self) -> dict:
        return {
            "몸통 앞발 앞무릎 방향 일치여부": "Torso Alignment",
            "상체 정면 균형잡기": "Upper Body Balance",
        }

    def counters(self) -> dict:
        return {
            "left_reps": self._left_reps,
            "right_reps": self._right_reps,
        }

    def get_feedback(self) -> Tuple[Optional[str], Optional[str]]:
        return (self._feedback_en, self._feedback_ko)

    def get_feedback_from_conditions(
        self, results: List[Tuple[str, float, float, bool]]
    ) -> Tuple[Optional[str], Optional[str]]:
        feedback_map = {
            "몸통 앞발 앞무릎 방향 일치여부": ("Align torso with front leg", "몸통을 앞발 방향으로 맞춰주세요"),
            "상체 정면 균형잡기": ("Keep your upper body balanced", "상체 균형을 유지하세요"),
        }

        worst_condition: Optional[str] = None
        worst_prob = 1.0

        for cond_name, prob, threshold, is_true in results:
            if not is_true and prob < worst_prob:
                worst_prob = prob
                worst_condition = cond_name

        if worst_condition and worst_condition in feedback_map:
            return feedback_map[worst_condition]

        return (None, None)

    def get_feedback_from_rep_evaluations(
        self, rep_evaluations: List[Tuple[str, float, float]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """rep별 평가 결과를 기반으로 피드백 생성. threshold보다 낮은 모든 condition에 대한 피드백 제공."""
        from collections import defaultdict

        feedback_map = {
            "몸통 앞발 앞무릎 방향 일치여부": ("Align torso with front leg", "몸통을 앞발 방향으로 맞춰주세요"),
            "상체 정면 균형잡기": ("Keep your upper body balanced", "상체 균형을 유지하세요"),
        }

        if not rep_evaluations:
            return (None, None)

        # 조건별 평균 prob 계산
        cond_probs = defaultdict(list)
        cond_thresholds = {}
        for cond_name, prob, threshold in rep_evaluations:
            cond_probs[cond_name].append(prob)
            if cond_name not in cond_thresholds:
                cond_thresholds[cond_name] = threshold

        # threshold보다 낮은 모든 condition 찾기
        failed_conditions = []
        for cond_name, probs in cond_probs.items():
            avg_prob = sum(probs) / len(probs)
            threshold = cond_thresholds[cond_name]
            if avg_prob < threshold:
                failed_conditions.append((cond_name, avg_prob))

        if not failed_conditions:
            return (None, None)

        # 가장 낮은 평균 prob를 가진 조건부터 정렬
        failed_conditions.sort(key=lambda x: x[1])

        # 모든 실패한 조건에 대한 피드백 생성
        feedback_en_list = []
        feedback_ko_list = []
        for cond_name, _ in failed_conditions:
            if cond_name in feedback_map:
                en, ko = feedback_map[cond_name]
                feedback_en_list.append(en)
                feedback_ko_list.append(ko)

        if feedback_en_list:
            feedback_en = " | ".join(feedback_en_list)
            feedback_ko = " | ".join(feedback_ko_list)
            return (feedback_en, feedback_ko)

        return (None, None)


