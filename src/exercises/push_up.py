from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from .base import Exercise
from src.data.rule_based_feature import extract_pushup_features
from src.utils.angles import calculate_angle_2d, calculate_angle_with_horizontal_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import L_SH, R_SH, L_EL, R_EL, L_WR, R_WR


class PushUpExercise(Exercise):
    name = "push_up"

    def __init__(self) -> None:
        self._state = "idle"
        self._reps = 0
        self._down_threshold = 100.0
        self._up_threshold = 150.0
        self._was_down = False  # down 상태를 방문했는지 추적
        self._was_up_before_move = False  # move 전에 up이었는지 추적
        self._feedback_en = None  # 현재 피드백 메시지 (영어)
        self._feedback_ko = None  # 현재 피드백 메시지 (한국어)

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        return extract_pushup_features(pose_seq).astype(np.float32)

    def update_state(self, landmarks: np.ndarray) -> None:

        l_elbow_angle = calculate_angle_2d(landmarks[L_SH], landmarks[L_EL], landmarks[L_WR])
        r_elbow_angle = calculate_angle_2d(landmarks[R_SH], landmarks[R_EL], landmarks[R_WR])

        side = determine_profile_side(landmarks, pose_type="prone")
        angle = l_elbow_angle if side == "left" else r_elbow_angle

        if self._state == "idle":
            if angle < self._down_threshold:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
            elif angle > self._up_threshold:
                self._state = "up"
                self._was_down = False
                self._was_up_before_move = False
            else:
                self._state = "move"
        elif self._state == "down":
            if angle > self._up_threshold:
                self._state = "up"
                if self._was_down:
                    self._reps += 1
                    self._was_down = False
                self._was_up_before_move = False
            elif self._down_threshold <= angle <= self._up_threshold:
                self._state = "move"
        elif self._state == "up":
            if angle < self._down_threshold:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
            elif self._down_threshold <= angle <= self._up_threshold:
                self._state = "move"
                self._was_up_before_move = True
        elif self._state == "move":
            if angle < self._down_threshold:
                self._state = "down"
                self._was_down = True
                self._was_up_before_move = False
            elif angle > self._up_threshold:
                # move -> up: down을 거쳤으면 카운트
                if self._was_down:
                    self._reps += 1
                    self._was_down = False
                elif self._was_up_before_move:
                    # up -> move -> up (down을 거치지 않음): 피드백
                    self._feedback_en = "Lower your chest more"
                    self._feedback_ko = "더 내려가주세요"
                self._state = "up"
                self._was_up_before_move = False

    def get_state(self) -> str:
        return self._state

    def conditions_for_state(self):
        down = ["척추의 중립", "가슴의 충분한 이동", "고개 젖힘/숙임 여부"]
        up = ["척추의 중립", "손의 위치 가슴 중앙 여부", "고개 젖힘/숙임 여부"]
        move = ["척추의 중립", "고개 젖힘/숙임 여부"]
        if self._state == "down":
            return down
        if self._state == "up":
            return up
        if self._state == "move":
            return move
        return []

    def condition_name_map(self):
        return {
            "척추의 중립": "Spine Neutral",
            "가슴의 충분한 이동": "Chest Movement",
            "손의 위치 가슴 중앙 여부": "Hand Position",
            "고개 젖힘/숙임 여부": "Head Tilt",
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
            "척추의 중립": ("Align your spine", "척추 정렬을 맞춰주세요"),
            "손의 위치 가슴 중앙 여부": ("Position hands at chest center", "손을 가슴 중앙에 위치시키세요"),
            "고개 젖힘/숙임 여부": ("Keep head neutral", "고개를 중립 상태로 유지하세요"),
            "가슴의 충분한 이동": ("Lower chest more", "가슴을 더 내려가세요"),
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

    def get_feedback_from_rep_evaluations(
        self, rep_evaluations: List[Tuple[str, float, float]]
    ) -> tuple[Optional[str], Optional[str]]:
        """rep별 평가 결과를 기반으로 피드백 생성. threshold보다 낮은 모든 condition에 대한 피드백 제공."""
        from collections import defaultdict

        feedback_map = {
            "척추의 중립": ("Align your spine", "척추 정렬을 맞춰주세요"),
            "손의 위치 가슴 중앙 여부": ("Position hands at chest center", "손을 가슴 중앙에 위치시키세요"),
            "고개 젖힘/숙임 여부": ("Keep head neutral", "고개를 중립 상태로 유지하세요"),
            "가슴의 충분한 이동": ("Lower chest more", "가슴을 더 내려가세요"),
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


