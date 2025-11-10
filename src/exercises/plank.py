from __future__ import annotations

import time
from typing import List, Tuple, Optional
import numpy as np
from .base import Exercise
from src.data.rule_based_feature import extract_plank_features
from src.utils.angles import calculate_angle_2d
from src.utils.orientation import determine_profile_side
from src.utils.landmarks import L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE


class PlankExercise(Exercise):
    name = "plank"

    def __init__(self) -> None:
        self._reps = 0
        self._start_time: float | None = None
        self._last_rep_time: float | None = None
        self._rep_interval = 3.0  # 3초당 1회
        self._state = "idle"  # "idle" 또는 "hold"

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        return extract_plank_features(pose_seq).astype(np.float32)

    def update_state(self, landmarks: np.ndarray) -> None:
        
        l_angle = calculate_angle_2d(landmarks[L_SH], landmarks[L_HIP], landmarks[L_KNEE])
        r_angle = calculate_angle_2d(landmarks[R_SH], landmarks[R_HIP], landmarks[R_KNEE])
        
        side = determine_profile_side(landmarks, pose_type="prone")
        angle = l_angle if side == "left" else r_angle
        
        # 어깨-힙-무릎 각도에 따라 상태 변경
        if angle <= 100.0:
            self._state = "idle"
            return
        else:
            self._state = "hold"
        
        # 플랭크는 시간 기반으로 3초당 1회 카운팅
        current_time = time.time()
        
        if self._start_time is None:
            # 첫 프레임: 시작 시간 기록
            self._start_time = current_time
            self._last_rep_time = current_time
        
        # 마지막 rep 이후 경과 시간 확인
        elapsed_since_last_rep = current_time - self._last_rep_time
        
        if elapsed_since_last_rep >= self._rep_interval:
            # 3초 경과: rep 카운팅
            self._reps += 1
            self._last_rep_time = current_time

    def get_state(self) -> str:
        return self._state

    def counters(self) -> dict:
        return {"reps": self._reps}

    def condition_name_map(self):
        return {
            "팔꿈치가 어깨보다 안쪽에 위치하지 않음": "Elbow Position",
            "몸통과 엉덩이의 정렬 유지": "Spine-Hip Alignment",
            "상체의 지면으로부터 충분한 거리 유지": "Upper Body Height",
        }

    def get_feedback_from_rep_evaluations(
        self, rep_evaluations: List[Tuple[str, float, float]]
    ) -> tuple[Optional[str], Optional[str]]:
        """rep별 평가 결과를 기반으로 피드백 생성. threshold보다 낮은 모든 condition에 대한 피드백 제공."""
        from collections import defaultdict

        feedback_map = {
            "팔꿈치가 어깨보다 안쪽에 위치하지 않음": ("Keep elbows aligned with shoulders", "팔꿈치를 어깨와 정렬하세요"),
            "몸통과 엉덩이의 정렬 유지": ("Keep spine and hips aligned", "몸통과 엉덩이의 정렬을 유지하세요"),
            "상체의 지면으로부터 충분한 거리 유지": ("Maintain upper body height", "상체를 지면으로부터 충분히 올리세요"),
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


