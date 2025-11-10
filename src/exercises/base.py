from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class Exercise:
    """
    운동별 공통 인터페이스.
    - 상태 머신 갱신(update_state)
    - 특징 추출(extract_features)
    - 상태/카운터/조건/표시명 제공
    """

    name: str = ""

    def extract_features(self, pose_seq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_state(self, landmarks: np.ndarray) -> None:
        raise NotImplementedError

    def get_state(self) -> str:
        return "idle"

    def conditions_for_state(self) -> Optional[List[str]]:
        """
        현재 상태에서 평가할 조건 목록. None이면 전체 조건 표시.
        """
        return None

    def condition_name_map(self) -> Dict[str, str]:
        """한글→영문 표시 매핑."""
        return {}

    def counters(self) -> Dict[str, int]:
        """반복수 등 카운터."""
        return {}

    def get_feedback(self) -> tuple[Optional[str], Optional[str]]:
        """현재 상태에서의 피드백 메시지 (영어, 한국어) 튜플 반환. (None, None)이면 피드백 없음."""
        return (None, None)

    def get_feedback_from_conditions(self, results: List[tuple]) -> tuple[Optional[str], Optional[str]]:
        """
        results: [(condition_name, prob, threshold, is_true), ...]
        조건 평가 결과를 기반으로 피드백 생성 (영어, 한국어)
        """
        return (None, None)

    def get_feedback_from_rep_evaluations(
        self, rep_evaluations: List[Tuple[str, float, float]]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        rep별 평가 결과를 기반으로 피드백 생성.
        
        Args:
            rep_evaluations: rep 동안의 모든 평가 결과 [(condition_name, prob, threshold), ...]
            
        Returns:
            (영어 피드백, 한국어 피드백) 튜플. threshold보다 낮은 모든 condition에 대한 피드백 제공.
        """
        return (None, None)


