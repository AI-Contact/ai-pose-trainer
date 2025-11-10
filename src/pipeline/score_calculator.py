from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class ScoreCalculator:
    """운동 반복(rep)별 점수 계산 및 최종 점수 합산을 담당하는 클래스."""

    def __init__(self):
        self._rep_evaluations: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)
        # rep 번호 -> [(condition_name, prob, threshold), ...]
        self._current_rep = 0  # rep이 시작되기 전에는 0
        self._prev_counters: Dict[str, int] = {}
        self._rep_scores: Dict[int, float] = {}  # rep 번호 -> 점수

    def update(
        self,
        counters: Dict[str, int],
        evaluations: List[Tuple[str, float, float]],
    ) -> Optional[int]:
        """프레임별 평가 결과를 업데이트하고, 새로운 rep가 시작되었는지 확인.

        Args:
            counters: 현재 카운터 (예: {"reps": 3}, {"left_reps": 2, "right_reps": 1})
            evaluations: 현재 프레임의 평가 결과 [(condition_name, prob, threshold), ...]

        Returns:
            완료된 rep 번호 (rep가 완료된 경우), None (rep가 완료되지 않은 경우)
        """
        # 총 rep 수 계산 (cross_lunge는 left_reps + right_reps)
        current_total_reps = sum(counters.values())

        # 이전 총 rep 수 계산
        prev_total_reps = sum(self._prev_counters.values())

        # rep 완료 감지: 카운터가 증가했는지 확인
        completed_rep = None

        # 첫 번째 rep의 경우, 카운터가 0일 때부터 rep 1에 대한 평가 결과를 수집
        if self._current_rep == 0:
            # 첫 번째 rep 시작 (카운터가 0일 때부터 시작)
            self._current_rep = 1

        # 현재 rep의 평가 결과 수집 (rep이 시작된 경우에만)
        if self._current_rep >= 1:
            self._rep_evaluations[self._current_rep].extend(evaluations)

        if current_total_reps > prev_total_reps:
            # 새로운 rep 완료
            # 완료된 rep 번호는 현재 _current_rep
            completed_rep = self._current_rep
            
            # 완료된 rep의 점수 계산 (해당 rep 동안 수집된 평가 결과 기반)
            if completed_rep >= 1 and completed_rep not in self._rep_scores:
                if completed_rep in self._rep_evaluations and self._rep_evaluations[completed_rep]:
                    self._rep_scores[completed_rep] = self._calculate_rep_score(
                        self._rep_evaluations[completed_rep]
                    )
            
            # 새로운 rep 시작 (rep 번호는 1부터 시작)
            self._current_rep = current_total_reps + 1

        self._prev_counters = counters.copy()
        return completed_rep

    def _calculate_rep_score(
        self, evaluations: List[Tuple[str, float, float]]
    ) -> float:
        """단일 rep의 점수 계산.

        Args:
            evaluations: rep 동안의 모든 평가 결과 [(condition_name, prob, threshold), ...]

        Returns:
            rep 점수 (0.0 ~ 1.0)
        
        점수 계산 방식:
        1. 각 프레임의 평가 결과에 대해:
           - prob >= threshold이면 점수 = 1.0
           - prob < threshold이면 점수 = prob (그대로 사용)
        2. 조건별로 모든 프레임의 점수를 평균냄
        3. 모든 조건의 평균 점수를 다시 평균냄 (rep 점수)
        """
        if not evaluations:
            return 0.0

        def calculate_score(prob: float, threshold: float) -> float:
            """threshold 기반 점수 계산: threshold 이상이면 1.0, 미만이면 prob 값 사용"""
            return 1.0 if prob >= threshold else prob

        # 조건별 평균 점수 계산
        cond_scores = defaultdict(list)
        cond_probs = defaultdict(list)  # 디버깅용: 원본 prob 값들
        cond_thresholds = {}  # 디버깅용: threshold 값들
        
        for cond_name, prob, threshold in evaluations:
            score = calculate_score(prob, threshold)
            cond_scores[cond_name].append(score)
            cond_probs[cond_name].append(prob)
            if cond_name not in cond_thresholds:
                cond_thresholds[cond_name] = threshold

        # 각 조건의 평균 점수를 구하고, 전체 평균 계산
        all_scores = []
        for cond_name, scores in cond_scores.items():
            avg_score = sum(scores) / len(scores)
            all_scores.append(avg_score)
            
            # 디버깅: 조건별 상세 정보 출력
            avg_prob = sum(cond_probs[cond_name]) / len(cond_probs[cond_name])
            threshold = cond_thresholds[cond_name]
            passed_count = sum(1 for s in scores if s >= 1.0)
            total_count = len(scores)
            print(f"    [{cond_name}] 프레임 수: {total_count}, 평균 prob: {avg_prob:.3f}, threshold: {threshold:.3f}, "
                  f"통과 프레임: {passed_count}/{total_count}, 조건 평균 점수: {avg_score:.3f}")

        if not all_scores:
            return 0.0

        rep_score = sum(all_scores) / len(all_scores)
        return rep_score

    def finalize(self) -> None:
        """마지막 rep의 점수 계산. rep이 실제로 완료된 경우에만 계산."""
        # rep이 시작되었지만 완료되지 않은 경우는 점수 계산하지 않음
        # (rep이 완료되면 이미 점수가 계산되어 있음)
        pass

    def get_rep_score(self, rep_num: int) -> Optional[float]:
        """특정 rep의 점수 반환."""
        return self._rep_scores.get(rep_num)

    def get_total_score(self) -> float:
        """모든 rep의 점수 평균."""
        if not self._rep_scores:
            return 0.0
        return sum(self._rep_scores.values()) / len(self._rep_scores)

    def get_rep_count(self) -> int:
        """완료된 rep 수 반환."""
        return len(self._rep_scores)

    def get_all_rep_scores(self) -> Dict[int, float]:
        """모든 rep의 점수 딕셔너리 반환."""
        return self._rep_scores.copy()

    def get_rep_evaluations(self, rep_num: int) -> List[Tuple[str, float, float]]:
        """특정 rep의 평가 결과 반환.
        
        Args:
            rep_num: rep 번호
            
        Returns:
            rep 동안의 모든 평가 결과 [(condition_name, prob, threshold), ...]
        """
        return self._rep_evaluations.get(rep_num, [])

