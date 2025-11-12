from __future__ import annotations

from typing import List, Tuple, Optional, Set
import cv2
from src.utils.landmarks import (
    L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_EAR, R_EAR, L_MOUTH, R_MOUTH
)


class FrameRenderer:
    def __init__(self, window_title: str, display_scale: float = 1.5) -> None:
        self.window_title = window_title
        self.display_scale = display_scale
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
    
    def _get_condition_joints(self, exercise_name: str, condition_name: str) -> Set[int]:
        """각 운동별 조건과 관련된 관절 인덱스 반환"""
        condition_joint_map = {
            "push_up": {
                "척추의 중립": {L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE},
                "고개 젖힘/숙임 여부": {L_EAR, R_EAR, L_MOUTH, R_MOUTH},
                "손의 위치 가슴 중앙 여부": {L_SH, R_SH, L_WR, R_WR},
                "가슴의 충분한 이동": {L_SH, R_SH, L_EL, R_EL},
            },
            "crunch": {
                "허리 지면 고정": {L_HIP, R_HIP},
                "견갑골이 지면으로부터 충분히 올라옴": {L_SH, R_SH, L_HIP, R_HIP},
                "이완시 긴장 유지": {L_SH, R_SH, L_EAR, R_EAR},
            },
            "plank": {
                "몸통과 엉덩이의 정렬 유지": {L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE},
                "상체의 지면으로부터 충분한 거리 유지": {L_SH, R_SH, L_EL, R_EL},
            },
            "cross_lunge": {
                "몸통 앞발 앞무릎 방향 일치여부": {L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_SH, R_SH},
                "상체 정면 균형잡기": {L_SH, R_SH, L_HIP, R_HIP},
            },
        }
        
        exercise_map = condition_joint_map.get(exercise_name, {})
        return exercise_map.get(condition_name, set())

    def draw_overlay(
        self,
        frame,
        exercise_name: str,
        state: str,
        counters: dict,
        results: Optional[List[Tuple[str, float, float, bool]]],
        condition_name_map: dict,
        frame_count: int,
        pose_landmarks=None,
        pose_connections=None,
        feedback: Optional[str] = None,
        warmup_remaining: Optional[float] = None,
    ):
        h0, w0 = frame.shape[:2]
        ref = 720.0
        scale = max(0.5, min(2.0, min(h0, w0) / ref))
        
        # Warmup 카운트다운 표시
        if warmup_remaining is not None and warmup_remaining > 0:
            countdown_number = int(warmup_remaining) + 1  # 3, 2, 1 표시
            if countdown_number >= 1 and countdown_number <= 3:
                h, w = frame.shape[:2]
                # 화면 중앙에 큰 숫자 표시
                font_scale = 8.0 * scale
                thickness = max(5, int(round(10 * scale)))
                text = str(countdown_number)
                
                # 텍스트 크기 계산
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # 화면 중앙 위치 계산
                x = (w - text_width) // 2
                y = (h + text_height) // 2
                
                # 배경 원 그리기 (반투명)
                circle_radius = max(text_width, text_height) // 2 + 20
                overlay = frame.copy()
                cv2.circle(overlay, (w // 2, h // 2), circle_radius, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                # 숫자 텍스트 그리기 (흰색)
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        # 포즈 랜드마크 그리기
        if pose_landmarks is not None and pose_connections is not None:
            h, w = frame.shape[:2]
            cr = max(2, int(round(2.7 * scale)))  # 관절 원 크기 (더 작게)
            th_line = max(1, int(round(2.7 * scale)))  # 연결선 두께 (더 얇게)
            alpha = 0.6  # 연결선 투명도
            
            # 피드백이 있는 조건들 찾기 (is_true가 False인 조건들)
            feedback_joint_indices: Set[int] = set()
            if results:
                for cond_name, prob, threshold, is_true in results:
                    if not is_true:  # 피드백이 필요한 조건
                        joints = self._get_condition_joints(exercise_name, cond_name)
                        feedback_joint_indices.update(joints)
            
            # 연결선 그리기 (투명도 적용)
            # 투명도를 위해 overlay 이미지 생성
            overlay = frame.copy()
            
            for connection in pose_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx >= len(pose_landmarks.landmark) or end_idx >= len(pose_landmarks.landmark):
                    continue
                
                start_landmark = pose_landmarks.landmark[start_idx]
                end_landmark = pose_landmarks.landmark[end_idx]
                
                if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                    start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                    end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                    
                    # 피드백 관련 관절이 하나라도 포함되면 빨간색, 아니면 흰색
                    if start_idx in feedback_joint_indices or end_idx in feedback_joint_indices:
                        line_color = (0, 0, 255)  # 빨간색 (BGR)
                    else:
                        line_color = (255, 255, 255)  # 흰색 (BGR)
                    
                    cv2.line(overlay, start_point, end_point, line_color, th_line)
            
            # 투명도 적용하여 연결선 그리기
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # 관절 좌표 그리기 (불투명)
            for idx in range(len(pose_landmarks.landmark)):
                landmark = pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # 피드백 관련 관절이면 빨간색, 아니면 흰색
                    if idx in feedback_joint_indices:
                        joint_color = (0, 0, 255)  # 빨간색 (BGR)
                    else:
                        joint_color = (255, 255, 255)  # 흰색 (BGR)
                    
                    cv2.circle(frame, (x, y), cr, joint_color, -1)

        # 확대 출력
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_title, frame_resized)


