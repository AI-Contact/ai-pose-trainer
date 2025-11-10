from __future__ import annotations

from typing import List, Tuple, Optional
import cv2


class FrameRenderer:
    def __init__(self, window_title: str, display_scale: float = 1.5) -> None:
        self.window_title = window_title
        self.display_scale = display_scale
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

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
    ):
        h0, w0 = frame.shape[:2]
        ref = 720.0
        scale = max(0.5, min(2.0, min(h0, w0) / ref))

        # 스케일된 폰트/두께/간격
        fs_lg = 1.2 * scale
        fs_md = 0.9 * scale
        fs_sm = 0.8 * scale
        th_bold = max(1, int(round(3.5 * scale)))
        th_norm = max(1, int(round(2.5 * scale)))
        gap_lg = int(round(40 * scale))
        gap_md = int(round(35 * scale))
        gap_sm = int(round(25 * scale))
        y = int(round(30 * scale))
        # 카운터 표시
        if exercise_name == "push_up" and "reps" in counters:
            cv2.putText(frame, f"Push-ups: {counters['reps']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_lg, (255, 255, 0), th_bold)
            y += gap_lg
        elif exercise_name == "crunch" and "reps" in counters:
            cv2.putText(frame, f"Crunches: {counters['reps']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_lg, (255, 255, 0), th_bold)
            y += gap_lg
        elif exercise_name == "plank" and "reps" in counters:
            cv2.putText(frame, f"Plank: {counters['reps']} reps (3s each)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_lg, (255, 255, 0), th_bold)
            y += gap_lg
        elif exercise_name == "cross_lunge" and ("left_reps" in counters or "right_reps" in counters):
            left_reps = counters.get("left_reps", 0)
            right_reps = counters.get("right_reps", 0)
            cv2.putText(frame, f"Cross Lunges: L{left_reps} R{right_reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_lg, (255, 255, 0), th_bold)
            y += gap_lg

        # 상태 표시
        if state:
            state_colors = {
                "idle": (128, 128, 128), 
                "down": (0, 0, 255), 
                "up": (0, 255, 0), 
                "hold": (0, 200, 255), 
                "move": (200, 200, 0),
                "left_down": (0, 0, 255),
                "right_down": (0, 0, 255),
            }
            color = state_colors.get(state, (255, 255, 255))
            cv2.putText(frame, f"State: {state.upper()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_md, color, th_norm)
            y += gap_md

        cv2.putText(frame, f"Frame: {frame_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_md, (0, 255, 0), th_norm)
        y += int(round(30 * scale))

        # 피드백 표시 (있으면, 영어로)
        if feedback:
            feedback_en, _ = feedback if isinstance(feedback, tuple) else (feedback, None)
            if feedback_en:
                cv2.putText(frame, f"Feedback: {feedback_en}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_md, (0, 165, 255), th_bold)
                y += gap_md

        # 조건 표시
        if results:
            header = "Condition Checks:"
            cv2.putText(frame, header, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_sm, (255, 255, 255), th_norm)
            y += gap_sm
            for cond_name, prob, threshold, is_true in results:
                color = (0, 255, 0) if is_true else (0, 0, 255)
                status = "OK" if is_true else "BAD"
                cond_eng = condition_name_map.get(cond_name, cond_name)
                text = f"{cond_eng}: {status} ({prob:.3f} >= {threshold:.3f})"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs_sm, color, th_norm)
                y += gap_sm

        # 포즈 랜드마크 그리기
        if pose_landmarks is not None and pose_connections is not None:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            cr = max(2, int(round(3 * scale)))
            th_lm = max(2, int(round(3 * scale)))
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                pose_connections,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=th_lm, circle_radius=cr),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=th_lm),
            )

        # 확대 출력
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_title, frame_resized)


