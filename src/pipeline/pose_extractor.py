from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


class PoseExtractor:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=min_detection_confidence)

    def extract(self, frame_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
        result = self._pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark], dtype=np.float32)
            return landmarks, result.pose_landmarks
        return None, None

    def close(self) -> None:
        self._pose.close()


