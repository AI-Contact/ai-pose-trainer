from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2


class VideoSource:
    def __init__(self, realtime: bool, video: Optional[Path], camera_id: int = 0) -> None:
        if realtime:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with ID: {camera_id}")
        else:
            if video is None:
                raise ValueError("video path is required when realtime is False")
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video}")

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def fps(self) -> float:
        import cv2 as _cv2
        fps = self.cap.get(_cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 30.0

    def frame_size(self) -> tuple[int, int]:
        import cv2 as _cv2
        w = int(self.cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            # fallback은 최초 프레임에서 shape 사용 권장
            return (0, 0)
        return (w, h)


