import autorootcwd
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

import click

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_pose_from_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
            landmarks_all.append(landmarks)
        else:
            landmarks_all.append(np.zeros((33, 3)))  # mediapipe pose는 33개 관절

    cap.release()
    np.save(save_path, np.array(landmarks_all))
    print(f"Saved to {save_path}")

def extract_pose_from_frame_dir(frame_dir, save_path, pattern="*.jpg"):
    frame_paths = sorted(Path(frame_dir).glob(pattern))
    landmarks_all = []

    for img_path in frame_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark], dtype=np.float32)
            landmarks_all.append(landmarks)
        else:
            landmarks_all.append(np.zeros((33, 3), dtype=np.float32))

    np.save(save_path, np.array(landmarks_all, dtype=np.float32))
    print(f"Saved to {save_path}")

def batch_extract_from_root(root_dir, image_patterns=("*.jpg", "*.png"), overwrite=False):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root_dir}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    for seq_dir in sorted(subdirs):
        # Save directly inside the sequence folder as skel.npy
        out_path = seq_dir / "skel.npy"
        if out_path.exists() and not overwrite:
            print(f"Skip existing: {out_path}")
            continue

        # 프레임 디렉토리 결정: `frames/`가 있으면 그 안에서 탐색, 없으면 바로 하위 파일 탐색
        frames_dir = seq_dir / "frames"
        search_dir = frames_dir if frames_dir.exists() and frames_dir.is_dir() else seq_dir

        # 우선 jpg, 없으면 png 순서로 시도
        frame_glob = None
        for pat in image_patterns:
            if any(search_dir.glob(pat)):
                frame_glob = pat
                break
        if frame_glob is None:
            print(f"No frames found in {seq_dir}")
            continue

        try:
            extract_pose_from_frame_dir(str(search_dir), str(out_path), pattern=frame_glob)
        except Exception as e:
            print(f"Failed: {seq_dir} -> {e}")

def batch_extract_all(exercise_dir, image_patterns=("*.jpg",), overwrite=False):
    exercise_path = Path(exercise_dir)
    for subject_dir in sorted([p for p in exercise_path.iterdir() if p.is_dir()]):
        print(f"Processing subject: {subject_dir.name}")
        batch_extract_from_root(str(subject_dir), image_patterns=image_patterns, overwrite=overwrite)


@click.command()
@click.option("--exercise", default="leg_raise", type=click.Choice(["push_up", "plank", "crunch", "cross_lunge", "leg_raise"]), help="운동 종류")
def main(exercise: str):

    base_dir = Path(f"data/{exercise}")
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    print(f"Extracting skeletons for exercise: {exercise}")
    print(f"Base directory: {base_dir}")
    
    batch_extract_all(
        str(base_dir),
    )

if __name__ == "__main__":
    main()