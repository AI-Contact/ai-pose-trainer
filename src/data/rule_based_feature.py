import numpy as np
from pathlib import Path

def calculate_angle(a, b, c):
    """세 점 a, b, c로 각도 계산"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_pushup_features(pose_seq):
    """
    pose_seq: (frames, 33, 3) mediapipe 좌표
    반환: rule-based feature 벡터 (frames, N_features)
    """
    from src.utils.landmarks import (
        L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
        L_HIP, R_HIP, L_KNEE, R_KNEE,
        L_EAR, R_EAR, NOSE, L_ANKLE, R_ANKLE
    )
    
    features = []

    for frame in pose_seq:

        left_elbow_angle = calculate_angle(frame[L_SH], frame[L_EL], frame[L_WR])
        right_elbow_angle = calculate_angle(frame[R_SH], frame[R_EL], frame[R_WR])
        mean_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        spine_angle = calculate_angle(frame[L_SH], frame[L_HIP], frame[L_KNEE])
        head_tilt = calculate_angle(frame[L_EAR], frame[L_SH], frame[L_HIP])

        features.append([
            mean_elbow_angle,
            spine_angle,
            head_tilt
        ])

    return np.array(features)


def horizontal_angle_xy(point1, point2):
    """x-y 평면에서의 수평각도 (0도가 완전히 수평)"""
    dx = abs(point2[0] - point1[0])  # x 차이
    dy = abs(point2[1] - point1[1])  # y 차이
    angle = np.degrees(np.arctan2(dy, dx + 1e-6))
    return angle


def extract_plank_features(pose_seq):
    """
    pose_seq: (frames, 33, 3) mediapipe 좌표 (옆 모습 기준, 2D)
    반환: rule-based feature 벡터 (frames, N_features)
    
    Plank 판단에 필요한 feature들:
    - elbow_shoulder_alignment: 어깨-팔꿈치 정렬 (팔꿈치 위치 판단용)
    - elbow_distance: 두 팔꿈치 사이 거리 (팔꿈치 정렬 판단용)
    - spine_alignment: 척추 정렬 (몸통-엉덩이 정렬 판단용)
    - shoulder_height: 상체 높이 (지면으로부터 거리 판단용)
    
    Note: 조건과 feature는 1:1 매핑이 아니라, 여러 조건 판단에 필요한 feature를 계산
    """
    features = []

    from src.utils.landmarks import L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP, L_KNEE, R_KNEE
    
    for frame in pose_seq:
        # Feature: 어깨-팔꿈치 정렬 (팔꿈치 위치 판단에 사용)
        # 옆 모습 기준 (2D 좌표만 있음): 어깨와 팔꿈치의 수직(y) 위치와 수평 각도로 판단
        # 올바른 플랭크: 어깨와 팔꿈치가 거의 같은 높이 (y 좌표가 비슷)
        left_shoulder_y = frame[L_SH][1]
        left_elbow_y = frame[L_EL][1]
        right_shoulder_y = frame[R_SH][1]
        right_elbow_y = frame[R_EL][1]
        
        # 어깨-팔꿈치 y 좌표 차이 (작을수록 같은 높이, 좋음)
        left_y_diff = abs(left_shoulder_y - left_elbow_y)
        right_y_diff = abs(right_shoulder_y - right_elbow_y)
        mean_y_diff = (left_y_diff + right_y_diff) / 2
        
        # 어깨-팔꿈치 벡터의 수평각도 (x-y 평면에서)
        # 0도가 완전히 수평 (어깨-팔꿈치가 같은 높이에 있고 수평)
        left_horizontal_angle = horizontal_angle_xy(frame[L_SH], frame[L_EL])
        right_horizontal_angle = horizontal_angle_xy(frame[R_SH], frame[R_EL])
        mean_horizontal_angle = (left_horizontal_angle + right_horizontal_angle) / 2
        
        # Feature: y 차이와 수평각도 결합
        # y 차이가 작고(같은 높이), 각도가 작을수록(수평) 좋음
        # 정규화를 위해 각도를 비율로 변환 (0~90도 -> 0~1)
        normalized_angle = mean_horizontal_angle / 90.0
        # 두 feature 결합: y 차이 + 정규화된 각도 (가중 평균)
        elbow_shoulder_alignment = mean_y_diff + normalized_angle * 0.1  # 작을수록 좋음
        
        # 두 팔꿈치 사이의 거리 (옆 모습에서 팔꿈치 정렬 판단용)
        left_elbow = frame[L_EL]
        right_elbow = frame[R_EL]
        elbow_distance = np.linalg.norm(left_elbow[:2] - right_elbow[:2])  # x-y 평면에서 거리

        # Feature: 척추 정렬 (몸통-엉덩이 정렬 판단에 사용)
        # 어깨-엉덩이-무릎 각도 (180도에 가까울수록 정렬됨)
        left_spine_alignment = calculate_angle(frame[L_SH], frame[L_HIP], frame[L_KNEE])
        right_spine_alignment = calculate_angle(frame[R_SH], frame[R_HIP], frame[R_KNEE])
        mean_spine_alignment = (left_spine_alignment + right_spine_alignment) / 2

        # Feature: 상체 높이 (지면으로부터 거리 판단에 사용)
        # 어깨 y 좌표 (작을수록 높음, MediaPipe는 y가 아래쪽이 클수록 증가)
        left_shoulder_y = frame[L_SH][1]
        right_shoulder_y = frame[R_SH][1]
        mean_shoulder_height = (left_shoulder_y + right_shoulder_y) / 2

        features.append([
            elbow_shoulder_alignment,      # 어깨-팔꿈치 정렬 (작을수록 좋음: 같은 높이에 있고 수평)
            elbow_distance,                # 두 팔꿈치 사이 거리 (큰 값일수록 팔꿈치가 벌어짐, 좋음)
            mean_spine_alignment,          # 척추 정렬 각도 (180도에 가까울수록 좋음)
            mean_shoulder_height,          # 어깨 높이 (작을수록 상체가 높음, 좋음)
        ])

    return np.array(features)


def extract_crunch_features(pose_seq):
    """
    pose_seq: (frames, 33, 3) mediapipe 좌표 (옆 모습 기준, 2D)
    반환: rule-based feature 벡터 (frames, N_features)
    
    Crunch 판단에 필요한 feature들:
    - hip_stability: 엉덩이/허리 고정 정도 (허리 지면 고정 판단용)
    - shoulder_height: 어깨 높이 (견갑골 충분히 올라옴 판단용)
    - shoulder_velocity: 어깨 변화율 (어깨반동 없음 판단용)
    - min_shoulder_height: 최소 어깨 높이 (이완시 긴장 유지 판단용)
    
    Note: 조건과 feature는 1:1 매핑이 아니라, 여러 조건 판단에 필요한 feature를 계산
    """
    from src.utils.landmarks import (
        L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE,
    )
    
    features = []
    
    # 시퀀스 레벨 feature를 위해 이전 프레임 정보 저장
    prev_shoulder_y = None
    
    for i, frame in enumerate(pose_seq):
        # Feature: 엉덩이/허리 고정 정도 (허리 지면 고정 판단에 사용)
        # 엉덩이 y 좌표 (작을수록 높음, MediaPipe는 y가 아래쪽이 클수록 증가)
        # 고정되어 있으면 y 좌표 변화가 작아야 함
        left_hip_y = frame[L_HIP][1]
        right_hip_y = frame[R_HIP][1]
        mean_hip_y = (left_hip_y + right_hip_y) / 2
        
        # 전체 시퀀스에서 엉덩이 y 좌표의 분산을 계산하기 위해 저장
        # (실제로는 프레임별로는 불가능하므로, 현재 프레임의 엉덩이 위치만 저장)
        # 대신 시퀀스 전체를 보고 싶으면 이후에 추가 feature로 계산 가능
        
        # Feature: 어깨 높이 (견갑골 충분히 올라옴 판단에 사용)
        # 어깨 y 좌표 (작을수록 올라옴, 좋음)
        left_shoulder_y = frame[L_SH][1]
        right_shoulder_y = frame[R_SH][1]
        mean_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Feature: 어깨 변화율 (어깨반동 없음 판단에 사용)
        # 이전 프레임과의 차이 (변화가 급격하면 반동이 있는 것)
        if prev_shoulder_y is not None:
            shoulder_velocity = abs(mean_shoulder_y - prev_shoulder_y)
        else:
            shoulder_velocity = 0.0  # 첫 프레임은 0
        
        prev_shoulder_y = mean_shoulder_y
        
        # Feature: 어깨-허리 거리 (상체가 올라온 정도)
        # 어깨와 엉덩이의 y 좌표 차이 (클수록 상체가 많이 올라옴, 좋음)
        # 엉덩이가 아래이므로 mean_hip_y > mean_shoulder_y, 차이는 양수
        shoulder_hip_distance = mean_hip_y - mean_shoulder_y
        
        # Feature: 상체 각도 (상체가 얼마나 올라와 있는지)
        # 어깨-엉덩이-무릎 각도 (작을수록 상체가 많이 올라옴)
        left_upper_body_angle = calculate_angle(frame[L_SH], frame[L_HIP], frame[L_KNEE])
        right_upper_body_angle = calculate_angle(frame[R_SH], frame[R_HIP], frame[R_KNEE])
        mean_upper_body_angle = (left_upper_body_angle + right_upper_body_angle) / 2
        
        features.append([
            mean_hip_y,                      # 엉덩이 높이 (작을수록 높음, 고정 여부 판단)
            mean_shoulder_y,                  # 어깨 높이 (작을수록 올라옴, 좋음)
            shoulder_velocity,                # 어깨 변화율 (작을수록 반동 없음, 좋음)
            shoulder_hip_distance,            # 어깨-허리 거리 (클수록 상체 많이 올라옴, 좋음)
            mean_upper_body_angle,            # 상체 각도 (작을수록 상체 많이 올라옴, 좋음)
        ])
    
    # Feature: 시퀀스 레벨 - 최소 어깨 높이 (이완시 긴장 유지 판단에 사용)
    # 전체 시퀀스에서 가장 낮은(큰 y 값) 어깨 높이를 찾아서 각 프레임에 추가
    # 하지만 프레임별 feature이므로 일단 현재 프레임의 어깨 높이만 사용
    # (이후 sliding window나 전체 시퀀스를 보는 방식으로 확장 가능)
    
    return np.array(features)


def extract_cross_lunge_features(pose_seq):
    """
    pose_seq: (frames, 33, 3) mediapipe 좌표 (정면 기준, 2D 사용)
    반환: rule-based feature 벡터 (frames, N_features)

    Cross Lunge 주요 feature:
    - front_hip_knee_distance: 앞다리 엉덩이-무릎 거리
    - shoulder_lr_x_diff: 좌/우 어깨 x 좌표 차이
    - hip_lr_y_diff: 좌/우 엉덩이 y 좌표 차이
    - front_leg_angle: 앞다리 발목-무릎-엉덩이 각도
    """
    from src.utils.landmarks import (
        L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    )
    
    features = []
    
    for frame in pose_seq:
        # 어깨 중심, 엉덩이 중심 계산 (x-y만 사용)
        shoulder_center = (frame[L_SH][:2] + frame[R_SH][:2]) / 2
        hip_center = (frame[L_HIP][:2] + frame[R_HIP][:2]) / 2
        
        # 앞다리 판단: 엉덩이-무릎 거리가 더 큰 다리를 앞다리로 추정
        left_hip_knee_dist = np.linalg.norm(frame[L_HIP][:2] - frame[L_KNEE][:2])
        right_hip_knee_dist = np.linalg.norm(frame[R_HIP][:2] - frame[R_KNEE][:2])

        if left_hip_knee_dist >= right_hip_knee_dist:
            front_indices = (L_HIP, L_KNEE, L_ANKLE)
            front_hip_knee_distance = left_hip_knee_dist
        else:
            front_indices = (R_HIP, R_KNEE, R_ANKLE)
            front_hip_knee_distance = right_hip_knee_dist

        front_hip_idx, front_knee_idx, front_ankle_idx = front_indices

        # Feature: 어깨 좌우 x 좌표 차이 (정면 균형)
        shoulder_lr_x_diff = abs(frame[L_SH][0] - frame[R_SH][0])

        # Feature: 엉덩이 좌우 y 좌표 차이 (골반 기울기)
        hip_lr_y_diff = abs(frame[L_HIP][1] - frame[R_HIP][1])

        # Feature: 앞다리 발목-무릎-엉덩이 각도 (앞다리 무릎 각도)
        front_leg_angle = calculate_angle(
            frame[front_ankle_idx][:3], frame[front_knee_idx][:3], frame[front_hip_idx][:3]
        )

        features.append([
            front_hip_knee_distance,
            shoulder_lr_x_diff,
            hip_lr_y_diff,
            front_leg_angle,
        ])
    
    return np.array(features)


if __name__ == "__main__":
    def batch_extract_features(root_dir, save=False, out_ext=".npy"):
        root = Path(root_dir)
        for npy_path in sorted(root.glob("*.npy")):
            pose_seq = np.load(npy_path)
            feats = extract_pushup_features(pose_seq)
            print(f"{npy_path.name}: {feats.shape}")
            # 앞 몇 행 미리보기
            print(feats[:10])
            if save:
                out_path = npy_path.with_suffix("")
                out_path = out_path.with_name(out_path.name + f"_features{out_ext}")
                if out_ext == ".npy":
                    np.save(out_path, feats.astype(np.float32))
                elif out_ext == ".csv":
                    np.savetxt(out_path, feats, delimiter=",", fmt="%.6f")
                else:
                    np.save(out_path, feats.astype(np.float32))
                print(f"Saved: {out_path}")

    batch_extract_features(r"D:\project\model-based-LSTM\data\push_up\570", save=False)