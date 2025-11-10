from __future__ import annotations

import numpy as np
from typing import Union


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """3D 상에서 세 점이 주어졌을 때 각도 계산.
    
    Args:
        a: 첫 번째 점 (3D 좌표)
        b: 두 번째 점 (3D 좌표, 각도의 꼭짓점)
        c: 세 번째 점 (3D 좌표)
    
    Returns:
        각도 (도 단위, 0~180도)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 벡터 계산: b를 기준으로 a와 c로 향하는 벡터
    ab = a - b
    cb = c - b
    
    # 내적을 이용한 각도 계산
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return 0.0  # 벡터가 너무 작으면 0 반환
    
    cosine_angle = dot_product / (norm_ab * norm_cb)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return float(angle)


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D 상에서 세 점이 주어졌을 때 각도 계산.
    
    Args:
        a: 첫 번째 점 (2D 좌표, x, y만 사용)
        b: 두 번째 점 (2D 좌표, 각도의 꼭짓점, x, y만 사용)
        c: 세 번째 점 (2D 좌표, x, y만 사용)
    
    Returns:
        각도 (도 단위, 0~180도)
    """
    a = np.array(a[:2])  # x, y만 사용
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    # 벡터 계산: b를 기준으로 a와 c로 향하는 벡터
    ab = a - b
    cb = c - b
    
    # 내적을 이용한 각도 계산
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return 0.0  # 벡터가 너무 작으면 0 반환
    
    cosine_angle = dot_product / (norm_ab * norm_cb)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return float(angle)


def calculate_angle_with_horizontal_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D 상에서 두 점이 주어졌을 때, 두 점을 이은 선과 수평선 사이의 각도 계산.
    
    Args:
        a: 첫 번째 점 (2D 좌표, x, y만 사용)
        b: 두 번째 점 (2D 좌표, x, y만 사용)
    
    Returns:
        각도 (도 단위, 0~90도)
    """
    a = np.array(a[:2])  # x, y만 사용
    b = np.array(b[:2])
    
    # 두 점을 이은 벡터
    ab_vec = b - a
    
    # 수평선 벡터 (x 방향)
    horizontal_vec = np.array([1.0, 0.0])
    
    # 두 벡터 사이의 각도 계산
    dot_product = np.dot(ab_vec, horizontal_vec)
    norm_ab = np.linalg.norm(ab_vec)
    
    if norm_ab < 1e-6:
        return 0.0  # 벡터가 너무 작으면 0 반환
    
    # 각도: 0도(수평) ~ 90도(수직)
    cos_angle = abs(dot_product) / (norm_ab * np.linalg.norm(horizontal_vec))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return float(angle)
