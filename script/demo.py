import autorootcwd
from pathlib import Path
import json
import time
import click
import cv2
import numpy as np
import mediapipe as mp

from src.exercises.registry import create_exercise
from src.pipeline.pose_extractor import PoseExtractor
from src.pipeline.sequence_buffer import SequenceBuffer
from src.pipeline.inference_service import InferenceService
from src.pipeline.renderer import FrameRenderer
from src.pipeline.video_source import VideoSource
from src.pipeline.score_calculator import ScoreCalculator
from src.utils.thresholds import load_thresholds, match_thresholds
from src.utils.labels import load_condition_names
from src.pipeline.rule_blender import blend_probs


@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["push_up", "plank", "crunch", "cross_lunge"]), help="운동 종류")
@click.option("--realtime", is_flag=True, default=False, help="실시간 웹캠 모드 활성화")
@click.option("--video", default=None, type=click.Path(path_type=Path), help="입력 비디오 파일 (realtime이 False일 때 필수)")
@click.option("--camera-id", default=0, type=int, help="웹캠 ID (realtime일 때 사용, 기본값: 0)")
@click.option("--ckpt", required=False, default=None, type=click.Path(path_type=Path), help="모델 체크포인트 경로 (기본: weight/{exercise}-final.ckpt)")
@click.option("--thresholds-csv", default=None, type=click.Path(path_type=Path), help="Threshold CSV 파일 (기본값: result/{exercise}/optimal_thresholds.csv)")
@click.option("--label-json", default=None, type=click.Path(path_type=Path), help="조건명을 위한 label.json (옵션)")
def main(exercise: str, realtime: bool, video: Path, camera_id: int, ckpt: Path, thresholds_csv: Path, label_json: Path):
    if not realtime and video is None:
        raise click.BadParameter("--video is required when --realtime is not set")

    if thresholds_csv is None:
        thresholds_csv = Path(f"result/{exercise}/optimal_thresholds.csv")
    
    if ckpt is None:
        ckpt = Path("weight") / f"{exercise}_final.ckpt"

    # 파이프라인 컴포넌트
    ex = create_exercise(exercise)
    pose = PoseExtractor(min_detection_confidence=0.5)
    buf = SequenceBuffer(max_len=32)
    infer = InferenceService(str(ckpt), device="cpu")
    video_src = VideoSource(realtime=realtime, video=video, camera_id=camera_id)

    thresholds = load_thresholds(thresholds_csv)
    condition_names = load_condition_names(bool(thresholds), list(thresholds.keys()), label_json)

    window_title = f"{exercise.replace('_', ' ').title()} Analysis"
    renderer = FrameRenderer(window_title=window_title, display_scale=1.5)

    mp_pose = mp.solutions.pose
    frame_count = 0
    
    # 피드백 및 confidence 추적
    feedback_history = []  # [(frame, feedback_msg), ...]
    all_evaluations = []  # [(condition_name, prob, threshold), ...] 모든 프레임의 평가
    current_feedback_en = None  # 현재 표시 중인 피드백 (영어)
    current_feedback_ko = None  # 현재 표시 중인 피드백 (한국어)
    prev_state = "idle"  # 이전 상태 추적
    
    # 점수 계산기 초기화
    score_calculator = ScoreCalculator()

    # 결과 비디오 저장 준비
    out_dir = Path("result") / exercise
    out_dir.mkdir(parents=True, exist_ok=True)
    if realtime:
        # realtime일 때: 원본은 data/demo에, CV 영상은 result에 저장
        original_dir = Path("data") / "demo"
        original_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_dir / "realtime_original.mp4"
        out_path = out_dir / "realtime_out.mp4"
    else:
        original_path = None
        out_path = out_dir / f"{video.stem}_out.mp4"
    writer = None
    original_writer = None

    # realtime일 때만 3초 대기 후 측정 시작
    start_time = time.time() if realtime else None
    warmup_duration = 3.0  # 3초 대기

    while True:
        ret, frame = video_src.read()
        if not ret:
            break
        
        # realtime일 때만 warmup 기간 확인
        if realtime and start_time is not None:
            current_time = time.time()
            elapsed_time = current_time - start_time
            is_warmup = elapsed_time < warmup_duration
        else:
            is_warmup = False
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_original = frame.copy() if realtime else None
        landmarks, pose_landmarks = pose.extract(frame_rgb)
        
        results = []
        frame_evaluations = []
        if landmarks is not None:
            ex.update_state(landmarks)
            buf.push(landmarks)
            if len(buf) >= 10 and not is_warmup:
                # 3초 대기 후부터 측정 시작
                feats = ex.extract_features(buf.as_array())
                probs = infer.predict(feats)

                # 상태별 룰-모델 블렌딩 (운동 확장 가능)
                if condition_names:
                    probs = blend_probs(ex.name, ex.get_state(), landmarks, condition_names, probs)
                targets = ex.conditions_for_state()
                if targets == []:
                    # 상태에 의해 표시하지 않음
                    results = []
                else:
                    filter_names = targets  # None이면 전체, list면 필터
                    results = match_thresholds(probs, condition_names, thresholds, filter_names)
                    # 평가 데이터 수집 (threshold 정보 포함)
                    for cond_name, prob, threshold, _ in results:
                        all_evaluations.append((cond_name, prob, threshold))
                        frame_evaluations.append((cond_name, prob, threshold))
        
        # 점수 계산기에 현재 프레임 평가 결과 업데이트 (매 프레임마다, warmup 기간 제외)
        if not is_warmup:
            completed_rep = score_calculator.update(ex.counters(), frame_evaluations)
        else:
            completed_rep = None
        if completed_rep is not None:
            rep_score = score_calculator.get_rep_score(completed_rep)
            if rep_score is not None:
                print(f"Rep {completed_rep} 완료! 점수: {rep_score:.3f}")
            
            # rep 완료 시 해당 rep 동안의 평가 결과를 기반으로 피드백 생성
            rep_evaluations = score_calculator.get_rep_evaluations(completed_rep)
            print(f"Rep {completed_rep} 평가 결과 수: {len(rep_evaluations)}")
            if rep_evaluations:
                # 조건별 평균 prob 확인
                from collections import defaultdict
                cond_probs = defaultdict(list)
                cond_thresholds = {}
                for cond_name, prob, threshold in rep_evaluations:
                    cond_probs[cond_name].append(prob)
                    if cond_name not in cond_thresholds:
                        cond_thresholds[cond_name] = threshold
                
                print(f"Rep {completed_rep} 조건별 평균 prob:")
                for cond_name, probs in cond_probs.items():
                    avg_prob = sum(probs) / len(probs)
                    threshold = cond_thresholds[cond_name]
                    is_failed = avg_prob < threshold
                    print(f"  - {cond_name}: {avg_prob:.3f} (threshold: {threshold:.3f}, failed: {is_failed})")
            
            rep_feedback_en, rep_feedback_ko = ex.get_feedback_from_rep_evaluations(rep_evaluations)
            
            if rep_feedback_ko:
                # rep 기반 피드백이 있으면 사용
                current_feedback_en = rep_feedback_en
                current_feedback_ko = rep_feedback_ko
                feedback_history.append((frame_count, current_feedback_ko))
            else:
                # 피드백 없으면 "Good job" 표시
                current_feedback_en = "Good job"
                current_feedback_ko = "잘했어요"
                feedback_history.append((frame_count, current_feedback_ko))

        # 상태 추적
        current_state = ex.get_state()
        prev_state = current_state

        renderer.draw_overlay(
            frame=frame,
            exercise_name=ex.name,
            state=ex.get_state(),
            counters=ex.counters(),
            results=results,
            condition_name_map=ex.condition_name_map(),
            frame_count=frame_count,
            pose_landmarks=pose_landmarks,
            pose_connections=mp_pose.POSE_CONNECTIONS,
            feedback=(current_feedback_en, current_feedback_ko),  # 다음 up까지 유지되는 피드백
        )

        # VideoWriter 초기화 및 프레임 기록
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = video_src.fps()
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
            
            # realtime일 때 원본 영상도 저장
            if realtime and original_path is not None:
                original_writer = cv2.VideoWriter(str(original_path), fourcc, fps, (w, h))
                if not original_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {original_path}")

        if writer is not None:
            # CV 영상 저장 (오버레이가 그려진 프레임)
            writer.write(frame)
            
            # realtime일 때 원본 영상도 저장
            if realtime and original_writer is not None and frame_original is not None:
                original_writer.write(frame_original)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    video_src.release()
    if writer is not None:
        writer.release()
    if original_writer is not None:
        original_writer.release()
    cv2.destroyAllWindows()

    # 마지막 rep 점수 계산
    score_calculator.finalize()

    # 종료 시 요약 출력
    print("\n" + "="*60)
    print("운동 분석 요약")
    print("="*60)
    
    # 1. 카운터
    counters = ex.counters()
    if counters:
        print(f"\n카운터:")
        for key, value in counters.items():
            print(f"  - {key}: {value}")
    
    # 2. Rep별 점수 및 최종 점수
    rep_count = score_calculator.get_rep_count()
    if rep_count > 0:
        print(f"\nRep별 점수:")
        all_rep_scores = score_calculator.get_all_rep_scores()
        for rep_num in sorted(all_rep_scores.keys()):
            rep_score = all_rep_scores[rep_num]
            print(f"  - Rep {rep_num}: {rep_score:.3f}")
        
        total_score = score_calculator.get_total_score()
        print(f"\n최종 점수 (모든 Rep 평균): {total_score:.3f} ({rep_count}회)")
    else:
        print("\n완료된 Rep가 없습니다.")
    
    # 3. 피드백 통계 (상위 5개)
    if feedback_history:
        from collections import Counter
        feedback_counter = Counter([msg for _, msg in feedback_history])
        print(f"\n가장 많이 나온 피드백 (상위 5개):")
        for i, (msg, count) in enumerate(feedback_counter.most_common(5), 1):
            print(f"  {i}. {msg}: {count}회")
    
    # 4. 조건별 평균 점수 (전체 프레임 기준)
    if all_evaluations:
        from collections import defaultdict
        
        def calculate_score(prob: float, threshold: float) -> float:
            """threshold 기반 점수 계산: threshold 이상이면 1.0, 미만이면 prob 값 사용"""
            return 1.0 if prob >= threshold else prob
        
        cond_scores = defaultdict(list)
        cond_thresholds = {}  # 조건별 threshold 저장
        for cond_name, prob, threshold in all_evaluations:
            score = calculate_score(prob, threshold)
            cond_scores[cond_name].append(score)
            if cond_name not in cond_thresholds:
                cond_thresholds[cond_name] = threshold
        
        print(f"\n조건별 평균 점수 (전체 프레임 기준):")
        for cond_name in sorted(cond_scores.keys()):
            avg_score = sum(cond_scores[cond_name]) / len(cond_scores[cond_name])
            threshold_val = cond_thresholds[cond_name]
            cond_eng = ex.condition_name_map().get(cond_name, cond_name)
            print(f"  - {cond_eng}: {avg_score:.3f} (threshold: {threshold_val:.3f}, {len(cond_scores[cond_name])}프레임)")
    else:
        print("\n평가 데이터가 없습니다.")
    
    print("="*60)


if __name__ == "__main__":
    main()
