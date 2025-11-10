import autorootcwd
from pathlib import Path
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import uuid

from src.exercises.registry import create_exercise
from src.pipeline.pose_extractor import PoseExtractor
from src.pipeline.sequence_buffer import SequenceBuffer
from src.pipeline.inference_service import InferenceService
from src.pipeline.renderer import FrameRenderer
from src.pipeline.score_calculator import ScoreCalculator
from src.utils.thresholds import load_thresholds, match_thresholds
from src.utils.labels import load_condition_names
from src.pipeline.rule_blender import blend_probs

app = Flask(__name__)
CORS(app)

# 전역 변수
pipeline_components = {
    'exercise': None,
    'pose': None,
    'buf': None,
    'infer': None,
    'video_src': None,
    'thresholds': None,
    'condition_names': None,
    'renderer': None,
    'score_calculator': None,
    'mp_pose': mp.solutions.pose,
    'is_running': False,
    'is_realtime': True,
    'video_path': None,
    'start_time': None,
    'warmup_duration': 3.0,
    'current_feedback_en': None,
    'current_feedback_ko': None,
    'frame_count': 0,
    'lock': threading.Lock(),
}

# 업로드된 비디오 저장 디렉토리
UPLOAD_FOLDER = Path('demo/uploads')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

def init_pipeline(exercise_name: str, is_realtime: bool = True, camera_id: int = 0, video_path: Path = None):
    """파이프라인 컴포넌트 초기화"""
    with pipeline_components['lock']:
        # 기존 리소스 정리
        if pipeline_components['video_src'] is not None:
            try:
                pipeline_components['video_src'].release()
            except:
                pass
            pipeline_components['video_src'] = None
        
        if pipeline_components['pose'] is not None:
            try:
                pipeline_components['pose'].close()
            except:
                pass
            pipeline_components['pose'] = None
        
        # Threshold 및 체크포인트 경로 설정
        thresholds_csv = Path(f"result/{exercise_name}/optimal_thresholds.csv")
        ckpt = Path("weight") / f"{exercise_name}_final.ckpt"
        
        # 파이프라인 컴포넌트 초기화
        pipeline_components['exercise'] = create_exercise(exercise_name)
        pipeline_components['pose'] = PoseExtractor(min_detection_confidence=0.5)
        pipeline_components['buf'] = SequenceBuffer(max_len=32)
        pipeline_components['infer'] = InferenceService(str(ckpt), device="cpu")
        
        # 비디오 소스 초기화
        if is_realtime:
            pipeline_components['video_src'] = cv2.VideoCapture(camera_id)
            pipeline_components['is_realtime'] = True
            pipeline_components['video_path'] = None
        else:
            if video_path is None:
                raise ValueError("video_path is required when is_realtime is False")
            pipeline_components['video_src'] = cv2.VideoCapture(str(video_path))
            pipeline_components['is_realtime'] = False
            pipeline_components['video_path'] = video_path
        
        if not pipeline_components['video_src'].isOpened():
            error_msg = f"Failed to open video source"
            if not is_realtime and video_path:
                error_msg += f": {video_path}"
            raise RuntimeError(error_msg)
        
        # Threshold 및 조건명 로드
        pipeline_components['thresholds'] = load_thresholds(thresholds_csv)
        pipeline_components['condition_names'] = load_condition_names(
            bool(pipeline_components['thresholds']),
            list(pipeline_components['thresholds'].keys()),
            None
        )
        
        # Renderer 초기화
        window_title = f"{exercise_name.replace('_', ' ').title()} Analysis"
        pipeline_components['renderer'] = FrameRenderer(window_title=window_title, display_scale=1.5)
        
        # Score calculator 초기화
        pipeline_components['score_calculator'] = ScoreCalculator()
        
        # 상태 초기화
        pipeline_components['current_feedback_en'] = None
        pipeline_components['current_feedback_ko'] = None
        pipeline_components['frame_count'] = 0
        # 실시간 모드일 때만 warmup 적용
        if is_realtime:
            pipeline_components['start_time'] = time.time()
        else:
            pipeline_components['start_time'] = None

def generate_frames():
    """비디오 프레임 생성기 (MJPEG 스트리밍)"""
    while True:
        # is_running 확인 및 프레임 읽기 (lock 안에서)
        with pipeline_components['lock']:
            if not pipeline_components['is_running']:
                break
            if pipeline_components['video_src'] is None:
                time.sleep(0.1)
                continue
            
            # 비디오 소스와 필요한 컴포넌트 가져오기
            video_src = pipeline_components['video_src']
            exercise = pipeline_components['exercise']
            pose = pipeline_components['pose']
            buf = pipeline_components['buf']
            infer = pipeline_components['infer']
            renderer = pipeline_components['renderer']
            score_calculator = pipeline_components['score_calculator']
            is_realtime = pipeline_components['is_realtime']
            start_time = pipeline_components['start_time']
            warmup_duration = pipeline_components['warmup_duration']
            condition_names = pipeline_components['condition_names']
            thresholds = pipeline_components['thresholds']
            mp_pose = pipeline_components['mp_pose']
            
            # lock 안에서 프레임 읽기 (FFmpeg 멀티스레딩 오류 방지)
            ret, frame = video_src.read()
            
            if not ret:
                # 비디오 파일이 끝났으면 종료
                if not is_realtime:
                    pipeline_components['is_running'] = False
                break
        
        # lock 밖에서 프레임 처리 (lock 안에서 읽은 프레임 사용)
        if not ret:
            break
        
        # Warmup 기간 확인 (실시간 모드일 때만)
        if is_realtime and start_time is not None:
            current_time = time.time()
            elapsed_time = current_time - start_time
            is_warmup = elapsed_time < warmup_duration
        else:
            is_warmup = False
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, pose_landmarks = pose.extract(frame_rgb)
        
        results = []
        frame_evaluations = []
        
        if landmarks is not None:
            exercise.update_state(landmarks)
            buf.push(landmarks)
            
            if len(buf) >= 10 and not is_warmup:
                # 추론 수행
                feats = exercise.extract_features(buf.as_array())
                probs = infer.predict(feats)
                
                # 룰-모델 블렌딩
                if condition_names:
                    probs = blend_probs(
                        exercise.name,
                        exercise.get_state(),
                        landmarks,
                        condition_names,
                        probs
                    )
                
                targets = exercise.conditions_for_state()
                if targets == []:
                    # 상태에 의해 표시하지 않음
                    results = []
                else:
                    # None이면 전체 조건 평가, list면 필터링
                    filter_names = targets  # None이면 전체, list면 필터
                    results = match_thresholds(
                        probs,
                        condition_names,
                        thresholds,
                        filter_names
                    )
                    # 평가 데이터 수집
                    for cond_name, prob, threshold, _ in results:
                        frame_evaluations.append((cond_name, prob, threshold))
        
        # 점수 계산기 업데이트
        if not is_warmup:
            with pipeline_components['lock']:
                completed_rep = score_calculator.update(
                    exercise.counters(),
                    frame_evaluations
                )
                
                if completed_rep is not None:
                    rep_evaluations = score_calculator.get_rep_evaluations(completed_rep)
                    rep_feedback_en, rep_feedback_ko = exercise.get_feedback_from_rep_evaluations(rep_evaluations)
                    
                    if rep_feedback_ko:
                        pipeline_components['current_feedback_en'] = rep_feedback_en
                        pipeline_components['current_feedback_ko'] = rep_feedback_ko
                    else:
                        pipeline_components['current_feedback_en'] = "Good job"
                        pipeline_components['current_feedback_ko'] = "잘하고 있어요!"
        
        # 오버레이 그리기
        with pipeline_components['lock']:
            frame_count = pipeline_components['frame_count']
            pipeline_components['frame_count'] += 1
            current_feedback_en = pipeline_components['current_feedback_en']
            current_feedback_ko = pipeline_components['current_feedback_ko']
        
        renderer.draw_overlay(
            frame=frame,
            exercise_name=exercise.name,
            state=exercise.get_state(),
            counters=exercise.counters(),
            results=results,
            condition_name_map=exercise.condition_name_map(),
            frame_count=frame_count,
            pose_landmarks=pose_landmarks,
            pose_connections=mp_pose.POSE_CONNECTIONS,
            feedback=(current_feedback_en, current_feedback_ko),
        )
        
        # JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # MJPEG 스트리밍 형식으로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 프레임 레이트 제어
        if is_realtime:
            # 실시간 모드: 약 30 FPS
            time.sleep(1/30)
        else:
            # 비디오 모드: 원본 비디오 FPS에 맞춤
            with pipeline_components['lock']:
                if pipeline_components['video_src'] is not None:
                    fps = pipeline_components['video_src'].get(cv2.CAP_PROP_FPS)
                else:
                    fps = 0
            if fps > 0:
                time.sleep(1/fps)
            else:
                time.sleep(1/30)

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 엔드포인트"""
    def generate():
        # is_running이 False이거나 video_src가 None이면 대기
        while True:
            with pipeline_components['lock']:
                if pipeline_components['is_running'] and pipeline_components['video_src'] is not None:
                    is_realtime = pipeline_components['is_realtime']
                    break
            time.sleep(0.1)
        
        # 프레임 생성 시작
        # 실시간 모드: generate_frames가 종료되어도 다시 시작 가능
        # 비디오 모드: generate_frames가 종료되면 (비디오 끝) 재시작하지 않음
        while True:
            for frame_data in generate_frames():
                yield frame_data
            
            # generate_frames가 종료됨
            with pipeline_components['lock']:
                # 비디오 모드이거나 is_running이 False이면 종료
                if not is_realtime or not pipeline_components['is_running'] or pipeline_components['video_src'] is None:
                    break
            # 실시간 모드에서만 재시작 시도
            time.sleep(0.1)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """비디오 파일 업로드"""
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'message': '비디오 파일이 없습니다.'
        }), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '파일이 선택되지 않았습니다.'
        }), 400
    
    # 파일 저장
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    video_path = UPLOAD_FOLDER / unique_filename
    file.save(str(video_path))
    
    return jsonify({
        'success': True,
        'video_path': str(video_path),
        'filename': unique_filename
    })

@app.route('/api/start', methods=['POST'])
def start_exercise():
    """운동 시작"""
    import flask
    data = flask.request.get_json()
    exercise_name = data.get('exercise', 'push_up')
    mode = data.get('mode', 'realtime')  # 'realtime' or 'video'
    camera_id = data.get('camera_id', 0)
    video_path = data.get('video_path', None)
    
    try:
        if mode == 'realtime':
            init_pipeline(exercise_name, is_realtime=True, camera_id=camera_id)
        else:
            if video_path is None:
                return jsonify({
                    'success': False,
                    'message': '비디오 경로가 필요합니다.'
                }), 400
            # 비디오 경로를 절대 경로로 변환
            video_path_abs = Path(video_path).resolve()
            if not video_path_abs.exists():
                return jsonify({
                    'success': False,
                    'message': f'비디오 파일을 찾을 수 없습니다: {video_path_abs}'
                }), 400
            init_pipeline(exercise_name, is_realtime=False, video_path=video_path_abs)
        
        pipeline_components['is_running'] = True
        return jsonify({
            'success': True,
            'message': f'{exercise_name} 운동이 시작되었습니다.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_exercise():
    """운동 중지"""
    with pipeline_components['lock']:
        pipeline_components['is_running'] = False
        
        # 점수 계산기 최종화
        if pipeline_components['score_calculator']:
            pipeline_components['score_calculator'].finalize()
        
        # 리소스 정리
        if pipeline_components['video_src']:
            try:
                pipeline_components['video_src'].release()
            except:
                pass
            pipeline_components['video_src'] = None
        
        if pipeline_components['pose']:
            try:
                pipeline_components['pose'].close()
            except:
                pass
            pipeline_components['pose'] = None
        
        # 결과 반환
        result = {
            'success': True,
            'message': '운동이 중지되었습니다.',
        }
        
        if pipeline_components['score_calculator']:
            rep_count = pipeline_components['score_calculator'].get_rep_count()
            if rep_count > 0:
                all_rep_scores = pipeline_components['score_calculator'].get_all_rep_scores()
                total_score = pipeline_components['score_calculator'].get_total_score()
                
                result['rep_count'] = rep_count
                result['rep_scores'] = {str(k): v for k, v in all_rep_scores.items()}
                result['total_score'] = total_score
        
        return jsonify(result)

@app.route('/api/status', methods=['GET'])
def get_status():
    """현재 상태 조회"""
    with pipeline_components['lock']:
        status = {
            'is_running': pipeline_components['is_running'],
            'is_realtime': pipeline_components['is_realtime'],
            'frame_count': pipeline_components['frame_count'],
        }
        
        if pipeline_components['exercise']:
            status['exercise'] = pipeline_components['exercise'].name
            status['state'] = pipeline_components['exercise'].get_state()
            status['counters'] = pipeline_components['exercise'].counters()
            status['feedback_en'] = pipeline_components['current_feedback_en']
            status['feedback_ko'] = pipeline_components['current_feedback_ko']
            
            # Warmup 상태
            if pipeline_components['start_time']:
                elapsed = time.time() - pipeline_components['start_time']
                status['warmup_remaining'] = max(0, pipeline_components['warmup_duration'] - elapsed)
                status['is_warmup'] = elapsed < pipeline_components['warmup_duration']
        
        if pipeline_components['score_calculator']:
            rep_count = pipeline_components['score_calculator'].get_rep_count()
            if rep_count > 0:
                all_rep_scores = pipeline_components['score_calculator'].get_all_rep_scores()
                total_score = pipeline_components['score_calculator'].get_total_score()
                
                status['rep_count'] = rep_count
                status['rep_scores'] = {str(k): v for k, v in all_rep_scores.items()}
                status['total_score'] = total_score
        
        return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
