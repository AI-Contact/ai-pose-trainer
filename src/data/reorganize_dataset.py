import autorootcwd
import json
import shutil
from pathlib import Path


def move_frames_into_subdir(sequence_dir: Path, frames_dirname: str = "frames") -> int:
    frames_dir = sequence_dir / frames_dirname
    frames_dir.mkdir(exist_ok=True)

    moved = 0
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for img in sequence_dir.glob(ext):
            target = frames_dir / img.name
            if target.exists():
                continue
            shutil.move(str(img), str(target))
            moved += 1
    return moved


def move_outputs(subject_dir: Path, sequence_name: str) -> None:
    # pose npy placed at subject level as <sequence>.npy -> move to pose.npy
    pose_src = subject_dir / f"{sequence_name}.npy"
    if pose_src.exists():
        pose_dst = subject_dir / sequence_name / "skel.npy"
        if not pose_dst.exists():
            shutil.move(str(pose_src), str(pose_dst))

    # features saved as <sequence>_features.npy or .csv
    feat_src_npy = subject_dir / f"{sequence_name}_features.npy"
    feat_src_csv = subject_dir / f"{sequence_name}_features.csv"
    if feat_src_npy.exists():
        feat_dst = subject_dir / sequence_name / "features.npy"
        if not feat_dst.exists():
            shutil.move(str(feat_src_npy), str(feat_dst))
    if feat_src_csv.exists():
        feat_dst = subject_dir / sequence_name / "features.csv"
        if not feat_dst.exists():
            shutil.move(str(feat_src_csv), str(feat_dst))


def write_label_into_sequence(subject_dir: Path, sequence_dir: Path) -> None:
    # subject-level label file like label_570.json
    candidates = list(subject_dir.glob("label_*.json"))
    if not candidates:
        return
    label_path = candidates[0]
    try:
        content = json.loads(label_path.read_text(encoding="utf-8"))
        (sequence_dir / "label.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def write_meta(sequence_dir: Path, frames_dirname: str = "frames") -> None:
    frames_dir = sequence_dir / frames_dirname
    num_frames = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))
    meta = {
        "sequence": sequence_dir.name,
        "num_frames": num_frames,
    }
    (sequence_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_subject_labels(subject_dir: Path) -> None:
    for p in subject_dir.glob("label_*.json"):
        try:
            p.unlink()
            print(f"Deleted subject label: {p}")
        except Exception:
            pass


def reorganize(base_dir: str, subjects: list[str] | None = None) -> None:
    base = Path(base_dir)
    target_subjects = subjects or [d.name for d in base.iterdir() if d.is_dir()]

    for subject in sorted(target_subjects):
        subject_dir = base / subject
        if not subject_dir.exists():
            continue

        for seq_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            moved = move_frames_into_subdir(seq_dir)
            move_outputs(subject_dir, seq_dir.name)
            write_label_into_sequence(subject_dir, seq_dir)
            write_meta(seq_dir)
            print(f"{subject}/{seq_dir.name}: frames moved {moved}")

        # 모든 시퀀스에 복사가 끝났으므로 상위 라벨 파일 삭제
        delete_subject_labels(subject_dir)


if __name__ == "__main__":
    reorganize(str(Path("data/leg_raise")), subjects=None)


