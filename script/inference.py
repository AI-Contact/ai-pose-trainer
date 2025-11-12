import autorootcwd
from pathlib import Path
import json

import click
import numpy as np
import torch

from src.data.dataloader import read_yaml_splits
from src.data.rule_based_feature import extract_pushup_features, extract_plank_features, extract_crunch_features, extract_cross_lunge_features, extract_leg_raise_features
from src.model.models import ExerciseModel


def get_feature_fn(exercise: str):
    """운동 종류에 따라 feature extraction 함수 반환"""
    feature_map = {
        "push_up": extract_pushup_features,
        "plank": extract_plank_features,
        "crunch": extract_crunch_features,
        "cross_lunge": extract_cross_lunge_features,
        "leg_raise": extract_leg_raise_features,
    }
    if exercise.lower() not in feature_map:
        raise ValueError(f"Unknown exercise type: {exercise}. Available: {list(feature_map.keys())}")
    return feature_map[exercise.lower()]


def infer_sequence(seq_dir: Path, model: ExerciseModel, feature_fn):
    skel = seq_dir / "skel.npy"
    pose_seq = np.load(skel)
    feats = feature_fn(pose_seq).astype(np.float32)
    device = next(model.parameters()).device
    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    names = None
    label_json = seq_dir / "label.json"
    if label_json.exists():
        data = json.loads(label_json.read_text(encoding="utf-8"))
        names = [c.get("condition", f"cond{i}") for i, c in enumerate(data.get("type_info", {}).get("conditions", []))]
    return names, probs


@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["push_up", "plank", "crunch", "cross_lunge", "leg_raise"]), help="운돓 종류")
@click.option("--base-dir", default=None, type=click.Path(path_type=Path), help="데이터 기본 디렉토리 (기본값: data/{exercise})")
@click.option("--yaml", "yaml_path", default=None, type=click.Path(path_type=Path), help="분할 YAML 파일 (기본값: data/{exercise}/data_split.yaml)")
@click.option("--use-split", default="valid", type=click.Choice(["train", "valid"]))
@click.option("--ckpt", required=True, type=click.Path(path_type=Path))
@click.option("--out-csv", default=None, type=click.Path(path_type=Path), help="출력 CSV 파일 (기본값: result/{exercise}/inference_{split}.csv)")
def main(exercise: str, base_dir: Path, yaml_path: Path, use_split: str, ckpt: Path, out_csv: Path):
    # 기본 경로 설정
    if base_dir is None:
        base_dir = Path(f"data/{exercise}")
    if yaml_path is None:
        yaml_path = Path(f"data/{exercise}/data_split.yaml")
    if out_csv is None:
        out_csv = Path(f"result/{exercise}/inference_{use_split}.csv")

    # Feature 함수 선택
    feature_fn = get_feature_fn(exercise)

    train_items, val_items = read_yaml_splits(str(yaml_path))
    items = train_items if use_split == "train" else val_items

    # GPU 호환 이슈 회피를 위해 기본적으로 CPU에 로드
    model = ExerciseModel.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.to("cpu")

    header = None
    rows = []
    for rel in items:
        seq_dir = base_dir / rel
        if not (seq_dir / "skel.npy").exists():
            continue
        names, probs = infer_sequence(seq_dir, model, feature_fn)
        if header is None:
            header = names or [f"cond{i}" for i in range(len(probs))]
        rows.append((rel, probs))
        pretty = ", ".join([f"{h}:{p:.3f}" for h, p in zip(header, probs)])
        print(f"{rel} -> {pretty}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("sequence," + ",".join(header or []) + "\n")
        for rel, probs in rows:
            f.write(rel + "," + ",".join(f"{p:.6f}" for p in probs) + "\n")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()


