import autorootcwd
import random
import click
from pathlib import Path

def make_splits(pushup_dir: str, train_ratio: float = 0.9, seed: int = 42):
    root = Path(pushup_dir)
    rng = random.Random(seed)

    all_train = []
    all_val = []    

    for subject_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        seqs = [p.name for p in sorted(subject_dir.iterdir()) if p.is_dir()]
        rng.shuffle(seqs)
        n_train = max(1, int(len(seqs) * train_ratio)) if len(seqs) > 1 else len(seqs)
        train_seqs = seqs[:n_train]
        val_seqs = seqs[n_train:]

        all_train += [f"{subject_dir.name}/{s}" for s in train_seqs]
        all_val += [f"{subject_dir.name}/{s}" for s in val_seqs]

        print(f"{subject_dir.name}: train {len(train_seqs)} / val {len(val_seqs)} (total {len(seqs)})")

    splits_dir = root
    splits_dir.mkdir(exist_ok=True)

    print(f"Saved: {splits_dir / 'train.txt'} ({len(all_train)})")
    print(f"Saved: {splits_dir / 'val.txt'} ({len(all_val)})")

    yaml_lines = [
        "train:",
    ]
    yaml_lines += [f"  - {item}" for item in all_train]
    yaml_lines += [
        "valid:",
    ]
    yaml_lines += [f"  - {item}" for item in all_val]

    (splits_dir / "data_split.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Saved: {splits_dir / 'data_split.yaml'}")

@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["cross_lunge", "crunch", "plank", "push_up"]), help="운동 종류")
@click.option("--base-dir", default=None, type=click.Path(path_type=Path), help="데이터 기본 디렉토리 (기본값: data/{exercise})")
@click.option("--train-ratio", default=0.9, type=float, help="Train 데이터 비율")
@click.option("--seed", default=42, type=int, help="랜덤 시드")
def main(exercise: str, base_dir: Path, train_ratio: float, seed: int):

    if base_dir is None:
        base_dir = Path(f"data/{exercise}")
    
    make_splits(
        str(base_dir),
        train_ratio=train_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    main()

