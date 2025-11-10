import autorootcwd
from pathlib import Path
import json

import click
import numpy as np

from rule_based_feature import extract_pushup_features


def is_empty_skeleton(npy_path: Path) -> bool:
    try:
        arr = np.load(npy_path)
        return arr.ndim < 2 or arr.shape[0] == 0
    except Exception:
        return True


@click.command()
@click.option("--base-dir", default=str(Path("data/push_up")), type=click.Path(path_type=Path))
@click.option("--save-csv", is_flag=True, default=True, help="결과를 CSV로 저장")
def main(base_dir: Path, save_csv: bool):
    base = base_dir
    bad = []  # (subject/sequence, reason)

    for subject_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for seq_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            rel = f"{subject_dir.name}/{seq_dir.name}"
            skel = seq_dir / "skel.npy"
            label = seq_dir / "label.json"

            if not skel.exists():
                bad.append((rel, "missing_skel"))
                continue

            if is_empty_skeleton(skel):
                bad.append((rel, "empty_skel"))
                continue

            try:
                feats = extract_pushup_features(np.load(skel))
                if feats.ndim != 2 or feats.shape[0] == 0:
                    bad.append((rel, "empty_features"))
            except Exception:
                bad.append((rel, "feature_error"))

            if not label.exists():
                bad.append((rel, "missing_label"))

    if not bad:
        print("No empty or invalid sequences found.")
    else:
        for rel, reason in bad:
            print(f"{rel} -> {reason}")

        if save_csv:
            out = base / "empty_sequences.csv"
            with out.open("w", encoding="utf-8") as f:
                f.write("sequence,reason\n")
                for rel, reason in bad:
                    f.write(f"{rel},{reason}\n")
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()


