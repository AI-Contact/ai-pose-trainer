import autorootcwd
from pathlib import Path
import json

import click


@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["cross_lunge", "crunch", "plank", "push_up", ]), help="운동 종류")
@click.option("--base-dir", default=None, type=click.Path(path_type=Path), help="데이터 기본 디렉토리 (기본값: data/{exercise})")
@click.option("--out-csv", default=None, type=click.Path(path_type=Path), help="출력 CSV 파일 (기본값: data/{exercise}/subject_labels.csv)")
def main(exercise: str, base_dir: Path, out_csv: Path):
    # 기본 경로 설정
    if base_dir is None:
        base_dir = Path(f"data/{exercise}")
    if out_csv is None:
        out_csv = Path(f"data/{exercise}/subject_labels.csv")
    
    base = base_dir
    rows = []
    header = None

    for subject_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        # subject 폴더 이름이 숫자인지 확인 (561, 562, ...)
        if not subject_dir.name.isdigit():
            continue

        # 첫 번째 시퀀스 폴더 찾기
        seq_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir()])
        if not seq_dirs:
            continue

        first_seq = seq_dirs[0]
        label_json = first_seq / "label.json"
        if not label_json.exists():
            continue

        try:
            data = json.loads(label_json.read_text(encoding="utf-8"))
            conditions = data.get("type_info", {}).get("conditions", [])
            values = [str(c.get("value", False)).lower() for c in conditions]

            if header is None:
                header = ["subject"] + [c.get("condition", f"cond{i}") for i, c in enumerate(conditions)]

            rows.append([subject_dir.name] + values)

        except Exception as e:
            print(f"Failed to read {label_json}: {e}")

    if not rows:
        print("No subject labels found.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"Saved: {out_csv} ({len(rows)} subjects)")


if __name__ == "__main__":
    main()

