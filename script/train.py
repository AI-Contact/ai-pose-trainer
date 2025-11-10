import autorootcwd
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataloader import ExerciseDataModule
from src.data.rule_based_feature import extract_pushup_features, extract_plank_features, extract_crunch_features, extract_cross_lunge_features
from src.model.models import ExerciseModel


def get_feature_fn(exercise_type: str):
    """운동 종류에 따라 feature extraction 함수 반환"""
    feature_map = {
        "push_up": extract_pushup_features,
        "pushup": extract_pushup_features,
        "plank": extract_plank_features,
        "crunch": extract_crunch_features,
        "cross_lunge": extract_cross_lunge_features,
    }
    if exercise_type.lower() not in feature_map:
        raise ValueError(f"Unknown exercise type: {exercise_type}. Available: {list(feature_map.keys())}")
    return feature_map[exercise_type.lower()]


@click.command()
@click.option("--exercise", default="push_up", type=click.Choice(["push_up", "plank", "crunch", "cross_lunge"]), help="운동 종류")
@click.option("--base-dir", default=None, type=click.Path(path_type=Path), help="데이터 기본 디렉토리 (기본값: data/{exercise})")
@click.option("--yaml", "yaml_path", default=None, type=click.Path(path_type=Path), help="분할 YAML 파일 (기본값: data/{exercise}/data_split.yaml)")
@click.option("--batch-size", default=8, type=int)
@click.option("--max-epochs", default=100, type=int)
def main(exercise: str, base_dir: Path, yaml_path: Path, batch_size: int, max_epochs: int):
    pl.seed_everything(42, workers=True)

    # 기본 경로 설정
    if base_dir is None:
        base_dir = Path(f"data/{exercise}")
    if yaml_path is None:
        yaml_path = Path(f"data/{exercise}/data_split.yaml")

    # Feature 함수 선택
    feature_fn = get_feature_fn(exercise)

    dm = ExerciseDataModule(
        base_dir=str(base_dir),
        yaml_path=str(yaml_path),
        feature_fn=feature_fn,
        batch_size=batch_size
    )
    dm.setup()

    ckpt_dir = Path("result") / exercise / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{exercise}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    model = ExerciseModel(input_dim=dm.input_dim, num_classes=dm.num_classes, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_cb, es_cb],
        check_val_every_n_epoch=10,
    )

    trainer.fit(model, dm)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()


