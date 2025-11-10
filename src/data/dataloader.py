import json
from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .rule_based_feature import extract_pushup_features, extract_plank_features, extract_crunch_features, extract_cross_lunge_features

def read_yaml_splits(yaml_path: str) -> Tuple[List[str], List[str]]:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    lines = p.read_text(encoding="utf-8").splitlines()
    cur = None
    train_list: List[str] = []
    val_list: List[str] = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("train:"):
            cur = "train"
            continue
        if line.startswith("valid:") or line.startswith("val:"):
            cur = "valid"
            continue
        if line.strip().startswith("- "):
            item = line.strip()[2:]
            if cur == "train":
                train_list.append(item)
            elif cur == "valid":
                val_list.append(item)
    return train_list, val_list


class PoseDataset(Dataset):
    def __init__(self, base_dir: str, items: List[str], feature_fn: Callable):
        self.samples = []  # list of (skel_path, label_path)
        self.feature_fn = feature_fn
        base = Path(base_dir)
        for rel in items:
            seq_dir = base / rel
            skel = seq_dir / "skel.npy"
            label = seq_dir / "label.json"
            if skel.exists() and label.exists():
                self.samples.append((skel, label))
        if len(self.samples) == 0:
            raise RuntimeError("No samples found for given split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skel_path, label_path = self.samples[idx]
        pose_seq = np.load(skel_path)  # (T, 33, 3)
        feats = self.feature_fn(pose_seq).astype(np.float32)  # (T, F)

        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        conditions = data.get("type_info", {}).get("conditions", [])
        y = np.array([1.0 if c.get("value", False) else 0.0 for c in conditions], dtype=np.float32)

        X = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y


def collate_pad(batch):
    Xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in Xs]
    T = max(lengths)
    F = Xs[0].shape[1]
    Xpad = torch.zeros(len(Xs), T, F, dtype=torch.float32)
    for i, x in enumerate(Xs):
        Xpad[i, : x.shape[0]] = x
    Y = torch.stack(ys, dim=0)
    return Xpad, Y


class ExerciseDataModule(pl.LightningDataModule):
    def __init__(self, base_dir: str, yaml_path: str, feature_fn: Callable, batch_size: int = 8):
        super().__init__()
        self.base_dir = base_dir
        self.yaml_path = yaml_path
        self.feature_fn = feature_fn
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_items, val_items = read_yaml_splits(self.yaml_path)
        self.train_ds = PoseDataset(self.base_dir, train_items, self.feature_fn)
        self.val_ds = PoseDataset(self.base_dir, val_items, self.feature_fn)
        # discover dims
        X0, y0 = self.train_ds[0]
        self.input_dim = X0.shape[1]
        self.num_classes = y0.shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_pad)


