"""Benchmark scene loading, per-scene standardisation and the split-safe K x K patch dataset."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (scene file under data/benchmarks, dataset tag used in checkpoint names)
DATASETS = {
    "indian_pines": "IndianPines",
    "pavia_university": "PaviaU",
    "yu_paint": "YUPaint",
}
PATCH = 11


def scene_path(dataset, root=None):
    return os.path.join(root or os.path.join(ROOT, "data", "benchmarks"), f"{dataset}.npz")


def split_path(dataset, seed=0, root=None):
    return os.path.join(root or os.path.join(ROOT, "data", "splits"), f"{dataset}_seed{seed}_split_10_10_80.npz")


def load_scene(dataset, root=None):
    """Return (data float32 HxWxC, gt int32 HxW; 0 = unlabeled, classes 1..N)."""
    path = scene_path(dataset, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_benchmarks.py` (Indian Pines / Pavia University must be "
            f"downloaded from their original sources, see data/benchmarks/README.md).")
    z = np.load(path)
    return z["data"].astype(np.float32), z["gt"].astype(np.int32)


def standardize_per_scene(x):
    """Channel-wise (x - mean) / std over all pixels of the scene (population std, eps clip)."""
    c = x.shape[2]
    x2 = x.reshape(-1, c).astype(np.float64)
    mean = x2.mean(axis=0).astype(np.float32).reshape(1, 1, c)
    std = np.sqrt(np.clip(x2.var(axis=0), 1e-12, None)).astype(np.float32).reshape(1, 1, c)
    return (x.astype(np.float32) - mean) / std


def pad_symmetric(x, pad=PATCH // 2):
    return np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")


def load_split(dataset, seed=0, root=None):
    s = np.load(split_path(dataset, seed, root), allow_pickle=True)
    return {k: np.asarray(s[f"{k}_indices"], dtype=np.int64) for k in ("train", "val", "test")}


def build_split_id_map(h, w, rc):
    m = np.zeros((h, w), dtype=np.int8)
    for i, k in enumerate(("train", "val", "test"), start=1):
        m[rc[k][:, 0], rc[k][:, 1]] = i
    return m


class PatchDataset(Dataset):
    """11 x 11 patches centred on `centers`; the target is the central K x K block, where only pixels that are
    labeled AND belong to this split contribute (everything else is -100 = ignored), so no label leaks across splits."""

    def __init__(self, data_pad, gt, split_id_map, centers, center_size=1, own_split_id=1):
        self.data_pad = data_pad
        self.pad = PATCH // 2
        self.k = int(center_size)
        rad = self.k // 2
        self.centers = centers
        self.own = int(own_split_id)
        self.gt_pad = np.pad(gt, ((rad, rad), (rad, rad)), mode="constant", constant_values=0)
        self.sid_pad = np.pad(split_id_map, ((rad, rad), (rad, rad)), mode="constant", constant_values=0)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, i):
        r, c = self.centers[i]
        p = self.pad
        patch = self.data_pad[r:r + 2 * p + 1, c:c + 2 * p + 1, :]
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32)
        yk = self.gt_pad[r:r + self.k, c:c + self.k]
        sk = self.sid_pad[r:r + self.k, c:c + self.k]
        valid = (yk != 0) & (sk == self.own)
        target = np.where(valid, yk.astype(np.int64) - 1, -100).astype(np.int64)
        return torch.from_numpy(patch), torch.from_numpy(target)
