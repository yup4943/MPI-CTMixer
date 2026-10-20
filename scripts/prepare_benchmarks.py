#!/usr/bin/env python3
"""Convert the benchmark scenes into the npz format used by this repository (data/benchmarks/<name>.npz with arrays
`data` (H,W,C uint16) and `gt` (H,W uint8; 0 = unlabeled)).

Indian Pines and Pavia University are NOT redistributed here; download them from their original source
(https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and pass the .mat files:

  python scripts/prepare_benchmarks.py \
      --ip_data Indian_pines_corrected.mat --ip_gt Indian_pines_gt.mat \
      --pu_data PaviaU.mat --pu_gt PaviaU_gt.mat \
      --yu_data YU_Paint/data2.mat --yu_gt YU_Paint/gt2.mat

Indian Pines is stored zero-padded by 2 pixels on every side (145x145 -> 149x149, data = 0, gt = 0); the released
train/val/test split files index this 149x149 layout and the per-scene standardisation includes the zero border.
Any subset of the datasets can be converted. Arrays are picked by shape, so the .mat variable names do not matter.
"""
import argparse
import os

import numpy as np
import scipy.io as sio

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmarks")


def pick(path, ndim):
    cands = {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__") and getattr(v, "ndim", 0) == ndim}
    if len(cands) != 1:
        raise ValueError(f"{path}: expected exactly one {ndim}-D array, found {list(cands)}")
    return next(iter(cands.values()))


def save(name, data, gt, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.npz")
    np.savez_compressed(path, data=data.astype(np.uint16), gt=gt.astype(np.uint8))
    print(f"[ok] {path}  data{data.shape}  labeled={int((gt > 0).sum())}  classes={int(gt.max())}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("ip_data", "ip_gt", "pu_data", "pu_gt", "yu_data", "yu_gt"):
        ap.add_argument("--" + k)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.ip_data and a.ip_gt:
        d, g = pick(a.ip_data, 3), pick(a.ip_gt, 2)
        if d.shape[:2] == (145, 145):
            d = np.pad(d, ((2, 2), (2, 2), (0, 0)))
            g = np.pad(g, ((2, 2), (2, 2)))
        assert d.shape == (149, 149, 200) and g.shape == (149, 149), (d.shape, g.shape)
        save("indian_pines", d, g, a.out)
    if a.pu_data and a.pu_gt:
        d, g = pick(a.pu_data, 3), pick(a.pu_gt, 2)
        assert d.shape == (610, 340, 103), d.shape
        save("pavia_university", d, g, a.out)
    if a.yu_data and a.yu_gt:
        d, g = pick(a.yu_data, 3), pick(a.yu_gt, 2)
        assert d.shape == (343, 688, 258), d.shape
        save("yu_paint", d, g, a.out)


if __name__ == "__main__":
    main()
