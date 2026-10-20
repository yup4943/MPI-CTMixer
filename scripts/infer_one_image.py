#!/usr/bin/env python3
"""Predict the label map of ONE hyperspectral image with MPI-CTMixer (overlapping multi-pixel inference).

Benchmark scene + released weights:
  python scripts/infer_one_image.py --dataset yu_paint --out_dir outputs/yu_paint

Your own cube (H x W x C array in a .npy / .npz['data'] / .mat file; C and the number of classes must match the weights):
  python scripts/infer_one_image.py --cube my_scene.npy --weights my_weights.pth --out_dir outputs/mine

The cube is standardised per channel over the whole image (as in training), tiled into 11 x 11 patches, every patch
predicts a central K x K block (K = 9), tiles are placed with stride S (default 5) and their class probabilities are
summed. Writes <name>_pred.npy (H x W, 0-based class ids) and <name>_pred.png (false-colour prediction; with ground
truth for benchmark scenes also the GT and the accuracy on labeled pixels).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi_ctmixer import data as D  # noqa: E402
from mpi_ctmixer import engine as E  # noqa: E402


def load_cube(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.endswith(".npz"):
        z = np.load(path)
        return z["data"].astype(np.float32)
    import scipy.io as sio
    arrs = [v for k, v in sio.loadmat(path).items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
    if len(arrs) != 1:
        raise ValueError("expected exactly one 3-D array in the .mat file")
    return arrs[0].astype(np.float32)


def colorize(label_map, ncls):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    pal = np.array([cmap(i % 20)[:3] for i in range(ncls)])
    out = np.zeros(label_map.shape + (3,))
    ok = label_map >= 0
    out[ok] = pal[label_map[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=list(D.DATASETS))
    g.add_argument("--cube")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="outputs")
    a = ap.parse_args()

    if a.dataset:
        weights = a.weights or os.path.join(D.ROOT, "weights", f"mpi_ctmixer_k9_{a.dataset}_seed0.pth")
        cube, gt = D.load_scene(a.dataset, a.data_root)
        name = a.dataset
    else:
        if not a.weights:
            ap.error("--weights is required with --cube")
        cube, gt, name = load_cube(a.cube), None, os.path.splitext(os.path.basename(a.cube))[0]
        weights = a.weights
    model, ck = E.load_checkpoint(weights, a.device)
    if cube.shape[2] != ck["channels"]:
        sys.exit(f"cube has {cube.shape[2]} channels but the weights expect {ck['channels']}")
    k, ncls = int(ck["center_size"]), int(ck["num_classes"])

    t0 = time.time()
    pred, n_fwd = E.predict_scene(model, D.standardize_per_scene(cube), k, a.stride, a.device, a.batch_size)
    dt = time.time() - t0
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, f"{name}_pred.npy"), pred)
    msg = f"{name}: {cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}  K={k} stride={a.stride}  {n_fwd} forward patches  {dt:.1f}s"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if gt is not None:
        lab = gt > 0
        acc = float((pred[lab] == gt[lab] - 1).mean())
        msg += f"  accuracy on all labeled pixels (train+val+test) = {acc:.4f}"
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(colorize(np.where(lab, gt - 1, -1), ncls)); ax[0].set_title("ground truth")
        ax[1].imshow(colorize(np.where(lab, pred, -1), ncls)); ax[1].set_title("MPI-CTMixer prediction (labeled pixels)")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(colorize(pred, ncls)); ax.set_title("MPI-CTMixer prediction")
        ax = [ax]
    for x in np.atleast_1d(ax):
        x.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f"{name}_pred.png"), dpi=150)
    print(msg)


if __name__ == "__main__":
    main()
