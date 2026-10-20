#!/usr/bin/env python3
"""Smoke test on the released ConcealHSI sample patches (two 11 x 11 x 150 patches per class, 13 classes).

The full ConcealHSI dataset (279 scenes) is not public. These 26 patches show the exact input format of the model
(channel-standardised reflectance, 150 bands, 11 x 11 pixels) and let you check the whole pipeline end to end:
  1. a forward pass with the K = 9 multi-pixel head (output: class probabilities for the central 9 x 9 block),
  2. the training step on the 9 x 9 central label block (a few optimisation steps on the 26 patches, which only
     demonstrates that the code runs and the loss decreases; it is NOT a benchmark),
  3. a per-class mean-spectrum figure.

  python scripts/demo_concealhsi_samples.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi_ctmixer import data as D  # noqa: E402
from mpi_ctmixer import engine as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(D.ROOT, "data", "concealhsi_samples", "concealhsi_samples.npz"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out_dir", default="outputs/concealhsi_demo")
    a = ap.parse_args()
    torch.manual_seed(0)

    z = np.load(a.samples)
    x = torch.from_numpy(z["patches"]).permute(0, 3, 1, 2).unsqueeze(1)          # (N, 1, 150, 11, 11)
    names = [str(n) for n in z["class_names"]]
    n, _, c, _, _ = x.shape
    k = 9
    y = torch.from_numpy(z["label_window"][:, 1:1 + k, 1:1 + k].astype(np.int64))  # central 9 x 9 block, -1 = unlabeled
    y = torch.where(y < 0, torch.full_like(y, -100), y)
    print(f"{n} patches, {c} bands, {len(names)} classes: {names}")

    model = E.build_model(c, len(names), k)
    model.eval()
    with torch.no_grad():
        prob = model(x)
    print("forward:", tuple(x.shape), "->", tuple(prob.shape), "(patches, classes, K, K); probabilities sum to",
          float(prob.sum(1).mean()))

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss = crit(model(x, return_logits=True), y)
        loss.backward()
        opt.step()
        if step in (1, a.steps // 2, a.steps):
            print(f"train step {step:3d}  loss {loss.item():.4f}")

    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cls = z["center_class"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ci, name in enumerate(names):
        sel = np.nonzero(cls == ci)[0]
        if len(sel):
            ax.plot(z["patches"][sel].reshape(len(sel), -1, c).mean(1).mean(0), label=name, lw=1)
    ax.set_xlabel("band index (150 bands, 470-900 nm)"); ax.set_ylabel("standardised value")
    ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "mean_spectra.png"), dpi=150)
    print("wrote", os.path.join(a.out_dir, "mean_spectra.png"))


if __name__ == "__main__":
    main()
