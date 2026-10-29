#!/usr/bin/env python3
"""Evaluate released (or newly trained) MPI-CTMixer weights on the held-out test pixels of a benchmark.

The whole scene is predicted with K x K output tiles; every test pixel is scored exactly once. Two schemes are
reported (paper Table 3 uses the overlapping one): stride 9 = non-overlapping inference, stride 5 = overlapping
inference (class probabilities of overlapping tiles are summed).

  python scripts/evaluate.py --dataset indian_pines
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi_ctmixer import data as D  # noqa: E402
from mpi_ctmixer import engine as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--weights", default=None, help="default: weights/mpi_ctmixer_k9_<dataset>_seed0.pth")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--strides", default="9,5")
    ap.add_argument("--data_root", default=None, help="folder with <dataset>.npz (default: data/benchmarks)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()

    w = a.weights or os.path.join(D.ROOT, "weights", f"mpi_ctmixer_k9_{a.dataset}_seed0.pth")
    model, ck = E.load_checkpoint(w, a.device)
    data, gt = D.load_scene(a.dataset, a.data_root)
    assert data.shape[2] == ck["channels"], (data.shape, ck["channels"])
    ncls = int(ck["num_classes"])
    test_rc = D.load_split(a.dataset, a.split_seed)["test"]
    data_norm = D.standardize_per_scene(data)
    k = int(ck["center_size"])

    out = {"dataset": a.dataset, "weights": os.path.basename(w), "center_size": k, "n_test_pixels": int(len(test_rc)),
           "device": a.device, "schemes": {}}
    for s in [int(x) for x in a.strides.split(",")]:
        t0 = time.time()
        pred, n_fwd = E.predict_scene(model, data_norm, k, s, a.device, a.batch_size)
        cm = E.evaluate_test_pixels(pred, gt, test_rc, ncls)
        oa, aa, kappa = E.metrics_from_cm(cm)
        out["schemes"][f"stride{s}"] = {"oa": oa, "aa": aa, "kappa": kappa, "forward_patches": n_fwd,
                                        "seconds": round(time.time() - t0, 2)}
        print(f"{a.dataset}  K={k} stride={s}  OA={oa:.5f}  AA={aa:.5f}  Kappa={kappa:.5f}  "
              f"({n_fwd} forward patches, {time.time() - t0:.1f}s)")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
