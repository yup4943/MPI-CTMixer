#!/usr/bin/env python3
"""Train MPI-CTMixer (K = 9, PSR block + dense QKV, sqrt-inverse class-weighted CE) on a benchmark scene.

Recipe used in the paper: Adam, lr 5e-4, weight decay 1e-4, cosine schedule over all epochs, batch 512, 500 epochs,
11 x 11 patches, 10 % / 10 % / 80 % train / val / test pixel split, checkpoint = epoch with the best validation OA.
Training on a GPU is not bit-wise deterministic, so re-training reproduces the reported accuracy within seed noise
(see the five-seed means in the paper), not to the last digit.

  python scripts/train.py --dataset indian_pines --seed 0 --out_dir runs/ip_seed0
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi_ctmixer import data as D  # noqa: E402
from mpi_ctmixer import engine as E  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device, ncls, criterion):
    model.eval()
    cm = torch.zeros((ncls, ncls), dtype=torch.int64, device=device)
    loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device).unsqueeze(1), yb.to(device)
        logits = model(xb, return_logits=True)
        loss += criterion(logits, yb).item()
        E.update_confusion_matrix_(cm, logits.argmax(1), yb, ncls)
    return loss / max(1, len(loader)), E.metrics_from_cm(cm.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--seed", type=int, default=0, help="training seed AND split file index (as in the paper's runs)")
    ap.add_argument("--center_size", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--ce_weight", choices=["sqrt_inverse", "none"], default="sqrt_inverse")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_batches", type=int, default=0, help="debug: stop each epoch after N batches")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="runs/train")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)

    data, gt = D.load_scene(a.dataset, a.data_root)
    h, w, c = data.shape
    ncls = int(gt.max())
    data_pad = D.pad_symmetric(D.standardize_per_scene(data))
    rc = D.load_split(a.dataset, a.seed)
    sid = D.build_split_id_map(h, w, rc)

    def loader(name, sid_val, shuffle):
        ds = D.PatchDataset(data_pad, gt, sid, [tuple(x) for x in rc[name]], a.center_size, sid_val)
        return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers,
                          pin_memory=device.type == "cuda")

    train_loader, val_loader = loader("train", 1, True), loader("val", 2, False)

    hist = np.bincount(gt[rc["train"][:, 0], rc["train"][:, 1]].astype(np.int64) - 1, minlength=ncls)
    weight = None
    if a.ce_weight == "sqrt_inverse":
        weight = torch.tensor(E.sqrt_inverse_class_weights(hist), device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)

    model = E.build_model(c, ncls, a.center_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=a.epochs)

    best_oa, best_epoch = -1.0, 0
    best_path = os.path.join(a.out_dir, f"mpi_ctmixer_k{a.center_size}_{a.dataset}_seed{a.seed}.pth")
    log = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True).unsqueeze(1), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(xb, return_logits=True), yb)
            loss.backward()
            opt.step()
            loss_sum, nb = loss_sum + loss.item(), nb + 1
            if a.max_train_batches and nb >= a.max_train_batches:
                break
        sched.step()
        va_loss, (va_oa, va_aa, va_kappa) = run_eval(model, val_loader, device, ncls, criterion)
        print(f"[{a.dataset} seed{a.seed}] epoch {epoch}/{a.epochs} train_loss={loss_sum / max(1, nb):.4f} "
              f"val_oa={va_oa:.4f} val_aa={va_aa:.4f} val_kappa={va_kappa:.4f} ({time.time() - t0:.1f}s)", flush=True)
        log.append(dict(epoch=epoch, train_loss=loss_sum / max(1, nb), val_oa=va_oa, val_aa=va_aa, val_kappa=va_kappa))
        if va_oa > best_oa:
            best_oa, best_epoch = va_oa, epoch
            torch.save({"state_dict": model.state_dict(), "dataset": a.dataset, "channels": c, "num_classes": ncls,
                        "center_size": a.center_size, "seed": a.seed, "best_epoch": epoch, "best_val_oa": va_oa,
                        "train_args": {k: getattr(a, k) for k in ("epochs", "lr", "weight_decay", "batch_size")}
                        | {"ce_weight_mode": a.ce_weight}}, best_path)
    json.dump(log, open(os.path.join(a.out_dir, "train_log.json"), "w"))
    print(f"best val OA {best_oa:.5f} at epoch {best_epoch} -> {best_path}")
    print(f"evaluate with: python scripts/evaluate.py --dataset {a.dataset} --weights {best_path} --split_seed {a.seed}")


if __name__ == "__main__":
    main()
