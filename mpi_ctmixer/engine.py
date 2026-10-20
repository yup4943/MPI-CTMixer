"""Metrics, class weights, model factory, training epoch loops and overlapping multi-pixel inference."""
import numpy as np
import torch
import torch.nn as nn

from .model import MPICTMixer

# Final MPI-CTMixer configuration used for every number in the paper (K = 9).
MODEL_KWARGS = dict(num_layers=1, num_heads=4, patch_size=1, emb_dim=128, head_dim=64, hidden_dim=64,
                    new_bands=160, reduce_bands=48, plainconv_width_mult=6.35)
IMAGE_SIZE = 11


def build_model(channels, num_classes, center_size=9):
    return MPICTMixer(channels=channels, num_classes=num_classes, image_size=IMAGE_SIZE,
                      center_size=center_size, **MODEL_KWARGS)


def load_checkpoint(path, device="cpu"):
    """Load a released weights file ({'state_dict', 'channels', 'num_classes', 'center_size', ...})."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["channels"], ck["num_classes"], ck["center_size"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, ck


# ---------------------------------------------------------------------------------------------- metrics
def update_confusion_matrix_(cm, preds, target, num_classes):
    mask = target != -100
    t, p = target[mask], preds[mask]
    if t.numel() == 0:
        return
    flat = t.to(torch.int64) * int(num_classes) + p.to(torch.int64)
    cm += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_cm(cm):
    """OA, AA (over classes present in the reference), Cohen's kappa."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    row, col, diag = cm.sum(1), cm.sum(0), np.diag(cm)
    oa = diag.sum() / total
    valid = row > 0
    aa = (diag[valid] / row[valid]).mean()
    pe = (row * col).sum() / (total * total)
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0
    return float(oa), float(aa), float(kappa)


def sqrt_inverse_class_weights(class_hist, eps=1.0):
    """w_c proportional to 1 / sqrt(n_c + eps), normalised to mean 1 (the paper's class-weighted CE)."""
    w = 1.0 / np.sqrt(np.asarray(class_hist, dtype=np.float64) + eps)
    return (w / w[np.isfinite(w) & (w > 0)].mean()).astype(np.float32)


# ---------------------------------------------------------------------------------------------- inference
def axis_centers(n, rad, stride):
    c = list(range(rad, n - rad, stride)) or [min(max(rad, n // 2), max(n - 1 - rad, 0))]
    if c[-1] != n - 1 - rad and n - 1 - rad >= rad:
        c.append(n - 1 - rad)
    return np.array(sorted(set(c)), dtype=np.int64)


@torch.no_grad()
def predict_scene(model, data_norm, center_size, stride, device, batch_size=1024):
    """Full-scene label map. KxK output tiles are placed on a grid with `stride` (stride=K: non-overlapping block
    filling; stride<K: overlapping inference). The class probabilities of all tiles covering a pixel are summed and the
    argmax is taken. Returns (label map HxW int64, number of forward patches)."""
    h, w, c = data_norm.shape
    k, rad = int(center_size), int(center_size) // 2
    pad = IMAGE_SIZE // 2
    x = torch.from_numpy(np.pad(data_norm, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")).to(device)
    x = x.permute(2, 0, 1).contiguous()                       # (C, H+10, W+10)
    unf = x.unfold(1, IMAGE_SIZE, 1).unfold(2, IMAGE_SIZE, 1)  # (C, H, W, 11, 11) view
    cy, cx = np.meshgrid(axis_centers(h, rad, stride), axis_centers(w, rad, stride), indexing="ij")
    by = torch.from_numpy(cy.ravel()).to(device)
    bx = torch.from_numpy(cx.ravel()).to(device)
    ncls = None
    acc = None
    for s in range(0, len(by), batch_size):
        yy, xx = by[s:s + batch_size], bx[s:s + batch_size]
        prob = model(unf[:, yy, xx].permute(1, 0, 2, 3).unsqueeze(1))      # (B, ncls, K, K)
        if acc is None:
            ncls = prob.shape[1]
            acc = torch.zeros(ncls, h, w, device=device)
            cls = torch.arange(ncls, device=device)[:, None]
        for dy in range(k):
            for dx in range(k):
                acc.index_put_((cls, (yy + dy - rad)[None], (xx + dx - rad)[None]), prob[:, :, dy, dx].T, accumulate=True)
    return acc.argmax(0).cpu().numpy(), len(by)


def evaluate_test_pixels(pred_map, gt, test_rc, num_classes):
    """Confusion matrix of a full-scene map on held-out test pixels, each pixel scored exactly once."""
    rows, cols = test_rc[:, 0], test_rc[:, 1]
    lab, pred = gt[rows, cols] - 1, pred_map[rows, cols]
    cm = np.zeros((num_classes, num_classes), np.int64)
    np.add.at(cm, (lab, pred), 1)
    return cm
