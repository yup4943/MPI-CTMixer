"""CPU smoke test that needs no dataset: model shapes, strict weight loading, and overlap-inference bookkeeping."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi_ctmixer import engine as E  # noqa: E402


def test_forward_shape_and_probabilities():
    m = E.build_model(channels=30, num_classes=5, center_size=9).eval()
    out = m(torch.randn(2, 1, 30, 11, 11))
    assert out.shape == (2, 5, 9, 9)
    assert torch.allclose(out.sum(1), torch.ones(2, 9, 9), atol=1e-5)


def test_released_weights_load_strictly():
    wdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    for f in sorted(os.listdir(wdir)):
        if f.endswith(".pth"):
            model, ck = E.load_checkpoint(os.path.join(wdir, f))
            assert ck["center_size"] == 9 and sum(p.numel() for p in model.parameters()) > 2_000_000


def test_overlap_prediction_covers_every_pixel():
    torch.manual_seed(0)
    model = E.build_model(channels=12, num_classes=4, center_size=9).eval()
    scene = np.random.RandomState(0).randn(23, 31, 12).astype(np.float32)
    for stride in (9, 5):
        pred, n = E.predict_scene(model, scene, 9, stride, "cpu")
        assert pred.shape == (23, 31) and pred.min() >= 0 and pred.max() < 4 and n > 0
    # overlapping inference needs more forward patches than block filling
    assert E.predict_scene(model, scene, 9, 5, "cpu")[1] > E.predict_scene(model, scene, 9, 9, "cpu")[1]


def test_metrics():
    cm = np.array([[8, 2], [1, 9]])
    oa, aa, kappa = E.metrics_from_cm(cm)
    assert abs(oa - 17 / 20) < 1e-9 and abs(aa - 0.85) < 1e-9 and 0.69 < kappa < 0.71
