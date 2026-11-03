import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from metrics import dice, iou, hd95


def test_identical_masks():
    m = np.zeros((32, 32), bool); m[8:20, 8:20] = True
    assert abs(dice(m, m) - 1) < 1e-6 and abs(iou(m, m) - 1) < 1e-6
    assert hd95(m, m) == 0


def test_disjoint_masks():
    a = np.zeros((32, 32), bool); a[:8, :8] = True
    b = np.zeros((32, 32), bool); b[20:, 20:] = True
    assert dice(a, b) < 1e-5 and iou(a, b) < 1e-5


def test_hd95_shift():
    a = np.zeros((64, 64), bool); a[20:40, 20:40] = True
    b = np.roll(a, 3, axis=1)
    assert 2 <= hd95(a, b) <= 4
