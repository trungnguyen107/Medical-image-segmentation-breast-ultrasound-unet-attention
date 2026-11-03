"""Các độ đo phân đoạn (Dice, IoU, Precision, Recall, Specificity, HD95) và chỉ số hình thái."""
import numpy as np
from scipy import ndimage as ndi

EPS = 1e-7


def confusion(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.sum(p & g); fp = np.sum(p & ~g); fn = np.sum(~p & g); tn = np.sum(~p & ~g)
    return tp, fp, fn, tn


def dice(pred, gt):
    tp, fp, fn, _ = confusion(pred, gt)
    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def iou(pred, gt):
    tp, fp, fn, _ = confusion(pred, gt)
    return (tp + EPS) / (tp + fp + fn + EPS)


def precision(pred, gt):
    tp, fp, _, _ = confusion(pred, gt)
    return (tp + EPS) / (tp + fp + EPS)


def recall(pred, gt):
    tp, _, fn, _ = confusion(pred, gt)
    return (tp + EPS) / (tp + fn + EPS)


def specificity(pred, gt):
    _, fp, _, tn = confusion(pred, gt)
    return (tn + EPS) / (tn + fp + EPS)


def _surface(m):
    m = m.astype(bool)
    return m & ~ndi.binary_erosion(m)


def hd95(pred, gt):
    """Khoảng cách Hausdorff phân vị 95 (điểm ảnh). Trả về NaN nếu một trong hai mặt nạ rỗng."""
    p, g = pred.astype(bool), gt.astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    dg = ndi.distance_transform_edt(~sg)
    dp = ndi.distance_transform_edt(~sp)
    d = np.concatenate([dg[sp], dp[sg]])
    return float(np.percentile(d, 95))


def all_metrics(pred, gt):
    return dict(dice=dice(pred, gt), iou=iou(pred, gt), precision=precision(pred, gt),
                recall=recall(pred, gt), specificity=specificity(pred, gt), hd95=hd95(pred, gt))


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def perimeter(mask):
    """Chu vi đường bao ngoài (điểm ảnh), tính trên đường viền đa giác của OpenCV."""
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in cnts))


def shape_features(mask, px_mm=None):
    """Diện tích, đường kính lớn nhất, độ tròn 4πA/P², tỉ lệ trục và độ đặc (solidity) của vùng dự đoán.

    px_mm: kích thước một điểm ảnh (mm). Nếu có, trả thêm diện tích (mm²) và đường kính (mm).
    """
    m = largest_component(mask) if mask.any() else mask.astype(bool)
    area = float(m.sum())
    out = dict(area_px=area, area_ratio=0.0, max_diameter_px=0.0, circularity=0.0, aspect_ratio=0.0, solidity=0.0)
    if area == 0:
        return out
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(float)
    from scipy.spatial import ConvexHull
    try:
        hv = ConvexHull(pts)
        hull = pts[hv.vertices]; hull_area = float(hv.volume)
    except Exception:
        hull = pts; hull_area = area
    dmax = float(np.max(np.linalg.norm(hull[:, None] - hull[None], axis=-1))) + 1.0
    per = perimeter(m)
    circ = float(min(1.0, 4 * np.pi * area / max(per, 1.0) ** 2))
    cov = np.cov((pts - pts.mean(0)).T) if len(pts) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))
    ar = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    out.update(area_ratio=area / m.size, max_diameter_px=dmax, circularity=circ, aspect_ratio=ar,
               solidity=float(min(1.0, area / max(hull_area, 1.0))), perimeter_px=per)
    if px_mm:
        out.update(area_mm2=area * px_mm ** 2, max_diameter_mm=dmax * px_mm, perimeter_mm=per * px_mm)
    return out
