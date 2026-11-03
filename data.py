"""Dữ liệu: bộ sinh ảnh siêu âm vú mô phỏng và bộ nạp BUSI.

Bộ sinh mô phỏng (SyntheticUS) tái tạo các đặc điểm chính của ảnh siêu âm vú đã nêu ở mục 1.2:
nhiễu đốm nhân (speckle), suy giảm theo độ sâu, khối giảm âm với bờ mờ, bóng cản âm phía sau khối
ác tính, tăng âm phía sau nang lành tính và các vùng tối "gây nhiễu" không phải khối u.
Đây KHÔNG phải ảnh y tế thật: chỉ dùng để kiểm thử quy trình khi chưa có BUSI.
"""
import os, glob
import numpy as np
import cv2
import imio
from scipy import ndimage as ndi

IMG = 96                       # kích thước ảnh dùng cho bộ mô phỏng (huấn luyện trên CPU)
TYPES = ["normal", "benign", "malignant"]
TYPE_P = [0.17, 0.56, 0.27]    # tỉ lệ gần với BUSI (133/437/210 trên 780)


def _speckle(rng, shape, sx, sy):
    a = ndi.gaussian_filter(rng.normal(size=shape), (sy, sx))
    b = ndi.gaussian_filter(rng.normal(size=shape), (sy, sx))
    r = np.sqrt(a ** 2 + b ** 2)
    return r / r.mean()


def make_sample(seed, kind=None, domain="A", size=IMG):
    """Trả về (ảnh uint8 [H,W], mặt nạ uint8 {0,1}, nhãn loại)."""
    rng = np.random.default_rng(seed)
    S = size; k = S / 128.0          # hệ số quy đổi các tham số tính theo điểm ảnh
    if kind is None:
        kind = rng.choice(TYPES, p=TYPE_P)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = dict(A=dict(gain=(0.85, 1.15), contrast=(0.9, 1.1), spx=(0.9, 1.3), spy=(1.2, 1.8), les=(0.22, 0.42), att=(1.2, 2.0)),
             B=dict(gain=(0.6, 0.9), contrast=(0.7, 0.9), spx=(1.6, 2.2), spy=(2.0, 3.0), les=(0.33, 0.55), att=(2.0, 3.0)))[domain]
    # nền mô: các lớp mô theo độ sâu + biến thiên chậm
    depth = yy / S
    tissue = 0.55 + 0.25 * ndi.gaussian_filter(rng.normal(size=(S, S)), S / 10) / 0.05 * 0.05
    tissue += 0.18 * np.exp(-((depth - rng.uniform(0.08, 0.2)) / 0.05) ** 2)      # lớp mỡ/da
    tissue += 0.12 * np.exp(-((depth - rng.uniform(0.7, 0.9)) / 0.06) ** 2)       # cân ngực
    tissue *= np.exp(-rng.uniform(*d["att"]) * depth * 0.6)                        # suy giảm theo độ sâu
    mask = np.zeros((S, S), np.float32)
    mult = np.ones((S, S), np.float32)
    # vùng tối gây nhiễu: bóng cản dạng cột (xương sườn) và nang nhỏ không phải khối u
    for _ in range(rng.integers(0, 3)):
        cx = rng.uniform(0.1, 0.9) * S; w = rng.uniform(0.04, 0.09) * S
        col = np.exp(-((xx - cx) / w) ** 2) * (yy > rng.uniform(0.3, 0.6) * S)
        mult *= 1 - rng.uniform(0.35, 0.6) * col
    for _ in range(rng.integers(0, 3)):
        cx, cy = rng.uniform(0.1, 0.9) * S, rng.uniform(0.2, 0.8) * S
        ax, ay = rng.uniform(0.02, 0.05) * S, rng.uniform(0.015, 0.04) * S
        blob = np.exp(-(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) ** 1.5)
        mult *= 1 - rng.uniform(0.4, 0.7) * blob
    for _ in range(rng.integers(0, 4)):                                             # điểm tăng âm
        cx, cy = rng.uniform(0.05, 0.95) * S, rng.uniform(0.15, 0.9) * S
        mult += rng.uniform(0.3, 0.7) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(1, 2.5) * k) ** 2)))
    if kind != "normal":
        cx, cy = rng.uniform(0.28, 0.72) * S, rng.uniform(0.32, 0.65) * S
        a = rng.uniform(0.09, 0.22) * S                      # bán trục ngang
        b = a * (rng.uniform(0.45, 0.8) if kind == "malignant" else rng.uniform(0.55, 0.95))
        th = np.arctan2((yy - cy) / b, (xx - cx) / a)
        amp = 0.14 if kind == "malignant" else 0.045
        r = 1 + sum(rng.normal(0, amp / k) * np.cos(k * th + rng.uniform(0, 6.28)) for k in range(2, 7))
        rho = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
        m = (rho < r).astype(np.float32)
        mask = m
        blur = (rng.uniform(1.8, 3.0) if kind == "malignant" else rng.uniform(0.8, 1.6)) * k
        soft = ndi.gaussian_filter(m, blur)
        mult *= 1 - (1 - rng.uniform(*d["les"])) * soft
        below = (yy > cy + b * 0.8) & (np.abs(xx - cx) < a * 0.95)
        colw = ndi.gaussian_filter(below.astype(np.float32), 3.0 * k) * np.clip((yy - cy) / (S * 0.4), 0, 1)
        if kind == "malignant":
            mult *= 1 - rng.uniform(0.15, 0.5) * colw                                # bóng cản phía sau
        else:
            mult *= 1 + rng.uniform(0.1, 0.35) * colw                                # tăng âm phía sau
    sp = _speckle(rng, (S, S), rng.uniform(*d["spx"]) * k, rng.uniform(*d["spy"]) * k)
    img = tissue * mult * sp
    img = (img - img.mean()) * rng.uniform(*d["contrast"]) + img.mean()
    img = img * rng.uniform(*d["gain"])
    img = np.clip(img, 1e-4, None)
    img = np.log1p(rng.uniform(15, 40) * img)                                       # nén log như máy siêu âm B-mode
    img = np.clip(img / (np.percentile(img, 99.7) + 1e-6), 0, 1)
    img = img ** rng.uniform(0.9, 1.3)
    return (img * 255).astype(np.uint8), (mask > 0.5).astype(np.uint8), kind


def build_split(n, seed0, domain="A", size=IMG):
    X = np.zeros((n, size, size), np.uint8); Y = np.zeros_like(X); K = []
    for i in range(n):
        x, y, k = make_sample(seed0 + i, domain=domain, size=size)
        X[i], Y[i] = x, y; K.append(k)
    return X, Y, np.array(K)


def load_synthetic(cache="results/synthetic_cache.npz", n_train=480, n_val=80, n_test=160, n_ext=160):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = {}
    for name, n, s0, dom in [("train", n_train, 1000, "A"), ("val", n_val, 50000, "A"),
                             ("test", n_test, 90000, "A"), ("ext", n_ext, 130000, "B")]:
        X, Y, K = build_split(n, s0, dom)
        out[f"X_{name}"], out[f"Y_{name}"], out[f"K_{name}"] = X, Y, K
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- BUSI -------------------------------------------
def load_busi(root, size=256, seed=0, split=(0.7, 0.1, 0.2)):
    """Nạp BUSI (Al-Dhabyani 2020): root/{benign,malignant,normal}/*.png, mặt nạ *_mask*.png.
    Gộp các mặt nạ của cùng một ảnh; chia phân tầng theo lớp; loại ảnh trùng lặp bằng băm trung bình."""
    items = []
    for kind in ["benign", "malignant", "normal"]:
        for f in sorted(glob.glob(os.path.join(root, kind, "*.png"))):
            if "_mask" in f:
                continue
            base = f[:-4]
            masks = sorted(glob.glob(base + "_mask*.png"))
            img = imio.imread(f, cv2.IMREAD_GRAYSCALE)
            m = np.zeros_like(img, np.uint8)
            for mf in masks:
                mm = imio.imread(mf, cv2.IMREAD_GRAYSCALE)
                if mm is not None and mm.shape == m.shape:
                    m |= (mm > 127).astype(np.uint8)
            items.append((img, m, kind))
    seen, uniq = set(), []
    for img, m, k in items:                                       # băm trung bình 8x8 để lọc trùng lặp
        h = (cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA) > img.mean()).tobytes()
        if h in seen:
            continue
        seen.add(h); uniq.append((img, m, k))
    rng = np.random.default_rng(seed)
    out = {n: ([], [], []) for n in ["train", "val", "test"]}
    for kind in ["benign", "malignant", "normal"]:
        idx = [i for i, it in enumerate(uniq) if it[2] == kind]
        rng.shuffle(idx)
        a, b = int(split[0] * len(idx)), int((split[0] + split[1]) * len(idx))
        for name, part in [("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])]:
            for i in part:
                img, m, k = uniq[i]
                out[name][0].append(_letterbox(img, size, cv2.INTER_AREA))
                out[name][1].append(_letterbox(m, size, cv2.INTER_NEAREST))
                out[name][2].append(k)
    res = {}
    for n, (X, Y, K) in out.items():
        res[f"X_{n}"], res[f"Y_{n}"], res[f"K_{n}"] = np.stack(X), np.stack(Y), np.array(K)
    return res


def _letterbox(img, size, interp):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)
    canvas = np.zeros((size, size), img.dtype)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def preprocess(img_u8, size):
    """Ảnh xám uint8 -> ảnh vuông kích thước `size` (giữ tỉ lệ, đệm viền) và chuẩn hóa [0,1]."""
    return _letterbox(img_u8, size, cv2.INTER_AREA).astype(np.float32) / 255.0


# ---------------------------------------------------------------- tăng cường -----------------------------------
def make_augment(size):
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
        A.RandomGamma((80, 120), p=0.4),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.4),
        A.ElasticTransform(alpha=20, sigma=6, p=0.2),
    ])


class SegDataset:
    """Bộ dữ liệu trả về (ảnh float32 [1,H,W], mặt nạ float32 [1,H,W])."""
    def __init__(self, X, Y, augment=None):
        self.X, self.Y, self.aug = X, Y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.aug is not None:
            r = self.aug(image=x, mask=y)
            x, y = r["image"], r["mask"]
        x = x.astype(np.float32) / 255.0
        return x[None], (y > 0).astype(np.float32)[None]
