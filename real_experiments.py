"""Thí nghiệm trên ảnh siêu âm vú THẬT (bộ STU-Hospital, 42 cặp ảnh – mặt nạ).

Nguồn dữ liệu: https://github.com/xbhlk/STU-Hospital  (Zhuang et al., PLoS ONE 2019, doi:10.1371/journal.pone.0221535)
Tải về:        python manage.py import-stu      (hoặc git clone vào data/external/STU-Hospital)

Các thí nghiệm
  R1  Zero-shot: mô hình M1–M4 huấn luyện trên ảnh mô phỏng, đánh giá thẳng trên 42 ảnh thật.
  R2  Kiểm định chéo 5 phần: tinh chỉnh (fine-tune) trên ảnh thật, đánh giá trên phần giữ lại.
      Mỗi cấu hình chạy 3 hạt ngẫu nhiên; Dice tính trên từng ảnh của phần kiểm tra rồi lấy trung bình.
      Cấu hình:
        M1, M2, M3, M4        công thức huấn luyện gốc, khởi tạo từ trọng số mô phỏng
        M4-scratch            M4, khởi tạo ngẫu nhiên (đo tác dụng của tiền huấn luyện trên ảnh mô phỏng)
        M1-R, M4-R            công thức cải tiến R (xem dưới) – để tách tác dụng của công thức khỏi kiến trúc
      Công thức cải tiến R:
        - giám sát sâu với trọng số giảm dần về 0 (annealing) ở 60% số epoch đầu – chỉ ảnh hưởng M4;
        - tăng cường đặc thù siêu âm: nhiễu đốm nhân, làm mờ / giảm độ phân giải, gamma rộng hơn;
        - AdamW + lịch tốc độ học cosine;
        - suy luận có lật ngang (TTA).
  Kiểm định thống kê: Wilcoxon theo cặp trên Dice từng ảnh (trung bình 3 hạt).

Chạy:  python real_experiments.py [--epochs 60] [--seeds 0 1 2] [--jobs 2]
Kết quả: results/real_results.json, results/figures/fig_real_*.png
"""
import argparse, glob, json, os, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
STU = os.path.join(ROOT, "data", "external", "STU-Hospital", "Hospital")
SIZE, BASE = 96, 8


def load_stu(path=STU, size=SIZE):
    import data as D
    X, Y, names = [], [], []
    for i in range(1, 200):
        fi, fm = os.path.join(path, f"Test_Image_{i}.png"), os.path.join(path, f"mask_{i}.png")
        if not os.path.exists(fi):
            continue
        img = cv2.imdecode(np.fromfile(fi, np.uint8), cv2.IMREAD_GRAYSCALE)
        m = cv2.imdecode(np.fromfile(fm, np.uint8), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, img.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127
        X.append((D.preprocess(img, size) * 255).astype(np.uint8))
        Y.append(cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))
        names.append(f"Test_Image_{i}")
    if not X:
        raise SystemExit(f"Không thấy ảnh STU trong {path}. Chạy: python manage.py import-stu")
    return np.stack(X), np.stack(Y), names


def augment(recipe):
    import albumentations as A
    base = [A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.6)]
    if recipe == "R":
        base += [A.RandomGamma((60, 150), p=0.5),
                 A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=False, elementwise=True, p=0.5),
                 A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.Downscale(scale_range=(0.5, 0.8))], p=0.3),
                 A.ElasticTransform(alpha=20, sigma=6, p=0.2)]
    else:
        base += [A.RandomGamma((80, 120), p=0.4), A.GaussNoise(std_range=(0.01, 0.06), p=0.4),
                 A.ElasticTransform(alpha=20, sigma=6, p=0.2)]
    return A.Compose(base)


@torch.no_grad()
def predict(model, X, tta=False):
    model.eval()
    x = torch.from_numpy(X.astype(np.float32) / 255.0)[:, None]
    p = torch.sigmoid(model(x))
    if tta:
        p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return p[:, 0].numpy()


def dice_per(P, Y):
    out = []
    for p, g in zip(P, Y > 0):
        if p.any():
            import metrics as MT
            p = MT.largest_component(p)
        den = p.sum() + g.sum()
        out.append(1.0 if den == 0 else 2 * (p & g).sum() / den)
    return np.array(out)


def train_one(job):
    """Một lần huấn luyện trên một phần (fold). Chạy trong tiến trình con."""
    torch.set_num_threads(1)
    import models as M
    from train import seg_loss
    from torch.utils.data import DataLoader
    import data as D
    cfg, seed, fold, epochs = job["cfg"], job["seed"], job["fold"], job["epochs"]
    X, Y, _ = load_stu()
    tr, te = np.array(job["train"]), np.array(job["test"])
    torch.manual_seed(seed); np.random.seed(seed)
    arch, recipe, init = cfg["arch"], cfg["recipe"], cfg["init"]
    model = M.build(arch, base=BASE)
    if init == "synthetic":
        ck = torch.load(os.path.join(ROOT, "weights", "final", arch + ".pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
    if recipe == "R":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5); sched = None
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(D.SegDataset(X[tr], Y[tr], augment(recipe)), batch_size=8, shuffle=True, drop_last=True, generator=g)
    for ep in range(epochs):
        model.train()
        ds_w = max(0.0, 1 - ep / (0.6 * epochs)) if recipe == "R" else 1.0
        for xb, yb in dl:
            out = model(xb)
            if isinstance(out, tuple):
                main, aux = out
                loss = seg_loss(main, yb) + ds_w * sum(w * seg_loss(a, yb) for w, a in zip([0.5, 0.25, 0.125], aux))
            else:
                loss = seg_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
    global _LAST
    if _KEEP:
        _LAST = model
    P = predict(model, X[te], tta=(recipe == "R")) > 0.5
    return dict(name=cfg["name"], seed=seed, fold=fold, idx=te.tolist(), dice=dice_per(P, Y[te]).tolist())


CONFIGS = [
    dict(name="M1", arch="M1", recipe="base", init="synthetic"),
    dict(name="M2", arch="M2", recipe="base", init="synthetic"),
    dict(name="M3", arch="M3", recipe="base", init="synthetic"),
    dict(name="M4", arch="M4", recipe="base", init="synthetic"),
    dict(name="M4-scratch", arch="M4", recipe="base", init="scratch"),
    dict(name="M1-R", arch="M1", recipe="R", init="synthetic"),
    dict(name="M4-R", arch="M4", recipe="R", init="synthetic"),
]


def folds(n, k=5, seed=2024):
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, k)
    return [(np.concatenate([parts[j] for j in range(k) if j != i]).tolist(), parts[i].tolist()) for i in range(k)]


def train_final(name, epochs=60, seed=0):
    """Huấn luyện cấu hình `name` trên toàn bộ 42 ảnh thật, lưu weights/real/<arch>-STU.pt để dùng trên web."""
    import models as M
    cfg = next(c for c in CONFIGS if c["name"] == name)
    n = len(load_stu()[0])
    torch.set_num_threads(2)
    job = dict(cfg=cfg, seed=seed, fold=-1, epochs=epochs, train=list(range(n)), test=[0])
    # tái dùng vòng lặp huấn luyện, nhưng giữ lại mô hình
    global _KEEP
    _KEEP = True
    r = train_one(job)
    model = _LAST
    cv = json.load(open(os.path.join(ROOT, "results", "real_results.json"), encoding="utf8"))["cv"].get(name, {}) \
        if os.path.exists(os.path.join(ROOT, "results", "real_results.json")) else {}
    os.makedirs(os.path.join(ROOT, "weights", "real"), exist_ok=True)
    out = os.path.join(ROOT, "weights", "real", f"{cfg['arch']}-STU.pt")
    torch.save(dict(state=model.state_dict(), model=cfg["arch"], base=BASE, size=SIZE, val_dice=cv.get("mean", float("nan")),
                    recipe=cfg["recipe"], tta=cfg["recipe"] == "R", trained_on="STU-Hospital (42 ảnh thật), khởi tạo từ trọng số mô phỏng"), out)
    print("Đã lưu", out, "· Dice kiểm định chéo", cv.get("mean"))


_KEEP, _LAST = False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--final", default=None, help="huấn luyện cấu hình này trên toàn bộ ảnh thật và lưu trọng số")
    a = ap.parse_args()
    if a.final:
        return train_final(a.final, a.epochs)
    import models as M
    from scipy.stats import wilcoxon
    X, Y, names = load_stu()
    n = len(X)
    print(f"STU-Hospital: {n} ảnh thật", flush=True)
    # R1: zero-shot
    zero = {}
    for k in ("M1", "M2", "M3", "M4"):
        ck = torch.load(os.path.join(ROOT, "weights", "final", k + ".pt"), map_location="cpu", weights_only=False)
        m = M.build(ck["model"], base=ck["base"]); m.load_state_dict(ck["state"])
        d = dice_per(predict(m, X) > 0.5, Y); zero[k] = dict(mean=float(d.mean()), sd=float(d.std()), per=d.tolist())
        print(f"R1 zero-shot {k}: Dice {d.mean():.3f} ± {d.std():.3f}", flush=True)
    # R2: kiểm định chéo
    fl = folds(n)
    cfgs = [c for c in CONFIGS if not a.only or c["name"] in a.only]
    jobs = [dict(cfg=c, seed=s, fold=f, epochs=a.epochs, train=tr, test=te) for c in cfgs for s in a.seeds for f, (tr, te) in enumerate(fl)]
    t0 = time.time(); res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, r in enumerate(ex.map(train_one, jobs)):
            res.append(r)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(jobs)} lần huấn luyện · {time.time() - t0:.0f}s", flush=True)
    per = {}
    for c in cfgs:
        mat = np.full((len(a.seeds), n), np.nan)
        for r in res:
            if r["name"] == c["name"]:
                mat[a.seeds.index(r["seed"]), r["idx"]] = r["dice"]
        per[c["name"]] = dict(per_image=np.nanmean(mat, 0).tolist(), seed_means=np.nanmean(mat, 1).tolist())
    summ = {}
    for k, v in per.items():
        pi, sm = np.array(v["per_image"]), np.array(v["seed_means"])
        summ[k] = dict(mean=float(pi.mean()), sd_img=float(pi.std()), sd_seed=float(sm.std()), median=float(np.median(pi)),
                       good=int((pi >= 0.8).sum()), n=n, cfg=next(c for c in CONFIGS if c["name"] == k))
        print(f"R2 {k:11s} Dice {pi.mean():.3f} (SD ảnh {pi.std():.3f}, SD hạt {sm.std():.3f}) · {summ[k]['good']}/{n} ảnh ≥ 0,80", flush=True)
    tests = []
    for a_, b_ in [("M4", "M1"), ("M4", "M2"), ("M4", "M3"), ("M4", "M4-scratch"), ("M4-R", "M4"), ("M4-R", "M1-R"), ("M1-R", "M1"), ("M4-R", "M1")]:
        if a_ in per and b_ in per:
            x, y = np.array(per[a_]["per_image"]), np.array(per[b_]["per_image"])
            p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
            tests.append(dict(a=a_, b=b_, diff=float((x - y).mean()), p=p, wins=int((x > y).sum()), n=n))
            print(f"  Wilcoxon {a_} vs {b_}: Δ={(x - y).mean():+.4f}, p={p:.3f}, thắng {int((x > y).sum())}/{n}", flush=True)
    out = dict(dataset=dict(name="STU-Hospital", n=n, source="https://github.com/xbhlk/STU-Hospital",
                            ref="Zhuang et al., PLoS ONE 14(8): e0221535, 2019", size=SIZE),
               setup=dict(epochs=a.epochs, seeds=a.seeds, folds=5, base=BASE), zero_shot=zero, cv=summ, per=per, tests=tests,
               names=names, time_s=round(time.time() - t0))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "real_results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("Đã lưu results/real_results.json", flush=True)


if __name__ == "__main__":
    main()
