"""Thí nghiệm trên bộ BUSI thật (Al-Dhabyani et al., 2020) – thí nghiệm R3.

Thiết lập
  - Đọc BUSI bằng data.load_busi: gộp mặt nạ nhiều khối u, lọc ảnh trùng bằng băm trung bình,
    chia 70/10/20 phân tầng theo lớp (lành / ác / bình thường), cố định hạt chia.
  - Ảnh 96×96 (letterbox), 8 kênh cơ sở – cùng cỡ với các mô hình của đồ án để chạy được trên CPU.
  - Khởi tạo từ trọng số học trên ảnh mô phỏng (đã chứng minh có ích ở R2).
  - Tối đa --epochs epoch, chọn epoch có Dice kiểm định tốt nhất.
  - Cấu hình: M1, M2, M3, M4 (công thức gốc) và M1-R, M4-R (công thức cải tiến R, xem real_experiments.py).
  - Đánh giá: tập kiểm tra BUSI (Dice/IoU/HD95 trên ảnh có khối u, tỉ lệ báo nhầm trên ảnh bình thường)
    và 42 ảnh STU-Hospital làm tập NGOÀI MIỀN (bệnh viện và thiết bị khác, không tinh chỉnh).

Chạy:  python busi_experiments.py --busi-root <thư mục Dataset_BUSI_with_GT> [--epochs 50] [--jobs 2]
       python busi_experiments.py --busi-root ... --final M4-R     # lưu weights/real/M4-BUSI.pt
Kết quả: results/busi_results.json
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SIZE, BASE = int(os.environ.get("BUSI_SIZE", 96)), 8
_CACHE = os.path.join(ROOT, "results", f"busi_cache_{SIZE}.npz")

CONFIGS = [
    dict(name="M1", arch="M1", recipe="base"),
    dict(name="M2", arch="M2", recipe="base"),
    dict(name="M3", arch="M3", recipe="base"),
    dict(name="M4", arch="M4", recipe="base"),
    dict(name="M1-R", arch="M1", recipe="R"),
    dict(name="M4-R", arch="M4", recipe="R"),
]


def load(root=None):
    if os.path.exists(_CACHE):
        z = np.load(_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import data as D
    d = D.load_busi(root, size=SIZE, seed=0)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    np.savez_compressed(_CACHE, **d)
    return d


def metrics_per(P, Y):
    import metrics as MT
    out = []
    for p, g in zip(P, Y > 0):
        if p.any():
            p = MT.largest_component(p)
        if g.any():
            out.append(dict(dice=float(MT.dice(p, g)), iou=float(MT.iou(p, g)),
                            hd95=float(MT.hd95(p, g)) if p.any() else None, lesion=True, fp=False))
        else:
            out.append(dict(dice=1.0 if not p.any() else 0.0, iou=None, hd95=None, lesion=False, fp=bool(p.any())))
    return out


def summarize(per, K=None):
    les = [x for x in per if x["lesion"]]; nor = [x for x in per if not x["lesion"]]
    hd = [x["hd95"] for x in les if x["hd95"] is not None]
    s = dict(n=len(per), n_lesion=len(les), dice=float(np.mean([x["dice"] for x in les])),
             dice_sd=float(np.std([x["dice"] for x in les])), iou=float(np.mean([x["iou"] for x in les])),
             hd95=float(np.mean(hd)) if hd else None, good=int(sum(x["dice"] >= 0.8 for x in les)),
             fp_rate=float(np.mean([x["fp"] for x in nor])) if nor else None)
    if K is not None:
        for k in ("benign", "malignant"):
            v = [x["dice"] for x, kk in zip(per, K) if kk == k and x["lesion"]]
            if v:
                s["dice_" + k] = float(np.mean(v))
    return s


def train_job(job):
    torch.set_num_threads(job.get("threads", 1))
    import models as M
    import data as D
    from train import seg_loss
    from torch.utils.data import DataLoader
    from real_experiments import augment, predict, load_stu
    cfg, seed, epochs = job["cfg"], job["seed"], job["epochs"]
    d = load()
    torch.manual_seed(seed); np.random.seed(seed)
    model = M.build(cfg["arch"], base=BASE)
    ck = torch.load(os.path.join(ROOT, "weights", "final", cfg["arch"] + ".pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"])
    R = cfg["recipe"] == "R"
    if R:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(D.SegDataset(d["X_train"], d["Y_train"], augment(cfg["recipe"])), batch_size=16, shuffle=True,
                    drop_last=True, generator=g)
    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0
        ds_w = max(0.0, 1 - ep / (0.6 * epochs)) if R else 1.0
        for xb, yb in dl:
            out = model(xb)
            if isinstance(out, tuple):
                main, aux = out
                loss = seg_loss(main, yb) + ds_w * sum(w * seg_loss(a, yb) for w, a in zip([0.5, 0.25, 0.125], aux))
            else:
                loss = seg_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        vd = summarize(metrics_per(predict(model, d["X_val"], tta=R) > 0.5, d["Y_val"]))["dice"]
        sched.step() if R else sched.step(1 - vd)
        hist.append(dict(epoch=ep + 1, loss=tl / len(dl), val_dice=vd))
        if vd > best:
            best = vd; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt = metrics_per(predict(model, d["X_test"], tta=R) > 0.5, d["Y_test"])
    Xs, Ys, _ = load_stu(size=SIZE)
    ps = metrics_per(predict(model, Xs, tta=R) > 0.5, Ys)
    res = dict(name=cfg["name"], seed=seed, best_val=best, hist=hist, time_s=round(time.time() - t0),
               test=summarize(pt, d["K_test"]), test_per=[x["dice"] for x in pt],
               test_lesion=[x["lesion"] for x in pt], stu=summarize(ps), stu_per=[x["dice"] for x in ps])
    if job.get("save"):
        os.makedirs(os.path.join(ROOT, "weights", "real"), exist_ok=True)
        torch.save(dict(state=model.state_dict(), model=cfg["arch"], base=BASE, size=SIZE, val_dice=best, test_dice=res["test"]["dice"],
                        recipe=cfg["recipe"], tta=R, trained_on="BUSI (Al-Dhabyani 2020), khởi tạo từ trọng số mô phỏng"),
                   job["save"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi-root", default=os.path.join(ROOT, "data", "external", "Dataset_BUSI_with_GT"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--tune", default=None, help="đường dẫn checkpoint cần chọn tham số hậu xử lý")
    a = ap.parse_args()
    if a.tune:
        return tune_post(a.tune)
    d = load(a.busi_root)
    print({k: v.shape for k, v in d.items() if k.startswith("X_")}, flush=True)
    if a.final:
        cfg = next(c for c in CONFIGS if c["name"] == a.final)
        out = os.path.join(ROOT, "weights", "real", f"{cfg['arch']}-BUSI{'' if SIZE == 96 else '-' + str(SIZE)}.pt")
        r = train_job(dict(cfg=cfg, seed=a.seeds[0], epochs=a.epochs, threads=2, save=out))
        print("Đã lưu", out, r["test"], r["stu"], flush=True)
        return
    cfgs = [c for c in CONFIGS if not a.only or c["name"] in a.only]
    jobs = [dict(cfg=c, seed=s, epochs=a.epochs) for s in a.seeds for c in cfgs]
    t0 = time.time(); res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(train_job, jobs):
            res.append(r)
            print(f"{r['name']} s{r['seed']}: val {r['best_val']:.3f} · BUSI test Dice {r['test']['dice']:.3f} "
                  f"(FP {r['test']['fp_rate']:.2f}) · STU {r['stu']['dice']:.3f} · {r['time_s']}s", flush=True)
    p = os.path.join(ROOT, "results", "busi_results.json")
    old = json.load(open(p, encoding="utf8")) if os.path.exists(p) else {"runs": []}
    keep = [x for x in old["runs"] if not any(x["name"] == r["name"] and x["seed"] == r["seed"] for r in res)]
    allr = keep + res
    from scipy.stats import wilcoxon
    agg = {}
    for c in CONFIGS:
        rs = [x for x in allr if x["name"] == c["name"]]
        if not rs:
            continue
        les = np.array(rs[0]["test_lesion"])
        tp = np.mean([np.array(x["test_per"]) for x in rs], 0)[les]
        sp = np.mean([np.array(x["stu_per"]) for x in rs], 0)
        agg[c["name"]] = dict(seeds=[x["seed"] for x in rs], test_dice=float(tp.mean()), test_seed_sd=float(np.std([x["test"]["dice"] for x in rs])),
                              stu_dice=float(sp.mean()), iou=float(np.mean([x["test"]["iou"] for x in rs])),
                              hd95=float(np.mean([x["test"]["hd95"] for x in rs])), fp=float(np.mean([x["test"]["fp_rate"] for x in rs])),
                              benign=float(np.mean([x["test"].get("dice_benign", np.nan) for x in rs])),
                              malignant=float(np.mean([x["test"].get("dice_malignant", np.nan) for x in rs])),
                              good=int((tp >= 0.8).sum()), n_lesion=int(les.sum()), _tp=tp.tolist(), _sp=sp.tolist())
    tests = []
    for x, y in [("M4", "M1"), ("M4", "M2"), ("M4", "M3"), ("M4-R", "M4"), ("M4-R", "M1-R"), ("M1-R", "M1"), ("M4-R", "M1")]:
        if x in agg and y in agg:
            for part in ("_tp", "_sp"):
                u, v = np.array(agg[x][part]), np.array(agg[y][part])
                pv = float(wilcoxon(u, v).pvalue) if np.any(u != v) else 1.0
                tests.append(dict(a=x, b=y, set="BUSI" if part == "_tp" else "STU", diff=float((u - v).mean()), p=pv,
                                  wins=int((u > v).sum()), n=len(u)))
    out = dict(dataset=dict(name="BUSI", ref="Al-Dhabyani et al., Data in Brief 28: 104863, 2020",
                            n={k[2:]: int(len(v)) for k, v in d.items() if k.startswith("X_")}, size=SIZE),
               setup=dict(epochs=a.epochs, base=BASE, init="synthetic"), runs=allr,
               summary={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in agg.items()}, tests=tests)
    json.dump(out, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    for k, v in out["summary"].items():
        print(f"{k:6s} BUSI {v['test_dice']:.3f} · STU {v['stu_dice']:.3f} · HD95 {v['hd95']:.2f} · FP {v['fp']:.2f}", flush=True)
    for t in tests:
        print(f"  {t['a']} vs {t['b']} [{t['set']}]: Δ={t['diff']:+.4f} p={t['p']:.3f} thắng {t['wins']}/{t['n']}", flush=True)
    print(f"Xong sau {time.time() - t0:.0f}s", flush=True)



# ------------------------------------------------------------------ hậu xử lý chống báo nhầm
def apply_post(prob, thr=0.5, min_area=0.0, min_peak=0.0):
    """prob: [H,W]. Giữ vùng liên thông lớn nhất; bỏ hẳn nếu vùng nhỏ hơn min_area (tỉ lệ khung)
    hoặc xác suất cao nhất trong vùng thấp hơn min_peak – tức coi là ảnh không có khối u."""
    import metrics as MT
    m = prob > thr
    if not m.any():
        return m
    m = MT.largest_component(m)
    if m.mean() < min_area or prob[m].max() < min_peak:
        return np.zeros_like(m)
    return m


def tune_post(model_path):
    """Chọn (ngưỡng, diện tích tối thiểu, đỉnh xác suất tối thiểu) trên TẬP KIỂM ĐỊNH, tối đa Dice trung bình
    trên mọi ảnh (ảnh bình thường: 1 nếu không báo khối u, 0 nếu báo nhầm). Ghi kết quả vào checkpoint."""
    import models as M
    from real_experiments import predict, load_stu
    d = load(); ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = M.build(ck["model"], base=ck["base"]); model.load_state_dict(ck["state"])
    tta = ck.get("tta", False)
    Pv = predict(model, d["X_val"], tta=tta); Yv = d["Y_val"] > 0

    def score(P, Y, **kw):
        s = []
        for p, g in zip(P, Y):
            m = apply_post(p, **kw)
            if g.any():
                den = m.sum() + g.sum(); s.append(2 * (m & g).sum() / den if den else 1.0)
            else:
                s.append(0.0 if m.any() else 1.0)
        return float(np.mean(s))
    grid = [dict(thr=t, min_area=a, min_peak=pk) for t in (0.4, 0.5, 0.6) for a in (0.0, 0.003, 0.006, 0.01, 0.02)
            for pk in (0.0, 0.7, 0.8, 0.9, 0.95)]
    base = dict(thr=0.5, min_area=0.0, min_peak=0.0)
    best = max(grid, key=lambda kw: score(Pv, Yv, **kw))
    out = {}
    Pt = predict(model, d["X_test"], tta=tta); Yt = d["Y_test"] > 0
    Xs, Ys, _ = load_stu(size=SIZE); Ps = predict(model, Xs, tta=tta)
    for name, kw in (("base", base), ("tuned", best)):
        per = []
        for p, g in zip(Pt, Yt):
            m = apply_post(p, **kw)
            if g.any():
                per.append(("les", 2 * (m & g).sum() / (m.sum() + g.sum())))
            else:
                per.append(("nor", bool(m.any())))
        les = [v for k, v in per if k == "les"]; nor = [v for k, v in per if k == "nor"]
        stu = []
        for p, g in zip(Ps, Ys > 0):
            m = apply_post(p, **kw); stu.append(2 * (m & g).sum() / (m.sum() + g.sum()))
        out[name] = dict(params=kw, val_score=score(Pv, Yv, **kw), test_lesion_dice=float(np.mean(les)),
                         test_fp=float(np.mean(nor)), test_missed=int(sum(v == 0 for v in les)), stu_dice=float(np.mean(stu)),
                         test_all=float(np.mean([v if k == "les" else (0.0 if v else 1.0) for k, v in per])))
    ck["post"] = best; torch.save(ck, model_path)
    p = os.path.join(ROOT, "results", "busi_results.json")
    if os.path.exists(p):
        j = json.load(open(p, encoding="utf8")); j["post"] = dict(model=os.path.basename(model_path), **out)
        json.dump(j, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
