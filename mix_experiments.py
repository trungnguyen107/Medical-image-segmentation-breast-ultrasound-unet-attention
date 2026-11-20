"""Thí nghiệm R4 – CẢI THIỆN DỮ LIỆU ĐẦU VÀO: gộp nhiều bộ ảnh siêu âm vú thật có mặt nạ.

Câu hỏi
  (a) Học trên nhiều bộ dữ liệu (nhiều bệnh viện, nhiều máy) có tốt hơn chỉ học trên BUSI không?
  (b) Khi đủ dữ liệu, M4 (Attention Gate + CBAM + giám sát sâu) có hơn U-Net, Attention U-Net, U-Net + CBAM không?

Dữ liệu (mọi ảnh đều có mặt nạ do bác sĩ vẽ)
  - BUSI (Ai Cập)      : dùng ĐÚNG cách chia của R3 (682 ảnh sau lọc trùng, 476/68/138) để so sánh được.
  - BUS-BRA (Brazil)   : 1.875 ảnh, 1.064 bệnh nhân – chia 70/10/20 THEO BỆNH NHÂN, phân tầng lành/ác.
  - BrEaST (Ba Lan)    : 252 ảnh có ảnh gốc + mặt nạ, mỗi bệnh nhân một ảnh – chia 70/10/20.
  - STU-Hospital (TQ)  : 42 ảnh – KHÔNG học, chỉ kiểm tra ngoài miền.
  Ảnh gần trùng giữa các bộ (dHash, Hamming ≤ 4) bị loại khỏi bộ sau.

Cấu hình (đều dùng công thức R, khởi tạo từ trọng số mô phỏng, ảnh 96×96)
  M1-R, M2-R, M3-R, M4-R học trên tập gộp; M4-R·BUSI là trọng số chỉ học BUSI (R3) để so sánh tác dụng của dữ liệu.

Chạy:  python mix_experiments.py [--epochs 40] [--jobs 2]
       python mix_experiments.py --final M4-R      # lưu weights/real/M4-MIX.pt cho web
Kết quả: results/mix_results.json
"""
import argparse, csv, glob, json, os, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "external")
SIZE, BASE = int(os.environ.get("MIX_SIZE", 96)), 8
_CACHE = os.path.join(ROOT, "results", f"mix_cache_{SIZE}.npz")
SETS = ("busi", "busbra", "breast")
NAMES = dict(busi="BUSI (Ai Cập)", busbra="BUS-BRA (Brazil)", breast="BrEaST (Ba Lan)", stu="STU (Trung Quốc, ngoài miền)")

CONFIGS = [dict(name=f"{a}-R", arch=a, recipe="R") for a in ("M1", "M2", "M3", "M4")]


def _lb(img, interp):
    import data as D
    return D._letterbox(img, SIZE, interp)


def _dhash(img):
    g = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def _split_groups(groups, strata, seed=0, split=(0.7, 0.1, 0.2)):
    """groups: mã bệnh nhân của từng ảnh; strata: nhãn phân tầng theo bệnh nhân. Trả về tên tập cho từng ảnh."""
    rng = np.random.default_rng(seed)
    pats = sorted(set(groups)); by = {}
    for p in pats:
        by.setdefault(strata[p], []).append(p)
    where = {}
    for s in sorted(by):
        ps = by[s]; rng.shuffle(ps)
        a, b = int(split[0] * len(ps)), int((split[0] + split[1]) * len(ps))
        for name, part in (("train", ps[:a]), ("val", ps[a:b]), ("test", ps[b:])):
            for p in part:
                where[p] = name
    return [where[g] for g in groups]


def _read_busbra():
    root = os.path.join(EXT, "BUS-BRA"); rows = list(csv.DictReader(open(os.path.join(root, "bus_data.csv"), encoding="utf8")))
    items = []
    for r in rows:
        img = cv2.imread(os.path.join(root, "Images", r["ID"] + ".png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(os.path.join(root, "Masks", r["ID"].replace("bus_", "mask_") + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None or m.shape != img.shape:
            continue
        items.append(dict(img=img, mask=(m > 127).astype(np.uint8), kind=r["Pathology"].strip().lower(), patient="bra" + r["Case"], file=r["ID"] + ".png"))
    return items


def _read_breast():
    root = os.path.join(EXT, "BrEaST"); items = []; lab = {}
    xl = glob.glob(os.path.join(root, "*clinical-data*.xlsx"))
    if xl:                                                     # nhãn lành/ác từ bảng lâm sàng gốc của BrEaST
        import openpyxl
        rows = list(openpyxl.load_workbook(xl[0], read_only=True).active.iter_rows(values_only=True))
        ci, cc = rows[0].index("Image_filename"), rows[0].index("Classification")
        lab = {str(r[ci]).split(".")[0]: str(r[cc]).strip().lower() for r in rows[1:] if r[ci]}
    for f in sorted(glob.glob(os.path.join(root, "case*_tumor.png"))):
        k = os.path.basename(f).split("_")[0]
        img = cv2.imread(os.path.join(root, k + ".png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None or m is None:
            continue
        m = (m.max(-1) if m.ndim == 3 else m) > 127
        if m.shape != img.shape:
            continue
        items.append(dict(img=img, mask=m.astype(np.uint8), kind=lab.get(k, "lesion"), patient="brs" + k, file=k + ".png"))
    return items


def load():
    """Trả về dict: X_<tập>_<bộ>, Y_..., K_... cho bộ ∈ SETS và tập ∈ train/val/test; kèm 'info' (số ảnh, số trùng)."""
    if os.path.exists(_CACHE):
        z = np.load(_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import busi_experiments as BE
    out, info = {}, {}
    b = BE.load()                                                           # BUSI: đúng cách chia của R3
    if b["X_train"].shape[1] != SIZE:
        raise SystemExit("Kích thước cache BUSI khác MIX_SIZE")
    hashes = []
    for sp in ("train", "val", "test"):
        out[f"X_{sp}_busi"], out[f"Y_{sp}_busi"], out[f"K_{sp}_busi"] = b[f"X_{sp}"], b[f"Y_{sp}"], b[f"K_{sp}"]
    # BUSI gốc (trước letterbox) để băm – dùng ảnh letterbox cũng được vì dHash bền với đổi kích thước
    for sp in ("train", "val", "test"):
        hashes += [_dhash(x) for x in b[f"X_{sp}"]]
    info["busi"] = dict(n=int(sum(len(b[f"X_{sp}"]) for sp in ("train", "val", "test"))), dup_removed=0)
    import real_data as RDm
    ix = RDm.HashIndex(4)
    for h in hashes:
        ix.add(h, "busi")
    for name, reader in (("busbra", _read_busbra), ("breast", _read_breast)):
        items = reader(); keep = []; dup = 0
        for it in items:
            h = _dhash(_lb(it["img"], cv2.INTER_AREA))
            if ix.find(h) is not None:
                dup += 1; continue
            ix.add(h, name); keep.append(it)
        strata = {}
        for it in keep:                      # BUS-BRA phân tầng lành/ác; BrEaST chia ngẫu nhiên (mỗi bệnh nhân một ảnh)
            strata.setdefault(it["patient"], it["kind"] if name == "busbra" else "lesion")
        where = _split_groups([it["patient"] for it in keep], strata)
        for sp in ("train", "val", "test"):
            sel = [it for it, w in zip(keep, where) if w == sp]
            out[f"X_{sp}_{name}"] = np.stack([_lb(it["img"], cv2.INTER_AREA) for it in sel])
            out[f"Y_{sp}_{name}"] = np.stack([_lb(it["mask"], cv2.INTER_NEAREST) for it in sel])
            out[f"K_{sp}_{name}"] = np.array([it["kind"] for it in sel])
        info[name] = dict(n=len(keep), dup_removed=dup, patients=len(set(it["patient"] for it in keep)))
    out["info"] = np.array(json.dumps(info))
    np.savez_compressed(_CACHE, **out)
    return out


def split_map():
    """Tên tệp → tập (train/val/test) đúng như R4, để CSDL của web dùng cùng cách chia (không rò rỉ ảnh huấn luyện sang tập kiểm tra).
    Ghi results/mix_split.json."""
    import busi_experiments as BE
    import real_data as RDm
    b = BE.load(); ix = RDm.HashIndex(4); out = {}
    for sp in ("train", "val", "test"):
        for x in b[f"X_{sp}"]:
            ix.add(_dhash(x), "busi")
    for name, reader in (("busbra", _read_busbra), ("breast", _read_breast)):
        keep = []
        for it in reader():
            h = _dhash(_lb(it["img"], cv2.INTER_AREA))
            if ix.find(h) is None:
                ix.add(h, name); keep.append(it)
        strata = {}
        for it in keep:                      # BUS-BRA phân tầng lành/ác; BrEaST chia ngẫu nhiên (mỗi bệnh nhân một ảnh)
            strata.setdefault(it["patient"], it["kind"] if name == "busbra" else "lesion")
        out[name] = {it["file"]: w for it, w in zip(keep, _split_groups([it["patient"] for it in keep], strata))}
    json.dump(out, open(os.path.join(ROOT, "results", "mix_split.json"), "w", encoding="utf8"), indent=0)
    return out


def train_job(job):
    torch.set_num_threads(job.get("threads", 1))
    import models as M
    import data as D
    from train import seg_loss
    from torch.utils.data import DataLoader
    from real_experiments import augment, predict, load_stu
    import busi_experiments as BE
    cfg, seed, epochs, sets = job["cfg"], job["seed"], job["epochs"], job.get("sets", SETS)
    d = load()
    X = np.concatenate([d[f"X_train_{s}"] for s in sets]); Y = np.concatenate([d[f"Y_train_{s}"] for s in sets])
    Xv = np.concatenate([d[f"X_val_{s}"] for s in sets]); Yv = np.concatenate([d[f"Y_val_{s}"] for s in sets])
    torch.manual_seed(seed); np.random.seed(seed)
    model = M.build(cfg["arch"], base=BASE)
    model.load_state_dict(torch.load(os.path.join(ROOT, "weights", "final", cfg["arch"] + ".pt"), map_location="cpu", weights_only=False)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(D.SegDataset(X, Y, augment("R")), batch_size=16, shuffle=True, drop_last=True, generator=g)
    best, best_state, hist, t0 = -1, None, [], time.time()
    for ep in range(epochs):
        model.train(); tl = 0
        ds_w = max(0.0, 1 - ep / (0.6 * epochs))
        for xb, yb in dl:
            out = model(xb)
            if isinstance(out, tuple):
                main, aux = out
                loss = seg_loss(main, yb) + ds_w * sum(w * seg_loss(a, yb) for w, a in zip([0.5, 0.25, 0.125], aux))
            else:
                loss = seg_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        vd = BE.summarize(BE.metrics_per(predict(model, Xv, tta=True) > 0.5, Yv))["dice"]
        sched.step(); hist.append(dict(epoch=ep + 1, loss=tl / len(dl), val_dice=vd))
        print(f"  {cfg['name']} ep{ep + 1}/{epochs} loss {tl / len(dl):.3f} val {vd:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vd > best:
            best = vd; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = dict(name=cfg["name"] + ("" if tuple(sets) == SETS else "·" + "+".join(sets)), seed=seed, best_val=best, hist=hist,
               time_s=round(time.time() - t0), n_train=int(len(X)))
    res.update(evaluate_model(model, d))
    if job.get("save"):
        torch.save(dict(state=model.state_dict(), model=cfg["arch"], base=BASE, size=SIZE, val_dice=best, test_dice=res["test"]["all"]["dice"],
                        recipe="R", tta=True, trained_on="BUSI + BUS-BRA + BrEaST (" + str(len(X)) + " ảnh huấn luyện), khởi tạo từ trọng số mô phỏng"),
                   job["save"])
    return res


def evaluate_model(model, d, tta=True):
    from real_experiments import predict, load_stu
    import busi_experiments as BE
    test, per = {}, {}
    allp = []
    for s in SETS:
        p = BE.metrics_per(predict(model, d[f"X_test_{s}"], tta=tta) > 0.5, d[f"Y_test_{s}"])
        test[s] = BE.summarize(p, d[f"K_test_{s}"]); per[s] = [x["dice"] for x in p]; allp += p
        per[s + "_lesion"] = [x["lesion"] for x in p]
    test["all"] = BE.summarize(allp)
    Xs, Ys, _ = load_stu(size=SIZE)
    ps = BE.metrics_per(predict(model, Xs, tta=tta) > 0.5, Ys)
    test["stu"] = BE.summarize(ps); per["stu"] = [x["dice"] for x in ps]; per["stu_lesion"] = [True] * len(ps)
    return dict(test=test, per=per)


def eval_checkpoint(path, name):
    import models as M
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["size"] != SIZE:
        raise SystemExit(f"{path}: cỡ {ck['size']} khác {SIZE}")
    model = M.build(ck["model"], base=ck["base"]); model.load_state_dict(ck["state"]); model.eval()
    r = dict(name=name, seed=0, best_val=ck.get("val_dice"), n_train=476, time_s=0, hist=[])
    r.update(evaluate_model(model, load(), tta=ck.get("tta", True)))
    return r


def wilcoxon_pairs(runs, pairs):
    from scipy.stats import wilcoxon
    by = {r["name"]: r for r in runs}; out = []
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        for s in SETS + ("stu", "all"):
            if s == "all":
                pa = np.concatenate([np.array(by[a]["per"][k])[np.array(by[a]["per"][k + "_lesion"])] for k in SETS])
                pb = np.concatenate([np.array(by[b]["per"][k])[np.array(by[b]["per"][k + "_lesion"])] for k in SETS])
            else:
                m = np.array(by[a]["per"][s + "_lesion"])
                pa, pb = np.array(by[a]["per"][s])[m], np.array(by[b]["per"][s])[m]
            dlt = pa - pb
            p = float(wilcoxon(pa, pb).pvalue) if np.any(dlt != 0) else 1.0
            out.append(dict(a=a, b=b, set=s, diff=float(dlt.mean()), p=p, wins=int((dlt > 0).sum()), n=int(len(dlt))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--baseline", default=os.path.join(ROOT, "weights", "real", "M4-BUSI-96.pt"),
                    help="trọng số M4-R chỉ học BUSI (R3) để so sánh tác dụng của dữ liệu")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--split-map", action="store_true", help="ghi results/mix_split.json cho CSDL web")
    a = ap.parse_args()
    if a.split_map:
        m = split_map(); print({k: len(v) for k, v in m.items()}); return
    d = load()
    info = json.loads(str(d["info"]))
    for s in SETS:
        info[s].update({sp: int(len(d[f"X_{sp}_{s}"])) for sp in ("train", "val", "test")})
    print(json.dumps(info, ensure_ascii=False), flush=True)
    if a.info:
        return
    if a.final:
        cfg = next(c for c in CONFIGS if c["name"] == a.final)
        out = os.path.join(ROOT, "weights", "real", f"{cfg['arch']}-MIX.pt")
        r = train_job(dict(cfg=cfg, seed=0, epochs=a.epochs, threads=1, save=out))   # 1 luồng: trùng với lượt trong thí nghiệm
        print("Đã lưu", out, json.dumps(r["test"]["all"]), flush=True)
        return
    p = os.path.join(ROOT, "results", "mix_results.json")
    old = json.load(open(p, encoding="utf8")) if os.path.exists(p) else {"runs": []}
    cfgs = [c for c in CONFIGS if not a.only or c["name"] in a.only]
    jobs = [dict(cfg=c, seed=0, epochs=a.epochs) for c in cfgs]
    t0 = time.time(); res = []
    if jobs:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(train_job, jobs):
                res.append(r)
                print(f"{r['name']}: val {r['best_val']:.3f} · test {r['test']['all']['dice']:.3f} · "
                      + " · ".join(f"{s} {r['test'][s]['dice']:.3f}" for s in SETS + ('stu',)) + f" · {r['time_s']}s", flush=True)
    if a.baseline and os.path.exists(a.baseline):
        res.append(eval_checkpoint(a.baseline, "M4-R·BUSI"))
    names = {r["name"] for r in res}
    runs = [x for x in old["runs"] if x["name"] not in names] + res
    tests = wilcoxon_pairs(runs, [("M4-R", "M1-R"), ("M4-R", "M2-R"), ("M4-R", "M3-R"), ("M2-R", "M1-R"), ("M4-R", "M4-R·BUSI")])
    json.dump(dict(dataset=info, setup=dict(size=SIZE, base=BASE, epochs=a.epochs, recipe="R", init="trọng số mô phỏng",
                                            selection="Dice trên tập kiểm định gộp"), runs=runs, tests=tests),
              open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    for r in sorted(runs, key=lambda r: r["name"]):
        print(f"{r['name']:10s} " + " · ".join(f"{s} {r['test'][s]['dice']:.3f}" for s in SETS + ("all", "stu")))
    for t in tests:
        print(f"  {t['a']} vs {t['b']} [{t['set']}]: Δ={t['diff']:+.4f} p={t['p']:.3f} thắng {t['wins']}/{t['n']}")
    print(f"Xong sau {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
