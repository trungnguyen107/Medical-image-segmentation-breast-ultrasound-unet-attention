"""Huấn luyện mô hình chạy nền từ giao diện web (mỗi lúc một tác vụ) và đánh giá mô hình trên CSDL."""
import json, os, threading, time
import numpy as np
import torch
from torch.utils.data import DataLoader

import data as D
import dataset_tools as T
import db
import metrics as MT
import models as M
from train import seg_loss, predict, batch_dice

ROOT = os.path.dirname(os.path.abspath(__file__))
USER_W = os.path.join(ROOT, "weights", "user")
_job = {"id": None, "thread": None, "stop": False}
_lock = threading.Lock()


def busy():
    t = _job["thread"]
    return t is not None and t.is_alive()


def start(cfg):
    """cfg: model_type, epochs, bs, lr, size, base, n_train, augment, name"""
    with _lock:
        if busy():
            raise RuntimeError("Đang có một tác vụ huấn luyện chạy. Hãy chờ hoặc dừng tác vụ đó.")
        n_avail = db.q("SELECT COUNT(*) n FROM images WHERE dataset=? AND split='train' AND mask IS NOT NULL", (cfg["dataset"],), one=True)["n"]
        n_val = db.q("SELECT COUNT(*) n FROM images WHERE dataset=? AND split='val' AND mask IS NOT NULL", (cfg["dataset"],), one=True)["n"]
        if n_avail < 8 or n_val < 2:
            raise RuntimeError("Bộ dữ liệu chưa đủ ảnh huấn luyện/kiểm định.")
        cfg["n_train"] = min(cfg["n_train"], n_avail)
        rid = db.insert("runs", dict(name=cfg["name"], model_type=cfg["model_type"], epochs=cfg["epochs"], bs=cfg["bs"], lr=cfg["lr"],
                                     size=cfg["size"], base=cfg["base"], n_train=cfg["n_train"], n_val=n_val, dataset=cfg["dataset"],
                                     augment=int(cfg["augment"]), status="queued", progress=0, history=[], created_at=db.now()))
        _job.update(id=rid, stop=False)
        th = threading.Thread(target=_run, args=(rid, cfg), daemon=True)
        _job["thread"] = th; th.start()
        return rid


def stop():
    if busy():
        _job["stop"] = True
        return _job["id"]
    return None


def _run(rid, cfg):
    t0 = time.time()
    try:
        torch.manual_seed(0); np.random.seed(0)
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))
        db.update("runs", rid, dict(status="running", message="Đang nạp dữ liệu…"))
        X, Y, _, _ = T.load_split("train", cfg["size"], cfg["dataset"], limit=cfg["n_train"])
        Xv, Yv, _, _ = T.load_split("val", cfg["size"], cfg["dataset"])
        model = M.build(cfg["model_type"], base=cfg["base"])
        params = M.count_params(model)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        ds = D.SegDataset(X, Y, D.make_augment(cfg["size"]) if cfg["augment"] else None)
        dl = DataLoader(ds, batch_size=cfg["bs"], shuffle=True, num_workers=0, drop_last=len(ds) > cfg["bs"])
        os.makedirs(USER_W, exist_ok=True)
        wpath = os.path.join(USER_W, f"run_{rid}.pt")
        hist, best, best_ep = [], -1.0, 0
        db.update("runs", rid, dict(params=params, weights=os.path.relpath(wpath, ROOT), message="Đang huấn luyện…"))
        nb = max(1, len(dl))
        for ep in range(1, cfg["epochs"] + 1):
            model.train(); tl = 0.0
            for bi, (xb, yb) in enumerate(dl):
                if _job["stop"]:
                    raise InterruptedError
                out = model(xb)
                if isinstance(out, tuple):
                    main, aux = out
                    loss = seg_loss(main, yb) + sum(w * seg_loss(a, yb) for w, a in zip([0.5, 0.25, 0.125], aux))
                else:
                    loss = seg_loss(out, yb)
                opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
                db.update("runs", rid, dict(progress=round(((ep - 1) + (bi + 1) / nb) / cfg["epochs"], 4)))
            vd = float(np.mean(batch_dice(predict(model, Xv), Yv)))
            sched.step(1 - vd)
            hist.append(dict(epoch=ep, loss=round(tl / nb, 5), val_dice=round(vd, 5), lr=opt.param_groups[0]["lr"], t=round(time.time() - t0, 1)))
            if vd > best:
                best, best_ep = vd, ep
                torch.save({"state": model.state_dict(), "model": cfg["model_type"], "base": cfg["base"], "size": cfg["size"],
                            "val_dice": vd, "run_id": rid, "name": cfg["name"]}, wpath)
            db.update("runs", rid, dict(epoch=ep, history=hist, best_val_dice=round(best, 5), best_epoch=best_ep,
                                        duration_s=round(time.time() - t0, 1), message=f"Epoch {ep}/{cfg['epochs']}"))
        db.update("runs", rid, dict(message="Đang đánh giá trên tập kiểm tra…"))
        res = evaluate(f"R{rid}", "test", dataset=cfg["dataset"])
        db.update("runs", rid, dict(status="done", progress=1, test_dice=res["dice"], test_iou=res["iou"],
                                    duration_s=round(time.time() - t0, 1), finished_at=db.now(), message="Hoàn tất"))
    except InterruptedError:
        st = "stopped" if os.path.exists(os.path.join(USER_W, f"run_{rid}.pt")) else "failed"
        db.update("runs", rid, dict(status=st, duration_s=round(time.time() - t0, 1), finished_at=db.now(),
                                    message="Đã dừng theo yêu cầu" + (" – giữ trọng số tốt nhất" if st == "stopped" else "")))
    except Exception as e:                                  # noqa
        db.update("runs", rid, dict(status="failed", finished_at=db.now(), message=f"Lỗi: {e}"))
    finally:
        _job["stop"] = False


# ------------------------------------------------------------------ đánh giá -------------------------------------------
_eval_cache = {}


def load_model(model_id):
    """M1–M4: trọng số có sẵn của đồ án; R<n>: mô hình huấn luyện trên web."""
    if model_id in _eval_cache:
        return _eval_cache[model_id]
    if model_id.startswith("R"):
        p = os.path.join(USER_W, f"run_{int(model_id[1:])}.pt")
    else:
        b = os.path.basename(model_id) + ".pt"
        p = os.path.join(ROOT, "weights", "final", b)
        if not os.path.exists(p):  # các biến thể ablation (A3, A4, D1–D3)
            p = os.path.join(ROOT, "weights", "ablation", b)
        if not os.path.exists(p):  # mô hình tinh chỉnh trên ảnh thật (M4-STU, M1-STU)
            p = os.path.join(ROOT, "weights", "real", b)
    if not os.path.exists(p):
        raise ValueError(f"Không tìm thấy trọng số của mô hình {model_id}.")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = M.build(ck["model"], base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    _eval_cache[model_id] = (m, ck)
    return _eval_cache[model_id]


def forget(model_id):
    _eval_cache.pop(model_id, None)


def evaluate(model_id, split="test", dataset=None):
    dataset = dataset or ("stu" if split == "real" else "sample")
    model, ck = load_model(model_id)
    X, Y, K, rows = T.load_split(split, ck["size"], dataset)
    if len(X) == 0:
        raise ValueError("Tập dữ liệu rỗng.")
    t0 = time.perf_counter()
    P = predict(model, X) > 0.5
    ms = (time.perf_counter() - t0) * 1000 / len(X)
    per = []
    for p, g, k, r in zip(P, Y > 0, K, rows):
        if p.any():
            p = MT.largest_component(p)
        d = 1.0 if (not p.any() and not g.any()) else MT.dice(p, g)
        per.append(dict(id=r["id"], kind=k, dice=round(float(d), 4), iou=round(float(MT.iou(p, g)) if (p.any() or g.any()) else 1.0, 4),
                        precision=float(MT.precision(p, g)) if p.any() else None, recall=float(MT.recall(p, g)) if g.any() else None,
                        hd95=MT.hd95(p, g) if (p.any() and g.any()) else None, fp=bool(p.any() and not g.any())))
    les = [x for x in per if x["kind"] != "normal"]
    nor = [x for x in per if x["kind"] == "normal"]
    mean = lambda L, k: float(np.mean([x[k] for x in L if x[k] is not None])) if any(x[k] is not None for x in L) else None
    by_kind = {k: round(mean([x for x in per if x["kind"] == k], "dice"), 4) for k in ["benign", "malignant", "normal"] if any(x["kind"] == k for x in per)}
    res = dict(model_id=model_id, split=split, n=len(per), dice=round(mean(les, "dice"), 4), dice_sd=round(float(np.std([x["dice"] for x in les])), 4),
               iou=round(mean(les, "iou"), 4), precision=round(mean(les, "precision"), 4), recall=round(mean(les, "recall"), 4),
               hd95=round(mean(les, "hd95"), 3) if mean(les, "hd95") is not None else None,
               fp_rate=round(float(np.mean([x["fp"] for x in nor])), 4) if nor else None, ms=round(ms, 2),
               by_kind=by_kind, per_image=[dict(id=x["id"], kind=x["kind"], dice=x["dice"]) for x in per], created_at=db.now())
    db.x("DELETE FROM evals WHERE model_id=? AND split=?", (model_id, split))
    db.insert("evals", res)
    return res


def degrade(X, kind, level, seed=123):
    rng = np.random.default_rng(seed); x = X.astype(np.float32) / 255.0
    if kind == "speckle":
        x = x * (1 + level * rng.normal(size=x.shape))
    elif kind == "gauss":
        x = x + level * rng.normal(size=x.shape)
    else:
        m = x.mean(axis=tuple(range(1, x.ndim)), keepdims=True); x = m + level * (x - m)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


LEVELS = dict(speckle=[0.0, 0.1, 0.2, 0.3, 0.4], gauss=[0.0, 0.05, 0.1, 0.15, 0.2], contrast=[1.0, 0.8, 0.6, 0.4, 0.2])


def robustness(model_ids, kind="speckle", dataset="sample", n=40):
    out = {}
    for mid in model_ids:
        model, ck = load_model(mid)
        X, Y, K, _ = T.load_split("test", ck["size"], dataset)
        sel = np.where(K != "normal")[0][:n]
        X, Y = X[sel], Y[sel] > 0
        vals = []
        for L in LEVELS[kind]:
            P = predict(model, degrade(X, kind, L)) > 0.5
            vals.append(round(float(np.mean([MT.dice(p, g) for p, g in zip(P, Y)])), 4))
        out[mid] = vals
    return dict(kind=kind, levels=LEVELS[kind], series=out)
