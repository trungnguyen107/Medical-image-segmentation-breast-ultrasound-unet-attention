"""Đánh giá toàn bộ mô hình đã huấn luyện: E1–E6, kiểm định thống kê, hình và results.json."""
import glob, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import data as D
import models as M
import metrics as MT
from train import predict

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#98a8b6", "axes.labelcolor": "#14202b", "xtick.color": "#5b6b7a", "ytick.color": "#5b6b7a",
                     "axes.grid": True, "grid.color": "#e4eaef", "grid.linewidth": 0.8, "axes.axisbelow": True})
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#6250d6", "A2": "#98a8b6", "A3": "#98a8b6", "A4": "#98a8b6"}
INK = "#14202b"; MUTED = "#5b6b7a"
FIG = "results/figures/"
MAIN = ["M1", "M2", "M3", "M4"]
ABL = ["M1", "M2", "A2", "A3", "A4", "M4"]
ABL_CODE = {"M1": "A0", "M2": "A1", "A2": "A2", "A3": "A3", "A4": "A4", "M4": "A5"}
ABL_DESC = {"A0": "—", "A1": "AG", "A2": "SE (đáy)", "A3": "CBAM (đáy)", "A4": "AG + CBAM (đáy)", "A5": "AG + CBAM (đáy) + giám sát sâu"}
RUNS = "weights/runs"


def load(key, seed):
    ck = torch.load(f"{RUNS}/{key}_s{seed}.pt", map_location="cpu", weights_only=False)
    m = M.build(ck["model"], base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    return m


def per_image(prob, Y, K, thr=0.5):
    """Độ đo cho từng ảnh có khối u; tỉ lệ dương tính giả cho ảnh bình thường."""
    P = prob > thr
    idx = np.nonzero(Y.sum((1, 2)) > 0)[0]; nidx = np.nonzero(Y.sum((1, 2)) == 0)[0]
    out = {k: np.zeros(len(idx)) for k in ["dice", "iou", "precision", "recall", "specificity", "hd95"]}
    for j, i in enumerate(idx):
        mm = MT.all_metrics(P[i], Y[i] > 0)
        for k in out: out[k][j] = mm[k]
    fp = float(np.mean([P[i].any() for i in nidx])) if len(nidx) else float("nan")
    return idx, out, fp


def degrade(X, kind, level, seed=123):
    rng = np.random.default_rng(seed); x = X.astype(np.float32) / 255.0
    if kind == "speckle":
        x = x * (1 + level * rng.normal(size=x.shape))
    elif kind == "gauss":
        x = x + level * rng.normal(size=x.shape)
    elif kind == "contrast":
        m = x.mean((1, 2), keepdims=True); x = m + level * (x - m)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def boot_ci(v, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(v)
    means = [np.mean(v[rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ms_std(a):
    a = np.asarray(a, float); return float(np.nanmean(a)), float(np.nanstd(a))


def savefig(fig, name):
    fig.savefig(FIG + name, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main(CFG):
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(2)
    S = D.load_synthetic()
    Xt, Yt, Kt = S["X_test"], S["Y_test"], S["K_test"]
    Xe, Ye, Ke = S["X_ext"], S["Y_ext"], S["K_ext"]
    seeds_of = {k: [s for s in CFG["seeds"] if os.path.exists(f"{RUNS}/{k}_s{s}.pt")] for k in ["M1", "M2", "M3", "M4", "A2", "A3", "A4"]}
    res = dict(config=CFG)
    cnt = lambda K, n: dict(n=int(n), normal=int((K == "normal").sum()), benign=int((K == "benign").sum()), malignant=int((K == "malignant").sum()))
    res["data"] = dict(train=cnt(S["K_train"], len(S["K_train"])), val=cnt(S["K_val"], len(S["K_val"])), test=cnt(Kt, len(Kt)), ext=cnt(Ke, len(Ke)))

    # ---------------- E1 + E4 + E5: dự đoán và độ đo ----------------
    prob_t, prob_e, met_t, met_e, fp_t, fp_e = {}, {}, {}, {}, {}, {}
    for k, ss in seeds_of.items():
        for s in ss:
            m = load(k, s)
            prob_t[k, s] = predict(m, Xt); prob_e[k, s] = predict(m, Xe)
            idx_t, met_t[k, s], fp_t[k, s] = per_image(prob_t[k, s], Yt, Kt)
            idx_e, met_e[k, s], fp_e[k, s] = per_image(prob_e[k, s], Ye, Ke)
            print("đánh giá", k, s, "Dice test", met_t[k, s]["dice"].mean().round(4), "ext", met_e[k, s]["dice"].mean().round(4), flush=True)
    names = M.NAMES
    main_rows, avg_dice = [], {}
    for k in seeds_of:
        ss = seeds_of[k]
        row = dict(key=k, name=names[k], n_seeds=len(ss))
        for mk in ["dice", "iou", "precision", "recall", "specificity", "hd95"]:
            row[mk] = ms_std([np.nanmean(met_t[k, s][mk]) for s in ss])
        row["fp_rate"] = ms_std([fp_t[k, s] for s in ss])
        avg_dice[k] = np.mean([met_t[k, s]["dice"] for s in ss], 0)
        row["ci"] = boot_ci(avg_dice[k])
        row["hd95_empty"] = int(sum(np.isnan(met_t[k, s]["hd95"]).sum() for s in ss))
        main_rows.append(row)
    res["main"] = main_rows
    wil = []
    for a, b in [("M2", "M1"), ("M3", "M1"), ("M4", "M1"), ("M4", "M2"), ("M4", "M3")]:
        d = avg_dice[a] - avg_dice[b]
        try:
            p = float(wilcoxon(avg_dice[a], avg_dice[b]).pvalue)
        except Exception:
            p = float("nan")
        wil.append(dict(a=a, b=b, mean_diff=float(d.mean()), p=p, wins=int((d > 0).sum()), n=int(len(d))))
    res["wilcoxon"] = wil
    # ablation (hạt 0)
    res["ablation"] = [dict(code=ABL_CODE[k], key=k, desc=ABL_DESC[ABL_CODE[k]], dice=float(met_t[k, 0]["dice"].mean()), iou=float(met_t[k, 0]["iou"].mean()),
                            hd95=float(np.nanmean(met_t[k, 0]["hd95"])), fp=float(fp_t[k, 0])) for k in ABL if (k, 0) in met_t]
    # E4 khái quát chéo
    ext = []
    for k in MAIN:
        ss = seeds_of[k]
        ind = ms_std([np.mean(met_t[k, s]["dice"]) for s in ss]); ex = ms_std([np.mean(met_e[k, s]["dice"]) for s in ss])
        ext.append(dict(key=k, name=names[k], dice_in=ind, dice_ext=ex, drop=ind[0] - ex[0], hd95_ext=ms_std([np.nanmean(met_e[k, s]["hd95"]) for s in ss]),
                        fp_ext=ms_std([fp_e[k, s] for s in ss])))
    res["external"] = ext
    # E5 theo nhóm
    idx_t = np.nonzero(Yt.sum((1, 2)) > 0)[0]
    area = Yt[idx_t].sum((1, 2)).astype(float); q = np.quantile(area, [1 / 3, 2 / 3])
    size_grp = np.where(area <= q[0], 0, np.where(area <= q[1], 1, 2)); kind_les = Kt[idx_t]
    grp = []
    for k in MAIN:
        d = np.mean([met_t[k, s]["dice"] for s in seeds_of[k]], 0)
        grp.append(dict(key=k, name=names[k], benign=float(d[kind_les == "benign"].mean()), malignant=float(d[kind_les == "malignant"].mean()),
                        small=float(d[size_grp == 0].mean()), medium=float(d[size_grp == 1].mean()), large=float(d[size_grp == 2].mean()),
                        fp=ms_std([fp_t[k, s] for s in seeds_of[k]])))
    res["groups"] = dict(rows=grp, area_quantiles=[float(q[0]), float(q[1])], n=[int((size_grp == i).sum()) for i in range(3)],
                         n_kind=[int((kind_les == "benign").sum()), int((kind_les == "malignant").sum())])
    # phân loại lỗi (hạt 0)
    err = {}
    for k in MAIN:
        P = prob_t[k, 0][idx_t] > 0.5; G = Yt[idx_t] > 0; c = dict(tot=int(len(idx_t)), empty=0, wrong=0, over=0, under=0, boundary=0, good=0)
        for i in range(len(idx_t)):
            mm = MT.all_metrics(P[i], G[i])
            if not P[i].any(): c["empty"] += 1
            elif mm["dice"] < 0.1: c["wrong"] += 1
            elif mm["precision"] < 0.7 <= mm["recall"]: c["over"] += 1
            elif mm["recall"] < 0.7 <= mm["precision"]: c["under"] += 1
            elif mm["dice"] < 0.85: c["boundary"] += 1
            else: c["good"] += 1
        err[k] = c
    res["errors"] = err

    # ---------------- E3 độ bền ----------------
    lev = dict(speckle=[0.0, 0.1, 0.2, 0.3, 0.4], gauss=[0.0, 0.05, 0.1, 0.15, 0.2], contrast=[1.0, 0.8, 0.6, 0.4, 0.2])
    rob = {kind: {"levels": lv, **{k: [] for k in MAIN}} for kind, lv in lev.items()}
    models_main = {(k, s): load(k, s) for k in MAIN for s in seeds_of[k]}
    for kind, lv in lev.items():
        for k in MAIN:
            rob[kind][k] = []
        for L in lv:
            Xd = Xt[idx_t] if L in (0.0, 1.0) else degrade(Xt[idx_t], kind, L)
            for k in MAIN:
                ds = []
                for s in seeds_of[k]:
                    pr = predict(models_main[k, s], Xd); ds.append(float(np.mean(
                        [MT.dice(pr[i] > 0.5, Yt[idx_t][i] > 0) for i in range(len(idx_t))])))
                rob[kind][k].append(ms_std(ds))
        print("độ bền", kind, flush=True)
    res["robustness"] = rob

    # ---------------- E6 hiệu năng ----------------
    from torch.utils.flop_counter import FlopCounterMode
    eff = []
    x1 = torch.rand(1, 1, CFG["size"], CFG["size"])
    for k in MAIN + ["A2", "A3", "A4"]:
        m = load(k, 0); m.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x1)
        for _ in range(5):
            with torch.no_grad(): m(x1)
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            with torch.no_grad(): m(x1)
            ts.append((time.perf_counter() - t) * 1000)
        eff.append(dict(key=k, name=names[k], params=M.count_params(m), gflops=fc.get_total_flops() / 1e9, ms=float(np.median(ts))))
    res["efficiency"] = eff

    # ---------------- đường cong huấn luyện ----------------
    curves, tt = {}, {}
    for k in MAIN:
        hs = [json.load(open(f"{RUNS}/{k}_s{s}_hist.json")) for s in seeds_of[k]]
        L = max(len(h["history"]) for h in hs)
        arr = np.array([[h["history"][min(i, len(h["history"]) - 1)]["val_dice"] for i in range(L)] for h in hs])
        lo = np.array([[h["history"][min(i, len(h["history"]) - 1)]["train_loss"] for i in range(L)] for h in hs])
        curves[k] = dict(val=arr.mean(0).tolist(), loss=lo.mean(0).tolist())
        tt[k] = dict(epochs=ms_std([len(h["history"]) for h in hs]), time=ms_std([h["train_time"] for h in hs]),
                     best_val=ms_std([h["best_val_dice"] for h in hs]))
    res["curves"] = curves; res["train_info"] = tt

    # ---------------- phân tích attention ----------------
    att_stats = {}
    for k in ["M2", "A4", "M4"]:
        m = load(k, 0); m.set_capture(True)
        ratios = [[] for _ in m.attention_gates()]
        for i in idx_t[:80]:
            xt = torch.from_numpy(Xt[i:i + 1].astype(np.float32) / 255.0)[None]
            with torch.no_grad(): m(xt)
            g = Yt[i] > 0
            for lv, gate in enumerate(m.attention_gates()):
                a = torch.nn.functional.interpolate(gate.last_alpha, size=g.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
                ratios[lv].append(a[g].mean() / max(a[~g].mean(), 1e-6))
        att_stats[k] = [float(np.mean(r)) for r in ratios]          # chỉ số 0 = kết nối tắt sâu nhất (tầng 4)
    res["attention"] = att_stats

    make_figures(res, S, prob_t, prob_e, met_t, idx_t, seeds_of, CFG)
    make_tables(res)
    json.dump(res, open("results/results.json", "w", encoding="utf8"), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print("Đã ghi results/results.json")




# ======================================================================================================================
def make_tables(res):
    f3 = lambda x: f"{x:.3f}".replace(".", ",")
    pm = lambda t, d=3: f"{t[0]:.{d}f} ± {t[1]:.{d}f}".replace(".", ",")
    T = []
    rows = [[f"{r['key']} – {r['name']}", pm(r["dice"]), pm(r["iou"]), pm(r["precision"]), pm(r["recall"]), pm(r["hd95"], 2), pm(r["fp_rate"])] for r in res["main"] if r["key"] in MAIN]
    best = int(np.argmax([r["dice"][0] for r in res["main"] if r["key"] in MAIN]))
    T.append(dict(title="E1 – So sánh mô hình trên tập kiểm tra miền A (trung bình ± độ lệch chuẩn qua các hạt ngẫu nhiên)",
                  columns=["Mô hình", "Dice", "IoU", "Precision", "Recall", "HD95 (px)", "Dương tính giả"], rows=rows, best=best))
    T.append(dict(title="E2 – Ablation (một hạt ngẫu nhiên)", columns=["Mã", "Thành phần", "Dice", "IoU", "HD95 (px)", "Dương tính giả"],
                  rows=[[a["code"], a["desc"], f3(a["dice"]), f3(a["iou"]), f"{a['hd95']:.2f}".replace(".", ","), f3(a["fp"])] for a in res["ablation"]]))
    T.append(dict(title="E4 – Khái quát chéo sang miền B", columns=["Mô hình", "Dice miền A", "Dice miền B", "Giảm", "Dương tính giả (B)"],
                  rows=[[f"{e['key']} – {e['name']}", pm(e["dice_in"]), pm(e["dice_ext"]), f3(e["drop"]), pm(e["fp_ext"])] for e in res["external"]]))
    T.append(dict(title="E6 – Hiệu năng tính toán", columns=["Mô hình", "Tham số (triệu)", "GFLOPs", "ms/ảnh (CPU)"],
                  rows=[[f"{e['key']} – {e['name']}", f"{e['params']/1e6:.3f}".replace(".", ","), f"{e['gflops']:.3f}".replace(".", ","), f"{e['ms']:.1f}".replace(".", ",")] for e in res["efficiency"]]))
    res["tables"] = T
    res["figures"] = [dict(title=t, file=f) for f, t in [
        ("fig_training_curves.png", "Đường cong Dice kiểm định theo epoch"), ("fig_main_comparison.png", "So sánh Dice, IoU và HD95 giữa các mô hình"),
        ("fig_boxplot_dice.png", "Phân bố Dice theo từng ảnh"), ("fig_ablation.png", "Kết quả ablation"), ("fig_robustness.png", "Độ bền với nhiễu và giảm tương phản"),
        ("fig_external.png", "Khái quát chéo sang miền B"), ("fig_subgroups.png", "Kết quả theo nhóm"), ("fig_qualitative.png", "So sánh định tính"),
        ("fig_attention_maps.png", "Bản đồ attention tại các kết nối tắt"), ("fig_errors.png", "Các ca khó nhất")]]
    res["meta"] = dict(data_note="Dữ liệu: ảnh siêu âm MÔ PHỎNG (không phải BUSI)",
                       setup=f"Huấn luyện 3 hạt ngẫu nhiên mỗi mô hình chính (M3: hạt 0, 2, 3 — hạt 1 bị sụp, xem mục 4.2), ảnh {res['config']['size']}×{res['config']['size']}, {res['config']['hw']}. Chạy lại với dữ liệu BUSI để có kết quả trên ảnh thật.")


def bar_labels(ax, bars, vals, fmt="{:.3f}", dy=0.004, fs=8):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v).replace(".", ","), ha="center", va="bottom", fontsize=fs, color=INK)


def make_figures(res, S, prob_t, prob_e, met_t, idx_t, seeds_of, CFG):
    names = M.NAMES
    # ---- đường cong huấn luyện
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for k in MAIN:
        v = res["curves"][k]["val"]; l = res["curves"][k]["loss"]
        axs[0].plot(range(1, len(v) + 1), v, color=COL[k], lw=2); axs[1].plot(range(1, len(l) + 1), l, color=COL[k], lw=2)
        axs[0].text(len(v) + 0.3, v[-1], k, color=COL[k], fontsize=9, va="center", fontweight="bold")
    axs[0].set_title("Dice kiểm định (trung bình các hạt ngẫu nhiên)", loc="left", fontsize=10); axs[1].set_title("Hàm mất mát huấn luyện", loc="left", fontsize=10)
    for a in axs: a.set_xlabel("Epoch")
    axs[0].set_ylabel("Dice"); axs[1].set_ylabel("Mất mát (BCE+Dice; M4 gồm đầu ra phụ)")
    fig.tight_layout(); savefig(fig, "fig_training_curves.png")
    # ---- so sánh chính
    rows = [r for r in res["main"] if r["key"] in MAIN]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, mk, ttl, lo in zip(axs, ["dice", "iou", "hd95"], ["Dice (cao hơn tốt hơn)", "IoU (cao hơn tốt hơn)", "HD95, điểm ảnh (thấp hơn tốt hơn)"], [None, None, 0]):
        vals = [r[mk][0] for r in rows]; sd = [r[mk][1] for r in rows]
        bars = ax.bar([r["key"] for r in rows], vals, yerr=sd, color=[COL[r["key"]] for r in rows], width=0.6, capsize=3, error_kw=dict(ecolor=INK, lw=1))
        if mk != "hd95":
            ax.set_ylim(max(0, min(vals) - 0.12), min(1.0, max(vals) + 0.06))
        bar_labels(ax, bars, vals, "{:.3f}" if mk != "hd95" else "{:.2f}", dy=(max(vals) - min(vals) + 0.01) * 0.12 + max(sd) * 1.0)
        ax.set_title(ttl, loc="left", fontsize=10)
    fig.tight_layout(); savefig(fig, "fig_main_comparison.png")
    # ---- boxplot Dice theo ảnh (trung bình qua hạt), tách lành/ác
    kind = S["K_test"][idx_t]
    fig, ax = plt.subplots(figsize=(8.8, 3.8)); pos = 0; ticks = []; labs = []
    for g, gname in [("benign", "Lành tính"), ("malignant", "Ác tính")]:
        for k in MAIN:
            d = np.mean([met_t[k, s]["dice"] for s in seeds_of[k]], 0)[kind == g]
            bp = ax.boxplot(d, positions=[pos], widths=0.65, patch_artist=True, showfliers=True, flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor=MUTED),
                            medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            bp["boxes"][0].set(facecolor=COL[k], alpha=0.85, edgecolor="white", lw=1.5); ticks.append(pos); labs.append(k); pos += 1
        ax.text(pos - 2.5, -0.17, gname, ha="center", fontsize=10, fontweight="bold", color=INK, transform=ax.get_xaxis_transform()); pos += 1
    ax.set_xticks(ticks); ax.set_xticklabels(labs); ax.set_ylabel("Dice từng ảnh"); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); savefig(fig, "fig_boxplot_dice.png")
    # ---- ablation
    ab = res["ablation"]; fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = [COL.get(a["key"], "#98a8b6") if a["key"] in MAIN else "#98a8b6" for a in ab]
    bars = ax.bar([a["code"] for a in ab], [a["dice"] for a in ab], color=cols, width=0.6)
    lo = min(a["dice"] for a in ab); hi = max(a["dice"] for a in ab); ax.set_ylim(max(0, lo - 0.1), min(1, hi + 0.05)); bar_labels(ax, bars, [a["dice"] for a in ab], dy=0.003)
    ax.set_xticks(range(len(ab))); ax.set_xticklabels([f"{a['code']}\n{a['desc']}" for a in ab], fontsize=8); ax.set_ylabel("Dice (tập kiểm tra, một hạt)")
    fig.tight_layout(); savefig(fig, "fig_ablation.png")
    # ---- độ bền
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, kind_, ttl, xl in zip(axs, ["speckle", "gauss", "contrast"], ["Nhiễu đốm nhân", "Nhiễu Gauss cộng", "Giảm tương phản"], ["Độ lệch chuẩn nhiễu nhân σ", "Độ lệch chuẩn nhiễu σ", "Hệ số tương phản (1 = gốc)"]):
        r = res["robustness"][kind_]; lv = r["levels"]
        for k in MAIN:
            m = np.array([t[0] for t in r[k]]); s = np.array([t[1] for t in r[k]])
            ax.plot(lv, m, color=COL[k], lw=2, marker="o", ms=4); ax.fill_between(lv, m - s, m + s, color=COL[k], alpha=0.12, lw=0)
            ax.text(lv[-1], m[-1], " " + k, color=COL[k], fontsize=9, va="center", fontweight="bold")
        ax.set_title(ttl, loc="left", fontsize=10); ax.set_xlabel(xl); ax.set_ylabel("Dice")
        if kind_ == "contrast": ax.invert_xaxis()
    fig.tight_layout(); savefig(fig, "fig_robustness.png")
    # ---- khái quát chéo
    ex = res["external"]; fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.36; x = np.arange(len(ex))
    b1 = ax.bar(x - w / 2, [e["dice_in"][0] for e in ex], w, color=[COL[e["key"]] for e in ex], label="Miền A (trong miền)")
    b2 = ax.bar(x + w / 2, [e["dice_ext"][0] for e in ex], w, color=[COL[e["key"]] for e in ex], alpha=0.45, hatch="///", edgecolor="white", label="Miền B (ngoài miền)")
    bar_labels(ax, b1, [e["dice_in"][0] for e in ex], dy=0.004); bar_labels(ax, b2, [e["dice_ext"][0] for e in ex], dy=0.004)
    ax.set_xticks(x); ax.set_xticklabels([e["key"] for e in ex]); ax.set_ylabel("Dice"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.0); fig.tight_layout(); savefig(fig, "fig_external.png")
    # ---- theo nhóm
    g = res["groups"]["rows"]; fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6)); w = 0.2
    for j, k in enumerate(MAIN):
        r = next(x for x in g if x["key"] == k)
        axs[0].bar(np.arange(2) + (j - 1.5) * w, [r["benign"], r["malignant"]], w, color=COL[k], label=k)
        axs[1].bar(np.arange(3) + (j - 1.5) * w, [r["small"], r["medium"], r["large"]], w, color=COL[k], label=k)
    axs[0].set_xticks(range(2)); axs[0].set_xticklabels(["Lành tính", "Ác tính"]); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["Nhỏ", "Vừa", "Lớn"])
    axs[0].set_title("Theo loại khối u", loc="left", fontsize=10); axs[1].set_title("Theo kích thước khối u (tam phân vị diện tích)", loc="left", fontsize=10)
    for a in axs: a.set_ylabel("Dice"); a.set_ylim(0, 1.0)
    axs[0].legend(frameon=False, ncol=4, fontsize=8, loc="lower left"); fig.tight_layout(); savefig(fig, "fig_subgroups.png")
    # ---- định tính
    idx_k = np.arange(len(idx_t)); dm = {k: met_t[k, 0]["dice"] for k in MAIN}
    benign = [i for i in idx_k if kind[i] == "benign"]; malig = [i for i in idx_k if kind[i] == "malignant"]
    pick = [benign[2], benign[9], malig[1], malig[7], malig[12]]
    fig, axs = plt.subplots(len(pick), 6, figsize=(11, 1.85 * len(pick)))
    for r, j in enumerate(pick):
        i = idx_t[j]; x = S["X_test"][i]; g_ = S["Y_test"][i] > 0
        panels = [("Ảnh", x, None), ("Chuẩn", x, g_)] + [(k, x, prob_t[k, 0][i] > 0.5) for k in ["M1", "M2", "M3", "M4"]]
        for c, (t, im, m) in enumerate(panels):
            ax = axs[r, c]; ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.axis("off"); ax.grid(False)
            if m is not None:
                ov = np.zeros((*m.shape, 4)); ov[m] = [0.92, 0.41, 0.2, 0.45]; ax.imshow(ov); ax.contour(g_, [0.5], colors="#ffd43b", linewidths=0.7)
                if t in MAIN: ax.set_title(f"{t}  Dice {dm[t][j]:.2f}".replace(".", ","), fontsize=8, color=COL[t], fontweight="bold")
                else: ax.set_title(t, fontsize=8, color=INK)
            elif r == 0: ax.set_title(t, fontsize=8, color=INK)
    fig.tight_layout(); savefig(fig, "fig_qualitative.png")
    # ---- bản đồ attention (M4, hạt 0)
    m = load("M4", 0); m.set_capture(True)
    sel = [idx_t[benign[2]], idx_t[malig[1]], idx_t[malig[7]]]
    fig, axs = plt.subplots(len(sel), 6, figsize=(11, 1.9 * len(sel)))
    for r, i in enumerate(sel):
        xt = torch.from_numpy(S["X_test"][i:i + 1].astype(np.float32) / 255.0)[None]
        with torch.no_grad(): m(xt)
        axs[r, 0].imshow(S["X_test"][i], cmap="gray"); axs[r, 0].contour(S["Y_test"][i], [0.5], colors="#ffd43b", linewidths=0.7)
        if r == 0: axs[r, 0].set_title("Ảnh + khối u", fontsize=8)
        for c, gate in enumerate(m.attention_gates()):
            a = torch.nn.functional.interpolate(gate.last_alpha, size=S["X_test"][i].shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            axs[r, c + 1].imshow(a, cmap="magma", vmin=0, vmax=1)
            if r == 0: axs[r, c + 1].set_title(f"AG tầng {4 - c}" + (" (sâu nhất)" if c == 0 else " (nông nhất)" if c == 3 else ""), fontsize=8)
        pr = predict(m, S["X_test"][i:i + 1])[0]; axs[r, 5].imshow(pr, cmap="gray", vmin=0, vmax=1)
        if r == 0: axs[r, 5].set_title("Xác suất đầu ra", fontsize=8)
        for a in axs[r]: a.axis("off"); a.grid(False)
    fig.tight_layout(); savefig(fig, "fig_attention_maps.png")
    # ---- ca khó nhất (M4, hạt 0): 5 ảnh Dice thấp nhất
    worst = np.argsort(dm["M4"])[:5]
    fig, axs = plt.subplots(2, 5, figsize=(10.5, 4.4))
    for c, j in enumerate(worst):
        i = idx_t[j]; x = S["X_test"][i]; pm_ = prob_t["M4", 0][i] > 0.5; g_ = S["Y_test"][i] > 0
        axs[0, c].imshow(x, cmap="gray"); axs[0, c].contour(g_, [0.5], colors="#ffd43b", linewidths=0.8)
        axs[0, c].set_title(f"{'Ác' if kind[j]=='malignant' else 'Lành'} tính · Dice {dm['M4'][j]:.2f}".replace(".", ","), fontsize=8)
        axs[1, c].imshow(x, cmap="gray"); ov = np.zeros((*pm_.shape, 4)); ov[pm_] = [0.92, 0.41, 0.2, 0.5]; axs[1, c].imshow(ov); axs[1, c].contour(g_, [0.5], colors="#ffd43b", linewidths=0.8)
        for a in axs[:, c]: a.axis("off"); a.grid(False)
    axs[0, 0].text(-0.08, 0.5, "Khối u thật (viền vàng)", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    axs[1, 0].text(-0.08, 0.5, "Dự đoán M4 (cam)", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout(); savefig(fig, "fig_errors.png")
