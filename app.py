"""SonoSeg – ứng dụng web phân đoạn khối u trên ảnh siêu âm vú bằng U-Net kết hợp Attention (Flask).

Chạy:   python app.py [--host 127.0.0.1] [--port 5000] [--open]

Thành phần
  - CSDL SQLite (db.py): ảnh, lần huấn luyện, kết quả đánh giá, lịch sử ca.
  - Bộ ảnh mẫu (dataset_tools.py): 420 ảnh siêu âm mô phỏng kèm mặt nạ trong data/sample/.
  - Huấn luyện nền + đánh giá (trainer.py).
  - Giao diện một trang (templates/index.html, static/).
Công cụ phục vụ nghiên cứu và minh họa, KHÔNG thay thế chẩn đoán của bác sĩ.
"""
import argparse, base64, csv, glob, io, json, os, threading, time, uuid, zipfile
from collections import OrderedDict

import numpy as np
import cv2
import imio
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

import data as D
import dataset_tools as T
import db
import metrics as MT
import models as M
import trainer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "weights", "final")
RESULTS = os.path.join(ROOT, "results")
BUILTIN = ("M1", "M2", "M3", "M4")
DESC = {
    "M1": "U-Net gốc: mã hóa – giải mã đối xứng với kết nối tắt; mốc so sánh.",
    "M2": "Attention U-Net: Attention Gate trên 4 kết nối tắt.",
    "M3": "U-Net + CBAM: chú ý kênh và không gian ở mọi khối tích chập.",
    "M4": "Mô hình đề xuất: Attention Gate + CBAM tầng đáy + giám sát sâu.",
}
MAX_BATCH = 100

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.json.ensure_ascii = False
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
_lock = threading.Lock()
_zips = OrderedDict()


# ================================================================== mô hình ==================================
def _eff():
    p = os.path.join(RESULTS, "results.json")
    return {e["key"]: e for e in json.load(open(p, encoding="utf8")).get("efficiency", [])} if os.path.exists(p) else {}


def list_models():
    eff, evals, out = _eff(), {}, []
    for e in db.q(f"SELECT model_id, split, dice FROM evals WHERE {ABL_SQL}"):
        evals.setdefault(e["model_id"], {})[e["split"]] = e["dice"]
    for k in BUILTIN:
        p = os.path.join(WEIGHTS, k + ".pt")
        if not os.path.exists(p):
            continue
        m, ck = trainer.load_model(k)
        out.append(dict(id=k, name=M.NAMES.get(ck["model"], k), type=ck["model"], source="builtin", size=ck["size"], base=ck["base"],
                        params=M.count_params(m), val_dice=round(float(ck["val_dice"]), 4), has_attention=bool(m.attention_gates()),
                        gflops=round(eff[k]["gflops"], 3) if k in eff else None, desc=DESC.get(k, ""), eval=evals.get(k, {})))
    for f in sorted(glob.glob(os.path.join(ROOT, "weights", "real", "*-STU.pt"))):
        k = os.path.basename(f)[:-3]
        m, ck = trainer.load_model(k)
        out.append(dict(id=k, name=f"{M.NAMES.get(ck['model'], ck['model'])} · ảnh thật", type=ck["model"], source="real", size=ck["size"], base=ck["base"],
                        params=M.count_params(m), val_dice=r(ck.get("val_dice"), 4), has_attention=bool(m.attention_gates()), gflops=None,
                        desc=f"Tinh chỉnh trên 42 ảnh siêu âm thật STU-Hospital từ trọng số {ck['model']}"
                             f"{' (công thức cải tiến R, có lật ngang khi suy luận)' if ck.get('tta') else ''}. Dice kiểm định chéo 5 phần: {str(r(ck.get('val_dice'), 3)).replace('.', ',')}.",
                        eval=evals.get(k, {})))
    for row in db.q("SELECT * FROM runs WHERE status IN ('done','stopped') AND weights IS NOT NULL ORDER BY id"):
        if not os.path.exists(os.path.join(ROOT, row["weights"])):
            continue
        out.append(dict(id=f"R{row['id']}", name=row["name"] or f"Lần huấn luyện #{row['id']}", type=row["model_type"], source="trained",
                        size=row["size"], base=row["base"], params=row["params"], val_dice=row["best_val_dice"], gflops=None,
                        has_attention=row["model_type"] in ("M2", "M4", "A4"), eval=evals.get(f"R{row['id']}", {}),
                        desc=f"Huấn luyện trên web: {row['model_type']}, {row['epochs']} epoch, {row['n_train']} ảnh, {row['size']}×{row['size']}."))
    return out


# ================================================================== ảnh ======================================
def b64png(arr, ext=".png", q=None):
    ok, buf = cv2.imencode(ext, arr, [cv2.IMWRITE_JPEG_QUALITY, q] if q else [])
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def decode_upload(fs, flags=cv2.IMREAD_GRAYSCALE):
    data = np.frombuffer(fs.read(), np.uint8)
    img = cv2.imdecode(data, flags) if data.size else None
    if img is None:
        raise ValueError(f"Không đọc được ảnh “{fs.filename}”. Hãy dùng PNG, JPG hoặc BMP.")
    if min(img.shape[:2]) < 16:
        raise ValueError("Ảnh quá nhỏ (tối thiểu 16×16 điểm ảnh).")
    return img


def fit_mask(g, shape):
    if g.shape != shape:
        g = cv2.resize(g.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return g


def get_input():
    """Ảnh đầu vào từ tệp tải lên hoặc từ CSDL (image_id). Trả (ảnh, mặt nạ chuẩn|None, nhãn, image_id)."""
    iid = request.form.get("image_id")
    if iid:
        row = db.q("SELECT * FROM images WHERE id=?", (int(iid),), one=True)
        if not row:
            raise ValueError("Ảnh không tồn tại trong CSDL.")
        img, m = T.read_item(row)
        if "gt" in request.files:
            m = fit_mask(decode_upload(request.files["gt"]) > 127, img.shape)
        return img, m, os.path.basename(row["filename"]), row["id"]
    if "image" not in request.files:
        raise ValueError("Thiếu ảnh đầu vào.")
    img = decode_upload(request.files["image"])
    gt = fit_mask(decode_upload(request.files["gt"]) > 127, img.shape) if "gt" in request.files else None
    return img, gt, request.form.get("label") or request.files["image"].filename or "ảnh tải lên", None


def letterbox_params(h, w, size):
    s = size / max(h, w); nh, nw = max(1, round(h * s)), max(1, round(w * s))
    return (size - nh) // 2, (size - nw) // 2, nh, nw


def infer(mid, img, thr=0.5, post=True, want_att=True):
    model, ck = trainer.load_model(mid); size = ck["size"]
    h, w = img.shape
    xt = torch.from_numpy(D.preprocess(img, size))[None, None]
    gates = model.attention_gates()
    with _lock:
        model.set_capture(want_att and bool(gates))
        t0 = time.perf_counter()
        with torch.no_grad():
            pr = torch.sigmoid(model(xt))
            if ck.get("tta"):                      # lật ngang khi suy luận (mô hình công thức R)
                model.set_capture(False)
                pr = (pr + torch.sigmoid(model(xt.flip(-1))).flip(-1)) / 2
            prob = pr[0, 0].numpy()
        ms = (time.perf_counter() - t0) * 1000
        maps = [torch.nn.functional.interpolate(g.last_alpha, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
                for g in gates] if (want_att and gates) else []
        model.set_capture(False)
    y0, x0, nh, nw = letterbox_params(h, w, size)
    back = lambda a: cv2.resize(a[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)
    prob_o = back(prob); mask = prob_o > thr
    if post and mask.any():
        mask = MT.largest_component(mask)
    return prob_o, mask, [back(a) for a in maps], ms


def overlay(img, mask, alpha=0.4, gt=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    m = mask.astype(bool); col = np.array([80, 120, 255], np.float32)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * col
    rgb = rgb.astype(np.uint8); t = max(1, round(max(img.shape) / 300))
    if gt is not None and gt.any():
        c, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, c, -1, (110, 230, 90), t)
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, c, -1, (40, 220, 255), t)
    return rgb


def heat(a, norm=True):
    a = np.clip((a - a.min()) / (np.ptp(a) + 1e-6), 0, 1) if norm else np.clip(a, 0, 1)
    return cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def describe(mask, gt=None, px_mm=None):
    f = MT.shape_features(mask, px_mm)
    out = dict(detected=bool(mask.any()), area_px=int(f["area_px"]), area_ratio=r(f["area_ratio"] * 100, 2),
               max_diameter_px=r(f["max_diameter_px"], 1), circularity=r(f["circularity"]),
               aspect_ratio=r(f["aspect_ratio"], 2), solidity=r(f["solidity"]))
    if px_mm:
        out.update(area_mm2=r(f.get("area_mm2"), 1), max_diameter_mm=r(f.get("max_diameter_mm"), 1))
    if gt is not None:
        g = gt.astype(bool); out["gt_present"] = bool(g.any())
        if g.any() or mask.any():
            out.update(dice=r(MT.dice(mask, g), 4), iou=r(MT.iou(mask, g), 4),
                       precision=r(MT.precision(mask, g), 4) if mask.any() else None,
                       recall=r(MT.recall(mask, g), 4) if g.any() else None, hd95_px=r(MT.hd95(mask, g), 2))
        else:
            out.update(dice=1.0, iou=1.0, precision=None, recall=None, hd95_px=None)
    return out


def params():
    f = request.form
    thr = min(0.95, max(0.05, float(f.get("threshold", 0.5))))
    post = f.get("post", "1") == "1"
    px = f.get("px_mm", "").strip().replace(",", ".")
    px_mm = float(px) if px else None
    if px_mm is not None and not (0 < px_mm < 10):
        raise ValueError("Kích thước điểm ảnh phải trong khoảng (0; 10) mm.")
    return thr, post, px_mm


def save_case(label, iid, mid, thr, met, ov):
    cid = db.insert("cases", dict(created_at=db.now(), image_label=label, image_id=iid, model_id=mid, threshold=thr,
                                  detected=int(met["detected"]), area_ratio=met["area_ratio"], diameter_px=met["max_diameter_px"],
                                  area_mm2=met.get("area_mm2"), diameter_mm=met.get("max_diameter_mm"), circularity=met["circularity"],
                                  solidity=met["solidity"], aspect_ratio=met["aspect_ratio"], dice=met.get("dice"), iou=met.get("iou"),
                                  precision=met.get("precision"), recall=met.get("recall"), hd95=met.get("hd95_px"), ms=met.get("ms")))
    th = cv2.resize(ov, (160, max(1, round(160 * ov.shape[0] / ov.shape[1]))), interpolation=cv2.INTER_AREA)
    rel = f"cases/{cid:06d}.jpg"; os.makedirs(db.path("cases"), exist_ok=True)
    imio.imwrite(db.path(rel), th, [cv2.IMWRITE_JPEG_QUALITY, 85]); db.update("cases", cid, dict(thumb=rel))
    return cid


# ================================================================== trang & tệp ===============================
@app.errorhandler(HTTPException)
def http_err(e):
    if request.path.startswith("/api/"):
        return jsonify(error="Tệp tải lên quá lớn (tối đa 64 MB)." if e.code == 413 else e.description), e.code
    return e


@app.errorhandler(ValueError)
def val_err(e):
    return jsonify(error=str(e)), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data/<path:name>")
def data_files(name):
    return send_from_directory(db.DATA, name, max_age=3600)


@app.route("/results/<path:name>")
def results_files(name):
    return send_from_directory(RESULTS, name, max_age=3600)


# ================================================================== API chung ==================================
@app.route("/api/health")
def api_health():
    return jsonify(status="ok", models=len(list_models()), torch=torch.__version__, threads=torch.get_num_threads(),
                   training=trainer.busy(), db=os.path.relpath(db.DB_PATH, ROOT))


@app.route("/api/models")
def api_models():
    return jsonify(list_models())


@app.route("/api/stats")
def api_stats():
    img = db.q("SELECT dataset, split, kind, COUNT(*) n FROM images GROUP BY dataset, split, kind")
    cases = db.q("SELECT COUNT(*) n, AVG(dice) dice, AVG(ms) ms, SUM(detected) det FROM cases", one=True)
    by_model = db.q("SELECT model_id, COUNT(*) n, AVG(dice) dice, AVG(ms) ms FROM cases GROUP BY model_id ORDER BY model_id")
    daily = db.q("SELECT substr(created_at,1,10) d, COUNT(*) n FROM cases GROUP BY d ORDER BY d DESC LIMIT 14")
    runs = db.q("SELECT status, COUNT(*) n FROM runs GROUP BY status")
    recent = db.q("SELECT id, created_at, image_label, model_id, dice, area_ratio, thumb FROM cases ORDER BY id DESC LIMIT 6")
    evals = db.q(f"SELECT model_id, split, dice, iou, ms FROM evals WHERE {ABL_SQL} ORDER BY model_id")
    return jsonify(images=img, cases=cases, cases_by_model=by_model, daily=daily[::-1], runs=runs, recent=recent, evals=evals,
                   models=len(list_models()), training=trainer.busy())


# ================================================================== phân đoạn ==================================
@app.route("/api/sample")
def api_sample():
    kind = request.args.get("type", "benign")
    if kind not in D.TYPES:
        return jsonify(error="Loại mẫu không hợp lệ."), 400
    domain = "B" if request.args.get("domain") == "B" else "A"
    seed = int(request.args.get("seed", np.random.randint(1, 10 ** 6)))
    img, gt, k = D.make_sample(seed, kind=kind, domain=domain, size=256)
    return jsonify(image=b64png(img), gt=b64png(gt * 255), kind=k, seed=seed, domain=domain)


def unc_img(img, prob):
    """Tô màu những điểm ảnh mô hình lưỡng lự (p gần 0,5) lên ảnh gốc; độ đậm theo 4p(1−p)."""
    u = (4 * prob * (1 - prob)).astype(np.float32)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.6
    col = cv2.applyColorMap((np.clip(u, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    a = np.clip(u * 1.4, 0, 1)[..., None]
    return (base * (1 - a) + col * a).astype(np.uint8)


def assess(img, prob, mask, thr, post, px_mm=None):
    """Độ tin cậy và vị trí – trình bày kết quả theo hướng thực tiễn.

    - prob_in   : xác suất trung bình bên trong vùng phân đoạn (mô hình “chắc” đến đâu)
    - stability : Dice giữa mặt nạ ở ngưỡng 0,3 và 0,7 – kết quả có phụ thuộc nhiều vào ngưỡng không
    - border    : tỉ lệ điểm ảnh lưỡng lự (0,3 < p < 0,7) so với diện tích vùng – biên rõ hay mờ
    """
    lo, hi = prob > 0.3, prob > 0.7
    if post:
        lo = MT.largest_component(lo) if lo.any() else lo
        hi = MT.largest_component(hi) if hi.any() else hi
    unsure = (prob > 0.3) & (prob < 0.7)
    out = dict(max_prob=r(prob.max()))
    if mask.any():
        stab = MT.dice(lo, hi) if (lo.any() or hi.any()) else 1.0
        pin = float(prob[mask].mean()); border = float(unsure.sum() / mask.sum())
        level = "high" if (stab >= 0.9 and pin >= 0.85) else "medium" if (stab >= 0.75 and pin >= 0.7) else "low"
        ys, xs = np.nonzero(mask); h, w = mask.shape
        cx, cy = xs.mean() / w, ys.mean() / h
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        horiz = "bên trái" if cx < 0.36 else "bên phải" if cx > 0.64 else "chính giữa"
        depth = "nông" if cy < 0.36 else "sâu" if cy > 0.64 else "vừa"
        out.update(prob_in=r(pin), stability=r(stab), border=r(border * 100, 1), level=level,
                   loc=dict(cx=r(cx * 100, 1), cy=r(cy * 100, 1), horiz=horiz, depth=depth, bw=bw, bh=bh,
                            bw_mm=r(bw * px_mm, 1) if px_mm else None, bh_mm=r(bh * px_mm, 1) if px_mm else None,
                            depth_mm=r(ys.min() * px_mm, 1) if px_mm else None))
    else:
        out.update(level="none", weak=bool(prob.max() > 0.3))
    return out


@app.route("/api/segment", methods=["POST"])
def api_segment():
    mid = request.form.get("model", "M4")
    thr, post, px_mm = params()
    img, gt, label, iid = get_input()
    prob, mask, maps, ms = infer(mid, img, thr, post)
    met = describe(mask, gt, px_mm); met["ms"] = round(ms, 1)
    conf = assess(img, prob, mask, thr, post, px_mm)
    ov = overlay(img, mask, gt=gt)
    cid = save_case(label, iid, mid, thr, met, ov) if request.form.get("save", "1") == "1" else None
    return jsonify(model=mid, case_id=cid, inference_ms=round(ms, 1), size=[int(img.shape[1]), int(img.shape[0])], threshold=thr,
                   label=label, image=b64png(img), mask=b64png((mask * 255).astype(np.uint8)), overlay=b64png(ov),
                   prob=b64png(heat(prob, norm=False)), attention=b64png(heat(np.mean(maps, 0))) if maps else None,
                   uncertainty=b64png(unc_img(img, prob)), confidence=conf,
                   gates=[b64png(heat(a)) for a in maps], gt=b64png((gt * 255).astype(np.uint8)) if gt is not None else None,
                   metrics=met)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    thr, post, px_mm = params()
    img, gt, _, _ = get_input()
    ids = [x for x in request.form.get("models", "").split(",") if x] or [m["id"] for m in list_models()]
    rows, votes = [], np.zeros(img.shape, np.float32)
    for mid in ids:
        _, mask, _, ms = infer(mid, img, thr, post, want_att=False)
        votes += mask
        name = next((m["name"] for m in list_models() if m["id"] == mid), mid)
        rows.append(dict(id=mid, name=name, inference_ms=round(ms, 1), overlay=b64png(overlay(img, mask, gt=gt)), metrics=describe(mask, gt, px_mm)))
    n = max(1, len(rows))
    frac = votes / n
    cons = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    hm = cv2.applyColorMap((frac * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    sel = votes > 0; cons[sel] = 0.35 * cons[sel] + 0.65 * hm[sel]
    union = int(sel.sum())
    return jsonify(rows=rows, consensus=b64png(cons.astype(np.uint8)), agreement=round(float(((votes == 0) | (votes == n)).mean()) * 100, 2),
                   full_agreement_in_union=round(float((votes == n).sum() / union * 100), 1) if union else 100.0)


@app.route("/api/robustness", methods=["POST"])
def api_robustness():
    thr, post, _ = params()
    kind = request.form.get("kind", "speckle")
    if kind not in trainer.LEVELS:
        raise ValueError("Loại nhiễu không hợp lệ.")
    img, gt, _, _ = get_input()
    ids = [x for x in request.form.get("models", "").split(",") if x] or list(BUILTIN)
    if request.form.get("mode") == "sweep":
        series = {}
        for mid in ids:
            vals = []
            for L in trainer.LEVELS[kind]:
                _, mask, _, _ = infer(mid, trainer.degrade(img, kind, L), thr, post, want_att=False)
                vals.append(r(MT.dice(mask, gt), 4) if gt is not None else r(mask.mean() * 100, 2))
            series[mid] = vals
        return jsonify(kind=kind, levels=trainer.LEVELS[kind], series=series, metric="dice" if gt is not None else "area")
    level = float(request.form.get("level", 0.2))
    deg = trainer.degrade(img, kind, level)
    rows = []
    for mid in ids:
        _, mask, _, _ = infer(mid, deg, thr, post, want_att=False)
        rows.append(dict(id=mid, overlay=b64png(overlay(deg, mask, gt=gt)), metrics=describe(mask, gt)))
    return jsonify(image=b64png(deg), kind=kind, level=level, rows=rows,
                   psnr=r(cv2.PSNR(img, deg), 2) if level not in (0.0, 1.0) else None)


# ================================================================== mô hình đề xuất: giải phẫu & ablation =========
# Ma trận ablation 2×2×2: (Attention Gate, CBAM tầng đáy, giám sát sâu) -> mã mô hình
ABL = {(0, 0, 0): "M1", (1, 0, 0): "M2", (0, 1, 0): "A3", (1, 1, 0): "A4",
       (0, 0, 1): "D1", (1, 0, 1): "D2", (0, 1, 1): "D3", (1, 1, 1): "M4"}
ABL_ONLY = ("A3", "A4", "D1", "D2", "D3")          # chỉ dùng cho trang ablation, không hiện ở danh sách mô hình
ABL_SQL = "model_id NOT IN (%s)" % ",".join(f"'{k}'" for k in ABL_ONLY)
# Bảng 4.x của báo cáo (thực nghiệm E3) – lấy từ results.json
ABL_REPORT_CODE = {"M1": "A0", "M2": "A1", "A3": "A3", "A4": "A4", "M4": "A5"}


def _has_w(k):
    return any(os.path.exists(os.path.join(ROOT, "weights", d, k + ".pt")) for d in ("final", "ablation"))


def _prep(img, size):
    return torch.from_numpy(D.preprocess(img, size))[None, None]


def _back_fn(img, size):
    h, w = img.shape
    y0, x0, nh, nw = letterbox_params(h, w, size)
    def back(t):
        a = t.detach().float()
        a = a[None, None] if a.dim() == 2 else a
        a = torch.nn.functional.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
        return cv2.resize(a[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def _blend(img, a, alpha=0.55):
    """Phủ bản đồ nhiệt (đã chuẩn hóa 0–1) lên ảnh xám."""
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    hm = heat(a, norm=False).astype(np.float32)
    return ((1 - alpha) * base + alpha * hm).astype(np.uint8)


def _norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def _feat(t):
    """Đặc trưng [1,C,h,w] -> bản đồ năng lượng trung bình |x| theo kênh."""
    return t[0].abs().mean(0)


@app.route("/api/anatomy", methods=["POST"])
def api_anatomy():
    """Chạy một ảnh qua mô hình và trả về từng bước trung gian: mã hóa, CBAM, Attention Gate, giám sát sâu, đầu ra."""
    mid = request.form.get("model", "M4")
    thr, post, _ = params()
    img, gt, label, _ = get_input()
    model, ck = trainer.load_model(mid); size = ck["size"]
    back = _back_fn(img, size)
    with _lock:
        t0 = time.perf_counter()
        tr = M.trace(model, _prep(img, size))
        ms = (time.perf_counter() - t0) * 1000
    prob = back(torch.sigmoid(tr["out"]))
    mask = prob > thr
    if post and mask.any():
        mask = MT.largest_component(mask)
    ref = gt if (gt is not None and gt.any()) else (mask if mask.any() else None)   # vùng khối u để đo độ tập trung
    res = lambda t: f"{t.shape[-1]}×{t.shape[-2]}"

    def fmap(t):
        a = _norm(back(_feat(t)))
        return b64png(_blend(img, a, 0.6))

    enc = [dict(level=i + 1, res=res(t), ch=int(t.shape[1]), img=fmap(t)) for i, t in enumerate(tr["enc"])]
    bott = dict(res=res(tr["bott"]), ch=int(tr["bott"].shape[1]), before=fmap(tr["bott"]), after=fmap(tr["bott_out"]))
    cbam = None
    if tr["cbam"] is not None:
        w = tr["cbam"]["channel"].numpy()
        sp = back(tr["cbam"]["spatial"])
        cbam = dict(channel=[round(float(v), 4) for v in w], n=len(w), mean=r(w.mean()), sd=r(w.std()),
                    hi=int((w > 0.5).sum()), spatial=b64png(_blend(img, _norm(sp), 0.6)),
                    spatial_res=res(tr["cbam"]["spatial"][None]))
    gates = []
    for k, a in tr["gates"]:
        A = back(a)
        g = dict(level=model.depth - k, res=f"{a.shape[-1]}×{a.shape[-2]}", img=b64png(_blend(img, np.clip(A, 0, 1), 0.6)),
                 mean=r(A.mean()))
        if ref is not None:
            fin, fout = float(A[ref].mean()), float(A[~ref].mean()) if (~ref).any() else 0.0
            g.update(inside=r(fin), outside=r(fout), focus=r(fin / max(fout, 1e-3), 2))
        gates.append(g)
    heads = []
    for k, t in enumerate(tr["aux"]):
        p = back(torch.sigmoid(t[0, 0])); m_ = p > thr
        src = tr["dec"][k]
        heads.append(dict(name=f"Đầu ra phụ {k + 1}", res=res(src), weight=[0.5, 0.25, 0.125][k] if k < 3 else None,
                          prob=b64png(heat(p, norm=False)), overlay=b64png(overlay(img, m_, gt=gt)),
                          dice=r(MT.dice(m_, gt), 4) if gt is not None else None, area=r(m_.mean() * 100, 2)))
    heads.append(dict(name="Đầu ra chính", res=res(tr["dec"][-1]), weight=1.0, main=True, prob=b64png(heat(prob, norm=False)),
                      overlay=b64png(overlay(img, mask, gt=gt)), dice=r(MT.dice(mask, gt), 4) if gt is not None else None,
                      area=r(mask.mean() * 100, 2)))
    cfg = M.CONFIGS.get(ck["model"], {})
    return jsonify(model=mid, type=ck["model"], name=M.NAMES.get(ck["model"], mid), size=size, ms=round(ms, 1), label=label,
                   flags=dict(ag=bool(cfg.get("attn_gate")), cbam=cfg.get("bottleneck") == "cbam" or bool(cfg.get("cbam_all")),
                              ds=bool(cfg.get("deep_sup"))),
                   image=b64png(img), gt=b64png((gt * 255).astype(np.uint8)) if gt is not None else None,
                   input_res=f"{size}×{size}", enc=enc, bott=bott, cbam=cbam, gates=gates, heads=heads,
                   metrics=describe(mask, gt))


def _abl_rows():
    rep = {}
    p = os.path.join(RESULTS, "results.json")
    if os.path.exists(p):
        for a in json.load(open(p, encoding="utf8")).get("ablation", []):
            rep[a["code"]] = a
    ev = {}
    for e in db.q("SELECT model_id, split, dice, iou, hd95, ms FROM evals"):
        ev.setdefault(e["model_id"], {})[e["split"]] = e
    rows = []
    for (ag, cb, ds), k in ABL.items():
        if not _has_w(k):
            continue
        m, ck = trainer.load_model(k)
        code = ABL_REPORT_CODE.get(k)
        rows.append(dict(id=k, ag=ag, cbam=cb, ds=ds, name=M.NAMES.get(k, k), params=M.count_params(m),
                         val_dice=r(ck["val_dice"], 4), test=ev.get(k, {}).get("test"), ext=ev.get(k, {}).get("ext"),
                         report=dict(code=code, dice=r(rep[code]["dice"], 4), hd95=r(rep[code]["hd95"], 3)) if code in rep else None))
    return rows


@app.route("/api/ablation")
def api_ablation():
    """Bảng ablation trên tập kiểm tra của CSDL (tự đánh giá các biến thể chưa có kết quả)."""
    for row in _abl_rows():
        for sp in ("test", "ext"):
            if row[sp] is None:
                trainer.evaluate(row["id"], sp)
    return jsonify(_abl_rows())


@app.route("/api/ablation/image", methods=["POST"])
def api_ablation_image():
    """Phân đoạn một ảnh bằng cả 8 tổ hợp; kèm bản đồ khác biệt giữa mô hình được chọn và mô hình đối chứng."""
    thr, post, _ = params()
    img, gt, _, _ = get_input()
    ref_id = request.form.get("ref", "M1")
    rows, masks = [], {}
    for (ag, cb, ds), k in ABL.items():
        if not _has_w(k):
            continue
        _, mask, _, ms = infer(k, img, thr, post, want_att=False)
        masks[k] = mask
        rows.append(dict(id=k, ag=ag, cbam=cb, ds=ds, ms=round(ms, 1), overlay=b64png(overlay(img, mask, gt=gt)),
                         metrics=describe(mask, gt)))
    diffs = {}
    if gt is not None and ref_id in masks:
        g = gt.astype(bool); R = masks[ref_id]
        for k, P in masks.items():
            fixed = (P == g) & (R != g)      # mô hình k đúng, đối chứng sai
            broke = (P != g) & (R == g)      # mô hình k sai, đối chứng đúng
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.55
            vis[fixed] = (60, 200, 60); vis[broke] = (60, 60, 230)
            c, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, c, -1, (255, 255, 255), max(1, round(max(img.shape) / 300)))
            diffs[k] = dict(img=b64png(vis.astype(np.uint8)), fixed=int(fixed.sum()), broke=int(broke.sum()))
    return jsonify(image=b64png(img), has_gt=gt is not None, ref=ref_id, rows=rows, diffs=diffs)


def _stem(name):
    s = os.path.splitext(os.path.basename(name))[0]
    for suf in ("_mask", "-mask", "_gt", "_label"):
        if s.lower().endswith(suf):
            return s[: -len(suf)], True
    return s, False


@app.route("/api/batch", methods=["POST"])
def api_batch():
    mid = request.form.get("model", "M4")
    thr, post, px_mm = params()
    files = request.files.getlist("images")
    if not files:
        raise ValueError("Chưa chọn ảnh.")
    imgs, gts = {}, {}
    for fs in files:
        stem, is_mask = _stem(fs.filename)
        (gts if is_mask else imgs)[stem] = fs
    if len(imgs) > MAX_BATCH:
        raise ValueError(f"Tối đa {MAX_BATCH} ảnh mỗi lần.")
    rows, zbuf = [], io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for stem in sorted(imgs):
            try:
                img = decode_upload(imgs[stem])
                gt = fit_mask(decode_upload(gts[stem]) > 127, img.shape) if stem in gts else None
                _, mask, _, ms = infer(mid, img, thr, post, want_att=False)
                met = describe(mask, gt, px_mm); ov = overlay(img, mask, gt=gt)
                z.writestr(f"masks/{stem}_pred.png", cv2.imencode(".png", (mask * 255).astype(np.uint8))[1].tobytes())
                z.writestr(f"overlays/{stem}_overlay.png", cv2.imencode(".png", ov)[1].tobytes())
                th = cv2.resize(ov, (160, max(1, round(160 * ov.shape[0] / ov.shape[1]))), interpolation=cv2.INTER_AREA)
                rows.append(dict(name=stem, ok=True, inference_ms=round(ms, 1), thumb=b64png(th, ".jpg", 85), metrics=met))
            except ValueError as e:
                rows.append(dict(name=stem, ok=False, error=str(e)))
        cols = ["name", "detected", "area_ratio", "max_diameter_px", "circularity", "aspect_ratio", "solidity",
                "area_mm2", "max_diameter_mm", "dice", "iou", "precision", "recall", "hd95_px", "inference_ms"]
        s = io.StringIO(); w = csv.writer(s); w.writerow(cols)
        for row in rows:
            if row["ok"]:
                m = row["metrics"]
                w.writerow([row["name"]] + ["" if m.get(c) is None else m.get(c) for c in cols[1:-1]] + [row["inference_ms"]])
        z.writestr("ket_qua.csv", "﻿" + s.getvalue())
    ok = [x for x in rows if x["ok"]]
    dices = [x["metrics"]["dice"] for x in ok if x["metrics"].get("dice") is not None]
    summary = dict(n=len(rows), ok=len(ok), with_gt=len(dices), detected=sum(x["metrics"]["detected"] for x in ok),
                   mean_dice=r(np.mean(dices), 4) if dices else None, std_dice=r(np.std(dices), 4) if dices else None,
                   mean_ms=r(np.mean([x["inference_ms"] for x in ok]), 1) if ok else None,
                   dices=dices, areas=[x["metrics"]["area_ratio"] for x in ok])
    zid = uuid.uuid4().hex[:12]; _zips[zid] = zbuf.getvalue()
    while len(_zips) > 5:
        _zips.popitem(last=False)
    return jsonify(model=mid, rows=rows, summary=summary, zip=f"/api/batch/{zid}.zip")


@app.route("/api/batch/<zid>.zip")
def api_batch_zip(zid):
    if zid not in _zips:
        return jsonify(error="Kết quả đã hết hạn, hãy chạy lại."), 404
    return send_file(io.BytesIO(_zips[zid]), mimetype="application/zip", as_attachment=True, download_name=f"ket_qua_{zid}.zip")


# ================================================================== CSDL ảnh ===================================
@app.route("/api/images")
def api_images():
    a = request.args
    where, args = ["1=1"], []
    for k in ("dataset", "split", "kind", "domain"):
        if a.get(k):
            where.append(f"{k}=?"); args.append(a[k])
    page, per = max(1, int(a.get("page", 1))), min(200, max(1, int(a.get("per", 48))))
    total = db.q(f"SELECT COUNT(*) n FROM images WHERE {' AND '.join(where)}", args, one=True)["n"]
    rows = db.q(f"SELECT * FROM images WHERE {' AND '.join(where)} ORDER BY id LIMIT ? OFFSET ?", args + [per, (page - 1) * per])
    for r_ in rows:
        r_["url"] = "/data/" + r_["filename"]; r_["mask_url"] = "/data/" + r_["mask"] if r_["mask"] else None
    return jsonify(total=total, page=page, per=per, rows=rows)


@app.route("/api/datasets")
def api_datasets():
    rows = db.q("SELECT dataset, split, kind, domain, COUNT(*) n, AVG(tumor_ratio) tr FROM images GROUP BY dataset, split, kind, domain")
    return jsonify(rows)


@app.route("/api/images/upload", methods=["POST"])
def api_images_upload():
    files = request.files.getlist("images")
    if not files:
        raise ValueError("Chưa chọn ảnh.")
    kind = request.form.get("kind", "unknown")
    imgs, gts = {}, {}
    for fs in files:
        stem, is_mask = _stem(fs.filename)
        (gts if is_mask else imgs)[stem] = fs
    ids = []
    for stem in sorted(imgs)[:MAX_BATCH]:
        img = decode_upload(imgs[stem])
        gt = fit_mask(decode_upload(gts[stem]) > 127, img.shape) if stem in gts else None
        ids.append(T.add_upload(img, gt, kind, stem))
    return jsonify(added=len(ids), ids=ids)


@app.route("/api/images/<int:iid>", methods=["DELETE"])
def api_images_delete(iid):
    row = db.q("SELECT * FROM images WHERE id=?", (iid,), one=True)
    if not row:
        return jsonify(error="Không tìm thấy ảnh."), 404
    if row["source"] != "upload":
        return jsonify(error="Chỉ xóa được ảnh do người dùng tải lên; bộ ảnh mẫu được giữ nguyên."), 400
    for p in (row["filename"], row["mask"]):
        if p and os.path.exists(db.path(p)):
            os.remove(db.path(p))
    db.x("DELETE FROM images WHERE id=?", (iid,))
    return jsonify(ok=True)


@app.route("/api/images/<int:iid>/split", methods=["POST"])
def api_images_split(iid):
    sp = request.json.get("split") if request.is_json else request.form.get("split")
    if sp not in ("train", "val", "test", "upload"):
        raise ValueError("Tập không hợp lệ.")
    row = db.q("SELECT * FROM images WHERE id=?", (iid,), one=True)
    if not row or not row["mask"]:
        raise ValueError("Ảnh cần có mặt nạ chuẩn để đưa vào tập huấn luyện/đánh giá.")
    db.update("images", iid, dict(split=sp, dataset="sample" if sp != "upload" and row["source"] == "upload" else row["dataset"]))
    return jsonify(ok=True)


# ================================================================== huấn luyện & đánh giá =======================
@app.route("/api/train", methods=["POST"])
def api_train():
    j = request.get_json(force=True, silent=True) or {}
    mt = j.get("model_type", "M4")
    if mt not in M.CONFIGS:
        raise ValueError("Kiểu mô hình không hợp lệ.")
    cfg = dict(model_type=mt, epochs=int(np.clip(int(j.get("epochs", 8)), 1, 200)), bs=int(np.clip(int(j.get("bs", 16)), 2, 64)),
               lr=float(np.clip(float(j.get("lr", 1e-3)), 1e-5, 1e-1)), size=int(j.get("size", 96)), base=int(j.get("base", 8)),
               n_train=int(np.clip(int(j.get("n_train", 240)), 8, 100000)), augment=bool(j.get("augment", True)),
               dataset=j.get("dataset", "sample"))
    if cfg["size"] not in (64, 96, 128, 160, 192, 256) or cfg["base"] not in (4, 8, 16, 32):
        raise ValueError("Kích thước ảnh hoặc số kênh cơ sở không hợp lệ.")
    cfg["name"] = (j.get("name") or f"{mt} · {cfg['epochs']} epoch · {cfg['size']}px").strip()[:60]
    try:
        rid = trainer.start(cfg)
    except RuntimeError as e:
        return jsonify(error=str(e)), 409
    return jsonify(run_id=rid)


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    return jsonify(stopped=trainer.stop())


@app.route("/api/runs")
def api_runs():
    rows = db.q("SELECT * FROM runs ORDER BY id DESC")
    for x in rows:
        x["history"] = json.loads(x["history"] or "[]")
    return jsonify(runs=rows, busy=trainer.busy())


@app.route("/api/runs/<int:rid>", methods=["DELETE"])
def api_runs_delete(rid):
    row = db.q("SELECT * FROM runs WHERE id=?", (rid,), one=True)
    if not row:
        return jsonify(error="Không tìm thấy."), 404
    if row["status"] in ("running", "queued") and trainer.busy():
        return jsonify(error="Không xóa được tác vụ đang chạy."), 409
    if row["weights"] and os.path.exists(os.path.join(ROOT, row["weights"])):
        os.remove(os.path.join(ROOT, row["weights"]))
    trainer.forget(f"R{rid}")
    db.x("DELETE FROM evals WHERE model_id=?", (f"R{rid}",)); db.x("DELETE FROM runs WHERE id=?", (rid,))
    return jsonify(ok=True)


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    j = request.get_json(force=True, silent=True) or {}
    split = j.get("split", "test")
    if split not in ("test", "ext", "val", "real"):
        raise ValueError("Tập đánh giá không hợp lệ.")
    ids = j.get("models") or [m["id"] for m in list_models()]
    if split == "real":                       # mô hình đã tinh chỉnh trên chính 42 ảnh thật: không đánh giá lại trên tập đó
        ids = [i for i in ids if not i.endswith("-STU")]
    out = [trainer.evaluate(mid, split) for mid in ids]
    return jsonify(out)


@app.route("/api/evals")
def api_evals():
    rows = db.q(f"SELECT * FROM evals WHERE {ABL_SQL} ORDER BY model_id, split")
    for x in rows:
        x["by_kind"] = json.loads(x["by_kind"] or "{}"); x["per_image"] = json.loads(x["per_image"] or "[]")
    return jsonify(rows)


@app.route("/api/evaluate/robustness", methods=["POST"])
def api_eval_rob():
    j = request.get_json(force=True, silent=True) or {}
    kind = j.get("kind", "speckle")
    if kind not in trainer.LEVELS:
        raise ValueError("Loại nhiễu không hợp lệ.")
    return jsonify(trainer.robustness(j.get("models") or list(BUILTIN), kind, n=int(j.get("n", 40))))


# ================================================================== lịch sử ca =================================
@app.route("/api/cases")
def api_cases():
    a = request.args
    where, args = ["1=1"], []
    if a.get("model"):
        where.append("model_id=?"); args.append(a["model"])
    if a.get("q"):
        where.append("image_label LIKE ?"); args.append(f"%{a['q']}%")
    page, per = max(1, int(a.get("page", 1))), min(200, int(a.get("per", 20)))
    total = db.q(f"SELECT COUNT(*) n FROM cases WHERE {' AND '.join(where)}", args, one=True)["n"]
    rows = db.q(f"SELECT * FROM cases WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ? OFFSET ?", args + [per, (page - 1) * per])
    return jsonify(total=total, page=page, per=per, rows=rows)


@app.route("/api/cases/<int:cid>", methods=["DELETE"])
def api_case_delete(cid):
    row = db.q("SELECT thumb FROM cases WHERE id=?", (cid,), one=True)
    if row and row["thumb"] and os.path.exists(db.path(row["thumb"])):
        os.remove(db.path(row["thumb"]))
    db.x("DELETE FROM cases WHERE id=?", (cid,))
    return jsonify(ok=True)


@app.route("/api/cases/clear", methods=["POST"])
def api_case_clear():
    for row in db.q("SELECT thumb FROM cases"):
        if row["thumb"] and os.path.exists(db.path(row["thumb"])):
            os.remove(db.path(row["thumb"]))
    db.x("DELETE FROM cases")
    return jsonify(ok=True)


@app.route("/api/cases.csv")
def api_cases_csv():
    rows = db.q("SELECT * FROM cases ORDER BY id")
    cols = ["id", "created_at", "image_label", "model_id", "threshold", "detected", "area_ratio", "diameter_px", "area_mm2", "diameter_mm",
            "circularity", "solidity", "aspect_ratio", "dice", "iou", "precision", "recall", "hd95", "ms"]
    s = io.StringIO(); w = csv.writer(s); w.writerow(cols)
    for x in rows:
        w.writerow(["" if x[c] is None else x[c] for c in cols])
    return Response("﻿" + s.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=lich_su_ca.csv"})


@app.route("/api/real")
def api_real():
    p = os.path.join(RESULTS, "real_results.json")
    if not os.path.exists(p):
        return jsonify({})
    j = json.load(open(p, encoding="utf8")); j.pop("per", None)
    for v in j.get("zero_shot", {}).values():
        v.pop("per", None)
    return jsonify(j)


@app.route("/api/results")
def api_results():
    p = os.path.join(RESULTS, "results.json")
    return jsonify(json.load(open(p, encoding="utf8"))) if os.path.exists(p) else jsonify({})


# ================================================================== khởi động ==================================
def bootstrap(log=print):
    db.init()
    n = T.build_sample(log=log)
    if not db.q("SELECT COUNT(*) n FROM evals", one=True)["n"]:
        log("Đánh giá ban đầu 4 mô hình có sẵn trên tập kiểm tra…")
        for k in BUILTIN:
            if os.path.exists(os.path.join(WEIGHTS, k + ".pt")):
                trainer.evaluate(k, "test"); trainer.evaluate(k, "ext")
    if os.environ.get("SONOSEG_NO_STU") is None and not db.q("SELECT COUNT(*) n FROM images WHERE dataset='stu'", one=True)["n"]:
        try:                                   # 42 ảnh siêu âm vú thật (STU-Hospital); bỏ qua nếu không có mạng
            T.import_stu(log=log)
        except Exception as e:  # noqa: BLE001
            log(f"Chưa nhập được ảnh thật STU-Hospital: {e}")
    have = {(e["model_id"], e["split"]) for e in db.q("SELECT model_id, split FROM evals")}
    if db.q("SELECT COUNT(*) n FROM images WHERE dataset='stu'", one=True)["n"]:
        for k in BUILTIN:
            if (k, "real") not in have and os.path.exists(os.path.join(WEIGHTS, k + ".pt")):
                trainer.evaluate(k, "real")
    for k in ABL_ONLY:                       # biến thể ablation (nếu có trọng số)
        for sp in ("test", "ext"):
            if _has_w(k) and (k, sp) not in have:
                trainer.evaluate(k, sp)
    db.x("UPDATE runs SET status='failed', message='Máy chủ khởi động lại khi đang huấn luyện' WHERE status IN ('running','queued')")
    return n


with app.app_context():
    bootstrap(log=lambda *a: None) if os.environ.get("SONOSEG_NO_BOOT") is None else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SonoSeg – phân đoạn khối u siêu âm vú")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true"); ap.add_argument("--open", action="store_true", help="tự mở trình duyệt")
    a = ap.parse_args()
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"SonoSeg đang chạy tại {url}  (Ctrl+C để dừng)")
    if a.open:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)
