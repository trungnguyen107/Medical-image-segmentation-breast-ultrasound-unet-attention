"""Nhập và quản lý ẢNH SIÊU ÂM THẬT (bình thường / lành tính / ác tính) cho SonoSeg.

Nguyên tắc
  - "Ảnh thật" = mọi ảnh không phải bộ mô phỏng (dataset 'sample') và không phải ảnh người dùng tải lên.
  - Mỗi ảnh được băm cảm nhận (dHash 64 bit) để LOẠI ẢNH TRÙNG LẶP, kể cả giữa các bộ khác nhau
    (BUSI có khoảng 19% ảnh trùng; trùng giữa tập huấn luyện và kiểm tra làm Dice bị thổi phồng).
  - Chia tập theo BỆNH NHÂN khi có mã bệnh nhân (mọi ảnh của một người nằm cùng một tập), phân tầng theo loại.
  - Bộ giữ riêng (--holdout) dùng làm tập kiểm tra ngoài (split 'ext'), không bao giờ vào huấn luyện.
  - Mục tiêu số lượng (mặc định 10.000 ảnh thật) chỉ để theo dõi tiến độ; phần mềm không sinh ảnh để "đủ số".

Phần mềm KHÔNG tự tải các bộ dữ liệu: hãy tải thủ công từ trang gốc, đọc giấy phép, rồi nhập bằng
  python manage.py import-real <tên bộ> <thư mục> [--manifest tệp.csv] [--holdout]
"""
import csv, glob, os, re
from collections import defaultdict

import cv2
import numpy as np

import db
import imio

TARGET_REAL = int(os.environ.get("SONOSEG_REAL_TARGET", "10000"))
KINDS = ("normal", "benign", "malignant")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Thông tin trích dẫn / giấy phép hiển thị trên trang "Cơ sở dữ liệu ảnh". Số ảnh là số công bố trong bài gốc.
# Giấy phép phải được kiểm tra lại trên trang gốc trước khi dùng (xem DATASETS.md).
REGISTRY = {
    "busi": dict(name="BUSI (Baheya, Ai Cập)", cite="Al-Dhabyani et al., Data in Brief 28, 2020", license="Chưa rõ – đọc trang gốc", published=780,
                 note="133 bình thường / 437 lành / 210 ác. ~19% ảnh trùng; nên dùng bản đã làm sạch."),
    "busbra": dict(name="BUS-BRA (Brazil)", cite="Gómez-Flores et al., Medical Physics, 2024", license="CC BY 4.0", published=1875,
                   note="1.064 bệnh nhân; chỉ lành tính và ác tính, không có ảnh bình thường."),
    "uclm": dict(name="BUS-UCLM (Tây Ban Nha)", cite="Vallez et al., Scientific Data, 2025", license="CC BY 4.0", published=683,
                 note="38 bệnh nhân: 419 bình thường / 174 lành / 90 ác. Nhiều ảnh cùng một người, bắt buộc chia theo bệnh nhân."),
    "breast": dict(name="BrEaST-Lesions-USG (Ba Lan)", cite="Pawłowska et al., Scientific Data, 2024", license="CC BY 4.0", published=270,
                   note="256 bệnh nhân, mặt nạ và mô tả BI-RADS."),
    "busis": dict(name="BUSIS", cite="Zhang et al., 2022", license="Theo yêu cầu", published=562, note=""),
    "udiat": dict(name="UDIAT / Dataset B (Tây Ban Nha)", cite="Yap et al., IEEE JBHI, 2018", license="Đăng ký với tác giả", published=163,
                  note="Nên giữ riêng làm tập kiểm tra ngoài (--holdout)."),
    "stu": dict(name="STU-Hospital (Shantou, Trung Quốc)", cite="Zhuang et al., PLoS ONE 14(8), 2019", license="Chưa ghi giấy phép", published=42,
                note="Chỉ dùng kiểm tra (tập 'real'), không dùng huấn luyện trên web."),
}

_CLASS_WORDS = [("malignant", ("malig", "ác", "ac tinh", "carcinoma", "cancer")),
                ("benign", ("benign", "lành", "lanh", "fibroadenoma", "cyst")),
                ("normal", ("normal", "bình thường", "binh thuong", "healthy"))]
_ALIASES = dict(image=("image", "img", "image_path", "file", "filename", "path", "id"),
                mask=("mask", "mask_path", "segmentation", "label_path", "gt"),
                label=("label", "class", "pathology", "category", "diagnosis", "type"),
                patient=("patient", "patient_id", "case", "case_id", "subject", "subject_id"))


def norm_label(s):
    t = (s or "").strip().lower()
    for k, words in _CLASS_WORDS:
        if any(w in t for w in words):
            return k
    return None


# ---------------------------------------------------------------- băm cảm nhận ----------------------------
def dhash(img):
    """dHash 64 bit (9×8) dưới dạng số nguyên – bền với đổi kích thước và nén JPEG nhẹ."""
    g = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


class HashIndex:
    """Tìm ảnh gần trùng (khoảng cách Hamming ≤ tol) nhanh bằng chỉ mục nguyên lý chuồng bồ câu:
    chia 64 bit thành tol+1 đoạn; hai băm cách nhau ≤ tol bit thì ít nhất một đoạn trùng khớp."""

    def __init__(self, tol=4):
        self.tol, self.parts = tol, tol + 1
        self.w = 64 // self.parts
        self.buckets = [defaultdict(list) for _ in range(self.parts)]

    def _chunks(self, h):
        for i in range(self.parts):
            lo = i * self.w
            hi = 64 if i == self.parts - 1 else lo + self.w
            yield i, (h >> lo) & ((1 << (hi - lo)) - 1)

    def find(self, h):
        for i, c in self._chunks(h):
            for other, tag in self.buckets[i].get(c, ()):
                if hamming(h, other) <= self.tol:
                    return tag
        return None

    def add(self, h, tag):
        for i, c in self._chunks(h):
            self.buckets[i][c].append((h, tag))


def _existing_index(tol):
    ix = HashIndex(tol)
    for r in db.q("SELECT dataset, filename, phash FROM images WHERE phash IS NOT NULL AND dataset NOT IN ('sample','uploads')"):
        ix.add(int(r["phash"], 16), f"{r['dataset']}/{os.path.basename(r['filename'])}")
    return ix


# ---------------------------------------------------------------- quét thư mục / bảng kê ----------------
def _is_img(f):
    return f.lower().endswith(IMG_EXT)


def _scan_folders(root):
    """Ảnh + mặt nạ theo quy ước <tên>.png / <tên>_mask*.png; nhãn lấy từ tên thư mục cha."""
    items = []
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if not _is_img(f) or "_mask" in f.lower() or f.lower().startswith("mask_"):
                continue
            p = os.path.join(dp, f); stem = os.path.splitext(p)[0]
            masks = sorted(m for m in glob.glob(glob.escape(stem) + "_mask*") if _is_img(m))
            parts = os.path.relpath(p, root).replace("\\", "/").split("/")[:-1]
            label = next((norm_label(x) for x in reversed(parts) if norm_label(x)), None)
            items.append(dict(image=p, masks=masks, label=label, patient=None))
    return items


def _scan_manifest(root, manifest):
    with open(manifest, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh); cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        pick = {k: next((cols[a] for a in al if a in cols), None) for k, al in _ALIASES.items()}
        if not pick["image"]:
            raise ValueError(f"Bảng kê thiếu cột ảnh (một trong: {', '.join(_ALIASES['image'])}). Cột hiện có: {list(cols)}")
        index = None; items = []
        for row in rd:
            ref = (row[pick["image"]] or "").strip()
            if not ref:
                continue
            def resolve(v):
                nonlocal index
                if not v:
                    return None
                p = v if os.path.isabs(v) else os.path.join(root, v)
                if os.path.exists(p):
                    return p
                if index is None:                      # cột ghi mã (không có đường dẫn/đuôi): tìm theo tên tệp
                    index = {}
                    for dp, _, fs in os.walk(root):
                        for f in fs:
                            if _is_img(f):
                                index.setdefault(os.path.splitext(f)[0].lower(), os.path.join(dp, f))
                return index.get(os.path.splitext(os.path.basename(v))[0].lower())
            ip = resolve(ref)
            if not ip:
                continue
            mp = resolve((row[pick["mask"]] or "").strip()) if pick["mask"] else None
            if not mp:                                  # thử quy ước tên: mask_<mã> hoặc <mã>_mask
                stem = os.path.splitext(os.path.basename(ip))[0]
                for cand in (stem + "_mask", "mask_" + stem, stem.replace("bus_", "mask_")):
                    mp = resolve(cand)
                    if mp:
                        break
            lab = norm_label(row[pick["label"]]) if pick["label"] else None
            items.append(dict(image=ip, masks=[mp] if mp else [], label=lab, patient=(row[pick["patient"]].strip() or None) if pick["patient"] else None))
    return items


# ---------------------------------------------------------------- chia theo bệnh nhân ---------------------
def assign_splits(items, ratios=(0.7, 0.1, 0.2), seed=0):
    """Gán 'train/val/test' cho từng mục theo NHÓM (bệnh nhân hoặc ảnh lẻ), phân tầng theo loại chiếm đa số của nhóm."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it["patient"] or f"_img{i}"].append(i)
    rng = np.random.default_rng(seed)
    by_kind = defaultdict(list)
    for g, idx in groups.items():
        kinds = [items[i]["label"] or "unknown" for i in idx]
        by_kind[max(set(kinds), key=kinds.count)].append(idx)
    out = {}
    for kind, gl in by_kind.items():
        order = rng.permutation(len(gl)); total = sum(len(gl[o]) for o in order); acc = 0
        for o in order:
            frac = acc / max(total, 1)
            sp = "train" if frac < ratios[0] else "val" if frac < ratios[0] + ratios[1] else "test"
            for i in gl[o]:
                out[i] = sp
            acc += len(gl[o])
    return [out[i] for i in range(len(items))]


# ---------------------------------------------------------------- nhập ---------------------------------
def _save(rel, arr):
    p = db.path(rel); os.makedirs(os.path.dirname(p), exist_ok=True); imio.imwrite(p, arr)


def _safe(s):
    return re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")[:60] or "img"


def import_real(dataset, root, manifest=None, holdout=False, ratios=(0.7, 0.1, 0.2), seed=0, dup_tol=4,
                require_mask=True, default_label=None, license=None, log=print):
    """Nhập một bộ ảnh siêu âm thật. Trả về dict thống kê (đã nhập, trùng lặp, bỏ qua vì thiếu mặt nạ/nhãn)."""
    db.init()
    if not os.path.isdir(root):
        raise ValueError(f"Không thấy thư mục: {root}")
    dataset = _safe(dataset).lower()
    if dataset in ("sample", "uploads"):
        raise ValueError("Tên bộ 'sample'/'uploads' dành riêng cho ảnh mô phỏng và ảnh tải lên.")
    items = _scan_manifest(root, manifest) if manifest else _scan_folders(root)
    ix = _existing_index(dup_tol)
    stat = dict(found=len(items), imported=0, duplicates=0, no_mask=0, no_label=0, unreadable=0, by_kind=defaultdict(int), dups=[])
    keep = []
    for it in items:
        it["label"] = it["label"] or norm_label(default_label)
        if it["label"] is None:
            stat["no_label"] += 1; continue
        img = imio.imread(it["image"], cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape) < 16:
            stat["unreadable"] += 1; continue
        m = np.zeros_like(img, np.uint8); has = False
        for mf in it["masks"]:
            mm = imio.imread(mf, cv2.IMREAD_GRAYSCALE)
            if mm is not None:
                if mm.shape != m.shape:
                    mm = cv2.resize(mm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                m |= (mm > 127).astype(np.uint8); has = True
        if it["label"] == "normal":
            has = True                                   # ảnh bình thường: mặt nạ rỗng là đúng
        if require_mask and not has:
            stat["no_mask"] += 1; continue
        h = dhash(img); twin = ix.find(h)
        if twin:
            stat["duplicates"] += 1; stat["dups"].append((os.path.basename(it["image"]), twin)); continue
        ix.add(h, f"{dataset}/{os.path.basename(it['image'])}")
        keep.append((it, img, m, has, h))
    splits = ["ext"] * len(keep) if holdout else assign_splits([k[0] for k in keep], ratios, seed)
    c = db.conn(); used = set()
    for (it, img, m, has, h), sp in zip(keep, splits):
        stem = _safe(os.path.splitext(os.path.basename(it["image"]))[0]); n_try = 1
        while f"{dataset}/{it['label']}/{stem}" in used:        # tên tệp trùng ở thư mục khác: thêm hậu tố
            n_try += 1; stem = f"{stem.split('~')[0]}~{n_try}"
        used.add(f"{dataset}/{it['label']}/{stem}")
        rel, relm = f"{dataset}/{it['label']}/{stem}.png", f"{dataset}/{it['label']}/{stem}_mask.png"
        _save(rel, img); _save(relm, m * 255)
        c.execute("INSERT OR REPLACE INTO images(dataset,filename,mask,kind,domain,split,source,width,height,tumor_ratio,created_at,patient,phash,license) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (dataset, rel, relm if has else None, it["label"], "B" if holdout else "A", sp, dataset, img.shape[1], img.shape[0],
                   float(m.mean()), db.now(), it["patient"], format(h, "016x"), license or REGISTRY.get(dataset, {}).get("license")))
        stat["imported"] += 1; stat["by_kind"][it["label"]] += 1
    c.commit(); stat["by_kind"] = dict(stat["by_kind"])
    log(f"[{dataset}] tìm thấy {stat['found']} · nhập {stat['imported']} · trùng {stat['duplicates']} · thiếu mặt nạ {stat['no_mask']} · "
        f"thiếu nhãn {stat['no_label']} · không đọc được {stat['unreadable']} · theo loại {stat['by_kind']}")
    return stat


def dedup_report(tol=4):
    """Quét toàn bộ ảnh thật đã nhập, liệt kê các cặp gần trùng (khác bộ hoặc cùng bộ)."""
    ix, pairs = HashIndex(tol), []
    for r in db.q("SELECT id, dataset, filename, phash FROM images WHERE phash IS NOT NULL AND dataset NOT IN ('sample','uploads') ORDER BY id"):
        h = int(r["phash"], 16); t = ix.find(h)
        if t:
            pairs.append((t, f"{r['dataset']}/{os.path.basename(r['filename'])}"))
        ix.add(h, f"{r['dataset']}/{os.path.basename(r['filename'])}")
    return pairs


# ---------------------------------------------------------------- tiến độ ---------------------------------
REAL_SQL = "dataset NOT IN ('sample','uploads')"


def progress(target=None):
    """Số ảnh thật hiện có so với mục tiêu; theo loại, theo bộ, theo tập; số ảnh có mặt nạ dùng được cho phân đoạn."""
    target = target or TARGET_REAL
    rows = db.q(f"SELECT dataset, kind, split, COUNT(*) n, SUM(mask IS NOT NULL) nm, COUNT(DISTINCT patient) pat FROM images WHERE {REAL_SQL} GROUP BY dataset, kind, split")
    total = sum(r["n"] for r in rows); masked = sum(r["nm"] or 0 for r in rows)
    by_kind = {k: sum(r["n"] for r in rows if r["kind"] == k) for k in KINDS}
    by_split = defaultdict(int)
    for r in rows:
        by_split[r["split"]] += r["n"]
    ds = []
    for d in sorted({r["dataset"] for r in rows}):
        rr = [r for r in rows if r["dataset"] == d]; info = REGISTRY.get(d, {})
        ds.append(dict(dataset=d, name=info.get("name", d), cite=info.get("cite", ""), license=info.get("license", "—"),
                       n=sum(r["n"] for r in rr), with_mask=sum(r["nm"] or 0 for r in rr), published=info.get("published"),
                       by_kind={k: sum(r["n"] for r in rr if r["kind"] == k) for k in KINDS}, note=info.get("note", "")))
    return dict(target=target, total=total, with_mask=masked, missing=max(0, target - masked), pct=round(100 * masked / target, 1) if target else None,
                by_kind=by_kind, by_split=dict(by_split), datasets=ds, registry=REGISTRY,
                synthetic=db.q("SELECT COUNT(*) n FROM images WHERE dataset='sample'", one=True)["n"])


# ---------------------------------------------------------------- nguồn ảnh -------------------------------
LINKS = {
    "busi": "https://doi.org/10.1016/j.dib.2019.104863", "busbra": "https://doi.org/10.1002/mp.16812",
    "uclm": "https://data.mendeley.com/datasets/7fvgj4jsp7", "breast": "https://doi.org/10.1038/s41597-024-02984-z",
    "stu": "https://github.com/xbhlk/STU-Hospital", "udiat": "https://doi.org/10.1109/JBHI.2017.2731873",
}


def export_sources(out_md="SOURCES.md", out_csv=os.path.join("data", "sources.csv")):
    """Ghi SOURCES.md (bảng nguồn + giấy phép + số ảnh đã dùng) và data/sources.csv (từng ảnh → bộ, bệnh nhân, loại, tập).
    Dùng để ghi nguồn ảnh vào báo cáo. Các số trong bảng là số ảnh THỰC SỰ đã nhập sau khi loại trùng."""
    p = progress(); root = os.path.dirname(os.path.abspath(__file__))
    lines = ["# Nguồn ảnh siêu âm thật đã dùng", "",
             f"Tạo lúc {db.now()} · {p['with_mask']} ảnh có mặt nạ · bình thường {p['by_kind']['normal']} / lành tính {p['by_kind']['benign']} / ác tính {p['by_kind']['malignant']}.", "",
             "| Bộ dữ liệu | Số ảnh đã dùng | Bình thường / Lành / Ác | Giấy phép | Trích dẫn | Liên kết |", "|---|---|---|---|---|---|"]
    for d in p["datasets"]:
        k = d["by_kind"]
        lines.append(f"| {d['name']} | {d['n']} | {k['normal']} / {k['benign']} / {k['malignant']} | {d['license']} | {d['cite']} | {LINKS.get(d['dataset'], '')} |")
    lines += ["", "Ảnh trùng lặp đã bị loại; ảnh của cùng một bệnh nhân luôn nằm cùng một tập. Hãy tuân thủ giấy phép và yêu cầu trích dẫn của từng bộ.", ""]
    with open(os.path.join(root, out_md), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))
    rows = db.q(f"SELECT dataset, filename, kind, split, patient, license, width, height FROM images WHERE {REAL_SQL} ORDER BY dataset, id")
    cp = os.path.join(root, out_csv); os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "file", "class", "split", "patient", "license", "width", "height", "citation"])
        for r in rows:
            w.writerow([r["dataset"], os.path.basename(r["filename"]), r["kind"], r["split"], r["patient"] or "", r["license"] or "",
                        r["width"], r["height"], REGISTRY.get(r["dataset"], {}).get("cite", "")])
    return len(rows)
