"""Quản lý bộ dữ liệu ảnh trong CSDL.

- build_sample(): sinh bộ ảnh mẫu (trích từ bộ sinh ảnh mô phỏng của đồ án) vào data/sample/
  theo cấu trúc giống BUSI: <loại>/<tên>.png và <tên>_mask.png; ghi danh sách vào bảng images.
- import_folder(): nhập một thư mục kiểu BUSI (benign/, malignant/, normal/ – mặt nạ hậu tố _mask).
- load_split(): đọc một tập (train/val/test/ext) thành mảng numpy để huấn luyện / đánh giá.
"""
import glob, os, shutil
import cv2
import imio
import numpy as np

import data as D
import db

SAMPLE_PLAN = [  # (split, số ảnh, hạt bắt đầu, miền) – cùng dải hạt với bộ dữ liệu của đồ án
    ("train", 240, 1000, "A"), ("val", 40, 50000, "A"), ("test", 80, 90000, "A"), ("ext", 60, 130000, "B")]
SAMPLE_SIZE = 128


def _save(rel, arr):
    p = db.path(rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    imio.imwrite(p, arr)


def build_sample(force=False, log=print):
    db.init()
    n = db.q("SELECT COUNT(*) n FROM images WHERE dataset='sample'", one=True)["n"]
    if n and not force:
        return n
    if force:
        db.x("DELETE FROM images WHERE dataset='sample'")
        shutil.rmtree(db.path("sample"), ignore_errors=True)
    rows, k = [], 0
    for split, cnt, s0, dom in SAMPLE_PLAN:
        for i in range(cnt):
            img, m, kind = D.make_sample(s0 + i, domain=dom, size=SAMPLE_SIZE)
            k += 1
            name = f"{split}_{dom}_{i + 1:03d}"
            rel, relm = f"sample/{kind}/{name}.png", f"sample/{kind}/{name}_mask.png"
            _save(rel, img); _save(relm, m * 255)
            rows.append((rel, relm, kind, dom, split, img.shape[1], img.shape[0], float(m.mean()), db.now()))
        log(f"  {split}: {cnt} ảnh (miền {dom})")
    c = db.conn()
    c.executemany("INSERT OR REPLACE INTO images(dataset,filename,mask,kind,domain,split,source,width,height,tumor_ratio,created_at) "
                  "VALUES('sample',?,?,?,?,?,'synthetic',?,?,?,?)", rows)
    c.commit()
    return len(rows)


def import_folder(root, dataset="busi", seed=0, split=(0.7, 0.1, 0.2), log=print):
    """Nhập thư mục kiểu BUSI vào data/<dataset>/ và CSDL (chia phân tầng train/val/test)."""
    db.init(); rng = np.random.default_rng(seed); n = 0
    for kind in ["benign", "malignant", "normal"]:
        files = [f for f in sorted(glob.glob(os.path.join(root, kind, "*.png"))) if "_mask" not in f]
        idx = rng.permutation(len(files)); a, b = int(split[0] * len(files)), int((split[0] + split[1]) * len(files))
        for j, fi in enumerate(idx):
            f = files[fi]; img = imio.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            m = np.zeros_like(img)
            for mf in sorted(glob.glob(f[:-4] + "_mask*.png")):
                mm = imio.imread(mf, cv2.IMREAD_GRAYSCALE)
                if mm is not None and mm.shape == m.shape:
                    m |= (mm > 127).astype(np.uint8)
            sp = "train" if j < a else "val" if j < b else "test"
            base = os.path.splitext(os.path.basename(f))[0].replace(" ", "_").replace("(", "").replace(")", "")
            rel, relm = f"{dataset}/{kind}/{base}.png", f"{dataset}/{kind}/{base}_mask.png"
            _save(rel, img); _save(relm, m * 255)
            db.x("INSERT OR REPLACE INTO images(dataset,filename,mask,kind,domain,split,source,width,height,tumor_ratio,created_at) "
                 "VALUES(?,?,?,?,'A',?,?,?,?,?,?)", (dataset, rel, relm, kind, sp, dataset, img.shape[1], img.shape[0], float(m.mean()), db.now()))
            n += 1
    log(f"Đã nhập {n} ảnh vào bộ '{dataset}'.")
    return n


STU_URL = "https://github.com/xbhlk/STU-Hospital/archive/refs/heads/master.zip"
STU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "external", "STU-Hospital")


def fetch_stu(log=print):
    """Tải bộ ảnh siêu âm vú THẬT STU-Hospital (42 cặp ảnh – mặt nạ, Zhuang et al. 2019) về data/external/."""
    hosp = os.path.join(STU_DIR, "Hospital")
    if os.path.isdir(hosp) and glob.glob(os.path.join(hosp, "Test_Image_*.png")):
        return hosp
    import io, shutil as sh, subprocess, tempfile, urllib.request, zipfile
    log("Đang tải STU-Hospital từ GitHub…")
    if sh.which("git"):                         # cách 1: git clone (nhanh, đi qua proxy của git)
        tmp = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/xbhlk/STU-Hospital.git", tmp],
                           check=True, timeout=180, capture_output=True)
            src = os.path.join(tmp, "Hospital")
            if os.path.isdir(src):
                os.makedirs(hosp, exist_ok=True)
                for f in glob.glob(os.path.join(src, "*.png")):
                    shutil.copy(f, hosp)
                return hosp
        except Exception as e:  # noqa: BLE001
            log(f"git clone không được ({e}); thử tải tệp ZIP…")
        finally:
            sh.rmtree(tmp, ignore_errors=True)
    req = urllib.request.Request(STU_URL, headers={"User-Agent": "SonoSeg"})   # cách 2: tải ZIP
    data = urllib.request.urlopen(req, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in z.namelist():
            if "/Hospital/" in n and n.lower().endswith(".png"):
                os.makedirs(hosp, exist_ok=True)
                with open(os.path.join(hosp, os.path.basename(n)), "wb") as f:
                    f.write(z.read(n))
    return hosp


def import_stu(log=print):
    """Nhập 42 ảnh thật STU-Hospital vào CSDL: bộ 'stu', tập 'real' (chỉ dùng để kiểm tra, không huấn luyện trên web)."""
    db.init(); hosp = fetch_stu(log); n = 0
    for i in range(1, 200):
        fi, fm = os.path.join(hosp, f"Test_Image_{i}.png"), os.path.join(hosp, f"mask_{i}.png")
        if not os.path.exists(fi):
            continue
        img = imio.imread(fi, cv2.IMREAD_GRAYSCALE); mm = imio.imread(fm, cv2.IMREAD_GRAYSCALE)
        if img is None or mm is None:
            continue
        m = cv2.resize(mm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        rel, relm = f"stu/stu_{i:02d}.png", f"stu/stu_{i:02d}_mask.png"
        _save(rel, img); _save(relm, m.astype(np.uint8) * 255)
        db.x("INSERT OR REPLACE INTO images(dataset,filename,mask,kind,domain,split,source,width,height,tumor_ratio,created_at) "
             "VALUES('stu',?,?,'unknown','STU','real','stu',?,?,?,?)", (rel, relm, img.shape[1], img.shape[0], float(m.mean()), db.now()))
        n += 1
    log(f"Đã nhập {n} ảnh thật STU-Hospital (tập 'real').")
    return n


def add_upload(img, mask=None, kind="unknown", name="upload"):
    db.init()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or "upload"
    rid = db.q("SELECT COALESCE(MAX(id),0)+1 n FROM images", one=True)["n"]
    rel = f"uploads/{rid:05d}_{safe}.png"; _save(rel, img)
    relm = None
    if mask is not None:
        relm = f"uploads/{rid:05d}_{safe}_mask.png"; _save(relm, (mask > 0).astype(np.uint8) * 255)
    return db.insert("images", dict(dataset="uploads", filename=rel, mask=relm, kind=kind, domain="?", split="upload", source="upload",
                                    width=img.shape[1], height=img.shape[0],
                                    tumor_ratio=float((mask > 0).mean()) if mask is not None else None, created_at=db.now()))


def read_item(row):
    img = imio.imread(db.path(row["filename"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được tệp ảnh {row['filename']} trong thư mục data/.")
    m = imio.imread(db.path(row["mask"]), cv2.IMREAD_GRAYSCALE) if row.get("mask") else None
    return img, (m > 127 if m is not None else None)


def load_split(split, size, dataset="sample", limit=None):
    rows = db.q("SELECT * FROM images WHERE dataset=? AND split=? AND mask IS NOT NULL ORDER BY id", (dataset, split))
    if limit:
        rows = rows[:limit]
    X = np.zeros((len(rows), size, size), np.uint8); Y = np.zeros_like(X); K = []
    for i, r in enumerate(rows):
        img, m = read_item(r)
        X[i] = D._letterbox(img, size, cv2.INTER_AREA)
        Y[i] = D._letterbox(m.astype(np.uint8), size, cv2.INTER_NEAREST)
        K.append(r["kind"])
    return X, Y, np.array(K), rows
