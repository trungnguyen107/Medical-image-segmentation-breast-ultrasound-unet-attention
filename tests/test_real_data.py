"""Kiểm thử bộ nhập ảnh thật (không cần PyTorch). Ảnh trong kiểm thử chỉ là dữ liệu giả để thử LOGIC nhập,
không phải ảnh y tế thật."""
import csv, os, shutil, sys, tempfile

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import db  # noqa: E402

_tmp = tempfile.mkdtemp(prefix="sonoseg_real_")
db.DATA = os.path.join(_tmp, "data"); db.DB_PATH = os.path.join(db.DATA, "app.db")
import data as D  # noqa: E402
import dataset_tools as T  # noqa: E402
import real_data as RD  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(_tmp, ignore_errors=True)


def _write(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True); cv2.imwrite(p, a)


def make_busi_like(root, n=(6, 12, 8), seed0=100):
    """Thư mục kiểu BUSI: normal/benign/malignant, mặt nạ *_mask.png; kèm 1 ảnh trùng hệt và 1 ảnh thiếu mặt nạ."""
    k = seed0
    for kind, cnt in zip(["normal", "benign", "malignant"], n):
        for i in range(cnt):
            img, m, _ = D.make_sample(k, kind=kind, size=128); k += 1
            _write(f"{root}/{kind}/{kind} ({i + 1}).png", img)
            _write(f"{root}/{kind}/{kind} ({i + 1})_mask.png", m * 255)
    shutil.copy(f"{root}/benign/benign (1).png", f"{root}/benign/benign (99).png")          # bản sao hệt
    shutil.copy(f"{root}/benign/benign (1)_mask.png", f"{root}/benign/benign (99)_mask.png")
    _write(f"{root}/malignant/malignant (50).png", D.make_sample(9999, kind="malignant", size=128)[0])   # thiếu mặt nạ


def test_import_folders_dedup_and_progress():
    root = os.path.join(_tmp, "busi_raw"); make_busi_like(root)
    st = RD.import_real("busi", root, log=lambda *a: None)
    assert st["found"] == 6 + 13 + 9 and st["duplicates"] == 1 and st["no_mask"] == 1
    assert st["imported"] == 26 and st["by_kind"] == {"normal": 6, "benign": 12, "malignant": 8}
    p = RD.progress()
    assert p["total"] == 26 and p["with_mask"] == 26 and p["target"] == 10000 and p["missing"] == 10000 - 26
    assert p["by_kind"]["normal"] == 6 and p["datasets"][0]["dataset"] == "busi"
    normal = db.q("SELECT * FROM images WHERE kind='normal' LIMIT 1", one=True)
    img, m = T.read_item(normal)
    assert m is not None and not m.any()                      # ảnh bình thường: mặt nạ rỗng
    assert {r["split"] for r in db.q("SELECT split FROM images")} <= {"train", "val", "test"}


def test_cross_dataset_duplicates_skipped():
    root = os.path.join(_tmp, "other"); os.makedirs(root)
    shutil.copy(os.path.join(_tmp, "busi_raw/malignant/malignant (1).png"), f"{root}/m1.png")
    shutil.copy(os.path.join(_tmp, "busi_raw/malignant/malignant (1)_mask.png"), f"{root}/m1_mask.png")
    st = RD.import_real("other", root, default_label="malignant", log=lambda *a: None)
    assert st["duplicates"] == 1 and st["imported"] == 0
    assert RD.dedup_report() == []                             # CSDL không chứa cặp trùng


def test_patient_level_split_and_holdout():
    root = os.path.join(_tmp, "pat"); rows = []
    for pat in range(10):
        for j in range(3):                                      # mỗi bệnh nhân 3 ảnh
            kind = "benign" if pat % 2 else "malignant"
            img, m, _ = D.make_sample(5000 + pat * 10 + j, kind=kind, size=128)
            _write(f"{root}/img/p{pat}_{j}.png", img); _write(f"{root}/msk/p{pat}_{j}.png", m * 255)
            rows.append(dict(ID=f"p{pat}_{j}", Mask=f"msk/p{pat}_{j}.png", Pathology=kind.capitalize(), Case=f"P{pat}"))
    with open(f"{root}/meta.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    st = RD.import_real("pat", root, manifest=f"{root}/meta.csv", log=lambda *a: None)
    assert st["imported"] == 30
    by_pat = {}
    for r in db.q("SELECT patient, split FROM images WHERE dataset='pat'"):
        by_pat.setdefault(r["patient"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_pat.values())            # không bệnh nhân nào nằm ở hai tập
    assert len({next(iter(s)) for s in by_pat.values()}) >= 2
    hold = os.path.join(_tmp, "hold"); os.makedirs(hold)
    img, m, _ = D.make_sample(777, kind="benign", size=128)
    _write(f"{hold}/benign/h.png", img); _write(f"{hold}/benign/h_mask.png", m * 255)
    RD.import_real("udiat", hold, holdout=True, log=lambda *a: None)
    assert db.q("SELECT split FROM images WHERE dataset='udiat'", one=True)["split"] == "ext"


def test_load_split_real_excludes_synthetic_and_stu():
    T.build_sample(force=True, log=lambda *a: None)
    X, Y, K, rows = T.load_split("train", 64, "real")
    assert len(X) > 0 and all(r["dataset"] not in ("sample", "uploads", "stu") for r in rows)
    assert RD.progress()["synthetic"] > 0 and RD.progress()["total"] == 26 + 30 + 1


def test_label_normalisation_and_errors():
    assert RD.norm_label("Malignant") == "malignant" and RD.norm_label("Lành tính") == "benign" and RD.norm_label("Normal") == "normal"
    assert RD.norm_label("???") is None
    with pytest.raises(ValueError):
        RD.import_real("sample", _tmp)
    with pytest.raises(ValueError):
        RD.import_real("x", os.path.join(_tmp, "khong_co"))
