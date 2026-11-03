"""Cơ sở dữ liệu SQLite của ứng dụng (data/app.db).

Bảng
  images : ảnh trong CSDL (bộ ảnh mẫu, ảnh tải lên, ảnh nhập từ BUSI) – tệp nằm trong data/
  runs   : các lần huấn luyện từ giao diện web (tham số, tiến trình, lịch sử epoch, kết quả)
  evals  : kết quả đánh giá mô hình trên một tập (test/ext/…)
  cases  : lịch sử các ca phân đoạn (chỉ số, ảnh thu nhỏ)
"""
import json, os, sqlite3, threading
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA, "app.db")
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset TEXT NOT NULL DEFAULT 'sample',
  filename TEXT NOT NULL UNIQUE,          -- đường dẫn tương đối trong data/
  mask TEXT,                              -- đường dẫn mặt nạ chuẩn (có thể NULL)
  kind TEXT,                              -- benign | malignant | normal | unknown
  domain TEXT DEFAULT 'A',                -- A: thiết bị huấn luyện, B: thiết bị khác
  split TEXT DEFAULT 'train',             -- train | val | test | ext | upload
  source TEXT DEFAULT 'synthetic',        -- synthetic | upload | busi
  width INTEGER, height INTEGER, tumor_ratio REAL,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_images_split ON images(split, kind);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT, model_type TEXT, epochs INTEGER, bs INTEGER, lr REAL, size INTEGER, base INTEGER,
  n_train INTEGER, n_val INTEGER, dataset TEXT, augment INTEGER,
  status TEXT, progress REAL DEFAULT 0, epoch INTEGER DEFAULT 0,
  best_val_dice REAL, best_epoch INTEGER, test_dice REAL, test_iou REAL,
  params INTEGER, duration_s REAL, weights TEXT, history TEXT, message TEXT,
  created_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS evals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id TEXT, split TEXT, n INTEGER,
  dice REAL, dice_sd REAL, iou REAL, precision REAL, recall REAL, hd95 REAL, fp_rate REAL, ms REAL,
  by_kind TEXT, per_image TEXT, created_at TEXT,
  UNIQUE(model_id, split)
);
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT, image_label TEXT, image_id INTEGER, model_id TEXT, threshold REAL,
  detected INTEGER, area_ratio REAL, diameter_px REAL, area_mm2 REAL, diameter_mm REAL,
  circularity REAL, solidity REAL, aspect_ratio REAL,
  dice REAL, iou REAL, precision REAL, recall REAL, hd95 REAL, ms REAL, thumb TEXT, note TEXT
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


def conn():
    c = getattr(_local, "c", None)
    if c is None:
        os.makedirs(DATA, exist_ok=True)
        c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.c = c
    return c


def init():
    c = conn(); c.executescript(SCHEMA); c.commit()


def q(sql, args=(), one=False):
    cur = conn().execute(sql, args)
    rows = [dict(r) for r in cur.fetchall()]
    return (rows[0] if rows else None) if one else rows


def x(sql, args=()):
    c = conn(); cur = c.execute(sql, args); c.commit()
    return cur.lastrowid


def insert(table, d):
    keys = list(d)
    return x(f"INSERT INTO {table}({','.join(keys)}) VALUES({','.join('?' * len(keys))})",
             [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in d.values()])


def update(table, rid, d):
    if not d:
        return
    sets = ",".join(f"{k}=?" for k in d)
    x(f"UPDATE {table} SET {sets} WHERE id=?",
      [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in d.values()] + [rid])


def path(rel):
    return os.path.join(DATA, rel) if rel else None
