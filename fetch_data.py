"""Tải và nhập các bộ ảnh siêu âm vú công khai có mặt nạ vào SonoSeg (cải thiện dữ liệu đầu vào – thí nghiệm R4).

  python fetch_data.py                 # tải BUS-BRA từ Zenodo (cần Internet), rồi nhập mọi bộ có trong data/external
  python fetch_data.py --local         # chỉ nhập các bộ đã giải nén sẵn trong data/external
  python fetch_data.py --mirror        # khi không vào được Zenodo: lấy BUS-BRA và BrEaST từ bản sao trên GitHub

Các bộ được nhận (đặt đúng tên thư mục trong data/external/):
  Dataset_BUSI_with_GT/   BUSI      – Al-Dhabyani et al., Data in Brief 28, 2020        (nhập bằng manage.py import-busi)
  BUS-BRA/                BUS-BRA   – Gómez-Flores et al., Medical Physics 51, 2024     CC BY 4.0, có bus_data.csv
  BrEaST/                 BrEaST    – Pawłowska et al., Scientific Data 11, 2024         CC BY 4.0, có bảng lâm sàng .xlsx
Ảnh trùng bị loại; ảnh của một bệnh nhân nằm cùng một tập. Nếu có results/mix_split.json, tập huấn luyện/kiểm tra
được chia ĐÚNG như thí nghiệm R4 để mô hình M4-MIX không bị đánh giá trên ảnh nó đã học.
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "external")
BUSBRA = os.path.join(EXT, "BUS-BRA")
BREAST = os.path.join(EXT, "BrEaST")
URL_BUSBRA = "https://zenodo.org/records/7730709/files/BUSBRA.zip?download=1"            # bản 1.0 (133,9 MB)
# Bản sao KHÔNG CHÍNH THỨC trên GitHub (giữ nguyên tệp gốc của BUS-BRA và BrEaST, cùng giấy phép CC BY 4.0).
MIRROR = "https://github.com/ii50-txt/BrEaST-Lesions_USG-images_and_masks"


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "SonoSeg"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); got += len(b)
            if total:
                print(f"\r  {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def fetch_busbra_zenodo():
    os.makedirs(BUSBRA, exist_ok=True); z = os.path.join(BUSBRA, "BUSBRA.zip")
    if not zipfile.is_zipfile(z):
        print("Đang tải BUS-BRA từ Zenodo…"); download(URL_BUSBRA, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(BUSBRA)
    # bản Zenodo giải nén vào BUSBRA/BUSBRA/{Images,Masks,bus_data.csv}: đưa lên một cấp
    inner = os.path.join(BUSBRA, "BUSBRA")
    if os.path.isdir(inner) and not os.path.exists(os.path.join(BUSBRA, "bus_data.csv")):
        for f in os.listdir(inner):
            shutil.move(os.path.join(inner, f), os.path.join(BUSBRA, f))


def fetch_mirror():
    """git sparse-checkout chỉ các thư mục ảnh cần dùng (khoảng 270 MB)."""
    tmp = tempfile.mkdtemp(prefix="sonoseg_mirror_")
    run = lambda *a: subprocess.run(a, cwd=tmp, check=True)  # noqa: E731
    print("Đang lấy BUS-BRA và BrEaST từ bản sao GitHub (không chính thức)…")
    run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", MIRROR, "m")
    m = os.path.join(tmp, "m")
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=m, check=True)
    with open(os.path.join(m, ".git", "info", "sparse-checkout"), "w") as fh:
        fh.write("/BUSBRA/\n/BrEaST-Lesions_USG-images_and_masks/\n/yolo11_seg_6class/images/\n/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx\n")
    subprocess.run(["git", "checkout", "-q", "HEAD"], cwd=m, check=True)
    if not os.path.exists(os.path.join(BUSBRA, "bus_data.csv")):
        shutil.copytree(os.path.join(m, "BUSBRA"), BUSBRA, dirs_exist_ok=True)
    os.makedirs(BREAST, exist_ok=True)
    for f in glob.glob(os.path.join(m, "yolo11_seg_6class", "images", "*", "case*.png")) + \
            glob.glob(os.path.join(m, "BrEaST-Lesions_USG-images_and_masks", "case*_tumor.png")) + \
            [os.path.join(m, "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx")]:
        shutil.copy2(f, BREAST)
    shutil.rmtree(tmp, ignore_errors=True)


def breast_manifest():
    """Bảng kê image,mask,label,patient cho BrEaST từ bảng lâm sàng gốc."""
    import openpyxl
    xl = glob.glob(os.path.join(BREAST, "*clinical-data*.xlsx"))
    if not xl:
        raise SystemExit("Thiếu bảng lâm sàng BrEaST-Lesions-USG-clinical-data-*.xlsx trong " + BREAST)
    rows = list(openpyxl.load_workbook(xl[0], read_only=True).active.iter_rows(values_only=True))
    h = rows[0]; ci, cm, cl, cid = (h.index(k) for k in ("Image_filename", "Mask_tumor_filename", "Classification", "CaseID"))
    out = os.path.join(BREAST, "manifest.csv"); n = 0
    with open(out, "w", encoding="utf8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "mask", "label", "patient"])
        for r in rows[1:]:
            if r[ci] and os.path.exists(os.path.join(BREAST, r[ci])):
                w.writerow([r[ci], r[cm] or "", r[cl], f"breast{r[cid]}"]); n += 1
    print(f"BrEaST: bảng kê {n} ảnh"); return out


def import_all():
    import db, real_data
    db.init()
    sm = os.path.join(ROOT, "results", "mix_split.json")
    split = json.load(open(sm, encoding="utf8")) if os.path.exists(sm) else {}
    have = {r["dataset"] for r in db.q("SELECT DISTINCT dataset FROM images")}
    if os.path.exists(os.path.join(BUSBRA, "bus_data.csv")) and "busbra" not in have:
        real_data.import_real("busbra", BUSBRA, manifest=os.path.join(BUSBRA, "bus_data.csv"), split_map=split.get("busbra"))
    if glob.glob(os.path.join(BREAST, "*clinical-data*.xlsx")) and "breast" not in have:
        real_data.import_real("breast", BREAST, manifest=breast_manifest(), split_map=split.get("breast"))
    n = real_data.export_sources()
    p = real_data.progress()
    print(f"Ảnh thật trong CSDL: {p['with_mask']} / mục tiêu {p['target']} ({p['pct']}%). SOURCES.md ghi {n} ảnh.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="không tải, chỉ nhập các bộ đã có trong data/external")
    ap.add_argument("--mirror", action="store_true", help="lấy BUS-BRA và BrEaST từ bản sao GitHub")
    a = ap.parse_args()
    os.makedirs(EXT, exist_ok=True)
    if a.mirror:
        fetch_mirror()
    elif not a.local and not os.path.exists(os.path.join(BUSBRA, "bus_data.csv")):
        try:
            fetch_busbra_zenodo()
        except Exception as e:  # noqa: BLE001
            print(f"Không tải được từ Zenodo ({e}). Thử: python fetch_data.py --mirror"); sys.exit(1)
    import_all()


if __name__ == "__main__":
    main()
