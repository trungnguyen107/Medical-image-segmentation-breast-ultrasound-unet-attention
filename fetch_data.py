"""Tải bộ ảnh siêu âm vú công khai BUS-BRA (1.875 ảnh, 1.064 bệnh nhân, CC BY 4.0) từ Zenodo rồi nhập vào SonoSeg.

Chạy trên máy CÓ Internet:   python fetch_data.py
Kết quả: data/external/BUS-BRA/ + ảnh được nhập (loại trùng, chia theo bệnh nhân) + SOURCES.md.

BUS-BRA chỉ có ảnh lành tính và ác tính. Ảnh BÌNH THƯỜNG và các bộ khác (BUS-UCLM, BUSI) tải tay từ trang gốc
(xem DATASETS.md), giải nén vào data/external/<tên>/ rồi nhập bằng manage.py import-real.

Nguồn: Gómez-Flores W., Gregorio-Calas M.J., Coelho de Albuquerque Pereira W. BUS-BRA: A breast ultrasound dataset for
assessing computer-aided diagnosis systems. Medical Physics 51, 3110–3123, 2024. Zenodo, CC BY 4.0.
"""
import os, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "https://zenodo.org/records/7730709/files/BUSBRA.zip?download=1"     # bản 1.0 (133,9 MB)
DEST = os.path.join(ROOT, "data", "external", "BUS-BRA")


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


def main():
    os.makedirs(DEST, exist_ok=True); z = os.path.join(DEST, "BUSBRA.zip")
    if not zipfile.is_zipfile(z):
        print("Đang tải BUS-BRA từ Zenodo…"); download(URL, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DEST)
    csvs = [os.path.join(dp, f) for dp, _, fs in os.walk(DEST) for f in fs if f.lower().endswith(".csv")]
    print("Tệp bảng kê:", csvs)
    if not csvs:
        sys.exit("Không thấy tệp CSV trong BUS-BRA. Hãy xem cấu trúc thư mục và tạo bảng kê (README, mục dữ liệu thật).")
    import real_data
    real_data.import_real("busbra", DEST, manifest=csvs[0])
    n = real_data.export_sources()
    print(f"Đã ghi SOURCES.md và data/sources.csv ({n} ảnh). Xem tiến độ: python manage.py progress")


if __name__ == "__main__":
    main()
