"""Tiện ích quản trị CSDL.

  python manage.py init                 # tạo CSDL, sinh bộ ảnh mẫu, đánh giá ban đầu
  python manage.py rebuild-sample       # sinh lại bộ ảnh mẫu
  python manage.py import-busi [thư mục]  # nhập BUSI (benign/, malignant/, normal/) – mặc định data/external/Dataset_BUSI_with_GT
  python manage.py import-stu           # tải và nhập 42 ảnh siêu âm vú THẬT (STU-Hospital) thành tập 'real'
  python manage.py import-real <bộ> <thư mục> [--manifest tệp.csv] [--holdout] [--default-label benign|malignant|normal]
                                        # nhập ảnh siêu âm THẬT: loại trùng, chia theo bệnh nhân, ghi giấy phép
  python manage.py progress             # số ảnh thật đã có so với mục tiêu 10.000
  python manage.py sources              # ghi SOURCES.md + data/sources.csv (nguồn từng ảnh) để đưa vào báo cáo
  python manage.py train-real --models M1,M4 --epochs 30   # huấn luyện U-Net và U-Net+Attention trên ảnh thật, so sánh ΔDice
  python manage.py dedup                # liệt kê cặp ảnh gần trùng giữa các bộ
  python manage.py stats                # thống kê CSDL
"""
import os, sys
os.environ["SONOSEG_NO_BOOT"] = "1"
import db, dataset_tools as T  # noqa: E402


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "stats"
    if cmd == "init":
        import app
        print("Đã khởi tạo:", app.bootstrap(), "ảnh mẫu")
    elif cmd == "rebuild-sample":
        print("Sinh lại bộ ảnh mẫu…"); print(T.build_sample(force=True), "ảnh")
    elif cmd == "import-stu":
        T.import_stu()
    elif cmd == "import-real":
        import argparse, real_data
        ap = argparse.ArgumentParser(prog="manage.py import-real"); ap.add_argument("dataset"); ap.add_argument("root")
        ap.add_argument("--manifest"); ap.add_argument("--holdout", action="store_true"); ap.add_argument("--default-label")
        ap.add_argument("--allow-no-mask", action="store_true"); ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args(argv[2:])
        real_data.import_real(a.dataset, a.root, manifest=a.manifest, holdout=a.holdout, default_label=a.default_label,
                              require_mask=not a.allow_no_mask, seed=a.seed)
    elif cmd == "progress":
        import real_data
        p = real_data.progress()
        print(f"Ảnh thật: {p['total']} (có mặt nạ {p['with_mask']}) / mục tiêu {p['target']} = {p['pct']}% · còn thiếu {p['missing']}")
        print("  theo loại:", p["by_kind"], "| theo tập:", p["by_split"])
        for d in p["datasets"]:
            print(f"  {d['dataset']:10s} {d['n']:6d} ảnh ({d['with_mask']} mặt nạ) · {d['license']} · {d['cite']}")
    elif cmd == "sources":
        import real_data
        print(real_data.export_sources(), "ảnh đã ghi vào SOURCES.md và data/sources.csv")
    elif cmd == "train-real":
        import real_train
        real_train.main(argv[2:])
    elif cmd == "dedup":
        import real_data
        pr = real_data.dedup_report()
        print(f"{len(pr)} cặp gần trùng"); [print("  ", a_, "≈", b_) for a_, b_ in pr[:50]]
    elif cmd == "import-busi":
        T.import_busi(argv[2] if len(argv) > 2 else T.BUSI_DIR)
    db.init()
    for r in db.q("SELECT dataset, split, COUNT(*) n FROM images GROUP BY dataset, split ORDER BY dataset, split"):
        print(f"  {r['dataset']:8s} {r['split']:6s} {r['n']:5d}")
    print("  ca phân đoạn:", db.q("SELECT COUNT(*) n FROM cases", one=True)["n"], "| lần huấn luyện:", db.q("SELECT COUNT(*) n FROM runs", one=True)["n"])


if __name__ == "__main__":
    main(sys.argv)
