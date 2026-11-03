"""Tiện ích quản trị CSDL.

  python manage.py init                 # tạo CSDL, sinh bộ ảnh mẫu, đánh giá ban đầu
  python manage.py rebuild-sample       # sinh lại bộ ảnh mẫu
  python manage.py import-busi <thư mục> # nhập BUSI (benign/, malignant/, normal/) thành bộ dữ liệu 'busi'
  python manage.py import-stu           # tải và nhập 42 ảnh siêu âm vú THẬT (STU-Hospital) thành tập 'real'
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
    elif cmd == "import-busi":
        T.import_folder(argv[2], dataset="busi")
    db.init()
    for r in db.q("SELECT dataset, split, COUNT(*) n FROM images GROUP BY dataset, split ORDER BY dataset, split"):
        print(f"  {r['dataset']:8s} {r['split']:6s} {r['n']:5d}")
    print("  ca phân đoạn:", db.q("SELECT COUNT(*) n FROM cases", one=True)["n"], "| lần huấn luyện:", db.q("SELECT COUNT(*) n FROM runs", one=True)["n"])


if __name__ == "__main__":
    main(sys.argv)
