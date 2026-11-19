"""Huấn luyện và so sánh mô hình trên ẢNH THẬT (dòng lệnh).

  python manage.py train-real --models M1,M4 --epochs 30 --size 128 --base 16

M1 = U-Net gốc (mốc), M4 = U-Net + Attention (đề xuất). Cùng dữ liệu, cùng tập chia, cùng hạt ngẫu nhiên.
In Dice/IoU trên tập kiểm tra thật, ΔDice so với U-Net, kiểm định Wilcoxon theo cặp trên từng ảnh và Dice theo loại.
"""
import argparse, json, time

import numpy as np

import db, real_data


def main(argv):
    ap = argparse.ArgumentParser(prog="manage.py train-real")
    ap.add_argument("--models", default="M1,M4"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true", help="tắt tăng cường lúc học (không ảnh hưởng số ảnh thật)")
    a = ap.parse_args(argv)
    import trainer
    db.init(); p = real_data.progress()
    tr = db.q("SELECT COUNT(*) n FROM images WHERE dataset NOT IN ('sample','uploads','stu') AND split='train' AND mask IS NOT NULL", one=True)["n"]
    if tr < 20:
        raise SystemExit("Chưa đủ ảnh thật để huấn luyện (cần ≥ 20 ảnh có mặt nạ trong tập train). Nhập dữ liệu trước: python fetch_data.py")
    print(f"Ảnh thật có mặt nạ: {p['with_mask']} · train {tr} · theo tập {p['by_split']}")
    rids = {}
    for mt in [m.strip() for m in a.models.split(",") if m.strip()]:
        cfg = dict(model_type=mt, epochs=a.epochs, bs=a.bs, lr=a.lr, size=a.size, base=a.base, n_train=10 ** 6,
                   augment=not a.no_augment, dataset="real", name=f"{mt} · ảnh thật · {a.epochs} epoch · {a.size}px")
        rid = trainer.start(cfg); rids[mt] = rid; t0 = time.time()
        while trainer.busy():
            r = db.q("SELECT epoch, best_val_dice, message FROM runs WHERE id=?", (rid,), one=True)
            print(f"\r  {mt}: epoch {r['epoch']}/{a.epochs} · Dice kiểm định tốt nhất {r['best_val_dice'] or 0:.3f} · {int(time.time() - t0)}s   ", end="", flush=True)
            time.sleep(5)
        r = db.q("SELECT * FROM runs WHERE id=?", (rid,), one=True); print()
        if r["status"] != "done":
            raise SystemExit(f"{mt} không hoàn tất: {r['message']}")
    ev = {mt: db.q("SELECT * FROM evals WHERE model_id=? AND split='test'", (f"R{rid}",), one=True) for mt, rid in rids.items()}
    print("\n| Mô hình | Dice | IoU | Precision | Recall | HD95 | Dice lành | Dice ác | FP ảnh bình thường |\n|---|---|---|---|---|---|---|---|---|")
    for mt, e in ev.items():
        bk = json.loads(e["by_kind"]); f = lambda v: "–" if v is None else f"{v:.3f}"
        print(f"| {mt} | {f(e['dice'])} | {f(e['iou'])} | {f(e['precision'])} | {f(e['recall'])} | {f(e['hd95'])} | {f(bk.get('benign'))} | {f(bk.get('malignant'))} | {f(e['fp_rate'])} |")
    if "M1" in ev and len(ev) > 1:
        base = {x["id"]: x["dice"] for x in json.loads(ev["M1"]["per_image"]) if x["kind"] != "normal"}
        for mt, e in ev.items():
            if mt == "M1":
                continue
            cur = {x["id"]: x["dice"] for x in json.loads(e["per_image"]) if x["kind"] != "normal"}
            ids = sorted(set(base) & set(cur)); d = np.array([cur[i] - base[i] for i in ids])
            try:
                from scipy.stats import wilcoxon
                pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except Exception:  # noqa: BLE001
                pv = float("nan")
            print(f"\nΔDice {mt} so với M1 = {d.mean():+.4f} (n = {len(d)} ảnh tổn thương, Wilcoxon p = {pv:.4f})."
                  f" {'Chưa có bằng chứng thống kê rằng ' + mt + ' tốt hơn.' if not pv < 0.05 or d.mean() <= 0 else mt + ' tốt hơn có ý nghĩa thống kê ở mức 0,05.'}")
    real_data.export_sources()
    print("\nĐã cập nhật SOURCES.md và data/sources.csv. Mô hình lưu ở weights/user/ và chọn được trên web (mục Phân đoạn).")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
