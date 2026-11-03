"""Chạy toàn bộ thí nghiệm E1–E6 (mục 3.5) và xuất results/results.json cùng các hình.

  python run_experiments.py --stage train      # huấn luyện M1–M4, A2–A4 × các hạt ngẫu nhiên (có thể chạy lại, tự bỏ qua mô hình đã có)
  python run_experiments.py --stage eval       # đánh giá, kiểm định thống kê, xuất results.json và hình
  python run_experiments.py --stage all
"""
import argparse, json, os, subprocess, sys, time, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np

KEYS = ["M1", "M2", "M3", "M4", "A2", "A3", "A4"]
MAIN = ["M1", "M2", "M3", "M4"]
CFG = dict(size=96, base=8, epochs=25, bs=16, lr=1e-3, patience=8, lr_patience=3, seeds=[0, 1, 2, 3], threads=2, hw="CPU 2 nhân (không GPU)")
RUNS = "weights/runs"


def stage_train(workers=2):
    import data as D
    D.load_synthetic()                                            # tạo bộ nhớ đệm trước khi chạy song song
    jobs = []
    for s in CFG["seeds"]:
        for k in KEYS:
            if k not in MAIN and s != 0:                          # ablation: chỉ hạt 0 để tiết kiệm thời gian
                continue
            out = f"{RUNS}/{k}_s{s}.pt"
            if os.path.exists(out.replace(".pt", "_hist.json")):
                continue
            jobs.append((k, s, out))
    print(f"{len(jobs)} lượt huấn luyện cần chạy", flush=True)

    def go(j):
        k, s, out = j
        cmd = [sys.executable, "train.py", "--model", k, "--seed", str(s), "--out", out, "--epochs", str(CFG["epochs"]),
               "--bs", str(CFG["bs"]), "--lr", str(CFG["lr"]), "--patience", str(CFG["patience"]), "--lr-patience", str(CFG["lr_patience"]),
               "--base", str(CFG["base"]), "--threads", "1"]
        t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
        open(f"{RUNS}/{k}_s{s}.log", "w").write(r.stdout + r.stderr)
        print(f"xong {k} s{s} trong {time.time()-t:.0f}s", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.stage in ("train", "all"):
        stage_train(a.workers)
    if a.stage in ("eval", "all"):
        import evaluate_all
        evaluate_all.main(CFG)
