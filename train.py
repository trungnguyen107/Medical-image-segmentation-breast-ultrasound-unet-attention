"""Huấn luyện một mô hình phân đoạn.

Ví dụ:
  python train.py --model M4 --data synthetic --epochs 30 --seed 0
  python train.py --model M4 --data busi --busi-root /duong/dan/Dataset_BUSI_with_GT --size 256 --epochs 150
Hàm mất mát 0,5·BCE + 0,5·Dice; Adam; ReduceLROnPlateau; dừng sớm theo Dice kiểm định.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data as D
import models as M


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def seg_loss(logits, y, lam=0.5):
    return lam * F.binary_cross_entropy_with_logits(logits, y) + (1 - lam) * dice_loss(logits, y)


@torch.no_grad()
def predict(model, X, bs=32, device="cpu"):
    """X: uint8 [N,H,W] -> xác suất float32 [N,H,W]."""
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32) / 255.0)[:, None].to(device)
        out.append(torch.sigmoid(model(xb))[:, 0].cpu().numpy())
    return np.concatenate(out)


def batch_dice(prob, Y):
    p = prob > 0.5; g = Y > 0
    inter = (p & g).sum((1, 2)); den = p.sum((1, 2)) + g.sum((1, 2))
    return np.where(den == 0, 1.0, 2 * inter / np.maximum(den, 1))


def run(args, splits=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    if splits is None:
        splits = D.load_synthetic() if args.data == "synthetic" else D.load_busi(args.busi_root, args.size, args.seed)
    size = splits["X_train"].shape[1]
    model = M.build(args.model, base=args.base)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.lr_patience)
    ds = D.SegDataset(splits["X_train"], splits["Y_train"], D.make_augment(size))
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True, generator=g)
    hist, best, bad = [], -1, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0
        for xb, yb in dl:
            out = model(xb)
            if isinstance(out, tuple):
                main, aux = out
                loss = seg_loss(main, yb) + sum(w * seg_loss(a, yb) for w, a in zip([0.5, 0.25, 0.125], aux))
            else:
                loss = seg_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        vd = float(np.mean(batch_dice(predict(model, splits["X_val"]), splits["Y_val"])))
        sched.step(1 - vd)
        hist.append(dict(epoch=ep, train_loss=tl / len(dl), val_dice=vd, lr=opt.param_groups[0]["lr"], time=time.time() - t0))
        print(f"[{args.model} s{args.seed}] ep {ep:3d} loss {tl/len(dl):.4f} val_dice {vd:.4f}", flush=True)
        if vd > best:
            best, bad = vd, 0
            torch.save({"state": model.state_dict(), "model": args.model, "base": args.base, "size": size, "val_dice": vd}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                break
    json.dump(dict(args=vars(args), history=hist, best_val_dice=best, train_time=time.time() - t0),
              open(args.out.replace(".pt", "_hist.json"), "w"), indent=1)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M4", choices=list(M.CONFIGS))
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "busi"])
    ap.add_argument("--busi-root", default="")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr-patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="weights/model.pt")
    run(ap.parse_args())
