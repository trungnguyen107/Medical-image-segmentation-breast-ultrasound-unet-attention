"""Các hình không phụ thuộc kết quả huấn luyện: kiến trúc, sơ đồ khối, ảnh dữ liệu mô phỏng."""
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import data as D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
OUT = "results/figures/"
INK, MUTED, LINE = "#14202b", "#5b6b7a", "#c9d3dc"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"


def box(ax, x, y, w, h, text, fc="#eef3f8", ec=LINE, tc=INK, fs=9, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, c=MUTED, ls="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---- Hình: kiến trúc mô hình đề xuất M4 -------------------------------------------------
def fig_arch():
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 12.2); ax.set_ylim(-0.7, 6.3); ax.axis("off")
    ys = [4.9, 3.7, 2.5, 1.3]; xe, xd = 0.6, 7.6; w, h = 1.9, 0.72
    lab = ["256² · 64", "128² · 128", "64² · 256", "32² · 512"]
    for i, y in enumerate(ys):
        box(ax, xe, y, w, h, f"Encoder {i+1}\n2×(Conv3×3-BN-ReLU)\n{lab[i]}", fc="#e4eefb", ec=BLUE, fs=8)
        box(ax, xd, y, w, h, f"Decoder {i+1}\nUp ⊕ Skip → 2×Conv\n{lab[i]}", fc="#e9f7f1", ec=AQUA, fs=8)
        box(ax, 4.05, y + 0.09, 1.35, 0.54, "Attention\nGate", fc="#fdeee6", ec=ORANGE, fs=8, bold=True)
        arrow(ax, (xe + w, y + h / 2), (4.05, y + h / 2), c=ORANGE)
        arrow(ax, (5.4, y + h / 2), (xd, y + h / 2), c=ORANGE)
        if i < 3:
            arrow(ax, (xe + w / 2, y), (xe + w / 2, ys[i + 1] + h), c=BLUE)
            arrow(ax, (xd + w / 2, ys[i + 1] + h), (xd + w / 2, y), c=AQUA)
    box(ax, 3.6, 0.15, 2.4, 0.78, "Tầng đáy 16² · 1024\n2×Conv + Dropout\nCBAM (chú ý kênh + không gian)", fc="#efecfb", ec=VIOLET, fs=8, bold=False)
    arrow(ax, (xe + w / 2, ys[3]), (xe + w / 2, 0.55), c=BLUE); arrow(ax, (xe + w / 2, 0.55), (3.6, 0.55), c=BLUE)
    arrow(ax, (6.0, 0.55), (xd + w / 2, 0.55), c=AQUA); arrow(ax, (xd + w / 2, 0.55), (xd + w / 2, ys[3]), c=AQUA)
    # g (tín hiệu điều khiển)
    for y in ys[:3]:
        ax.text(4.72, y + 0.74, "g", color=ORANGE, fontsize=8, ha="center", style="italic")
    ax.text(xe + w / 2, 5.85, "Ảnh siêu âm 256×256×1", ha="center", fontsize=9, color=INK, fontweight="bold")
    arrow(ax, (xe + w / 2, 5.78), (xe + w / 2, ys[0] + h), c=INK)
    box(ax, xd + 0.1, 5.65, 1.7, 0.45, "Conv1×1 + Sigmoid", fc="#fff", ec=INK, fs=8)
    arrow(ax, (xd + w / 2, ys[0] + h), (xd + w / 2, 5.65), c=INK)
    ax.text(xd + 0.95, 6.18, "Bản đồ xác suất khối u", ha="center", fontsize=9, fontweight="bold", color=INK)
    # giám sát sâu
    for y in ys[1:]:
        ax.add_patch(Rectangle((10.0, y + 0.18), 0.55, 0.36, fc="#fff8e6", ec=YELLOW, lw=1.2))
        ax.text(10.275, y + 0.36, "aux", ha="center", va="center", fontsize=7)
        arrow(ax, (xd + w, y + h / 2), (10.0, y + 0.36), c=YELLOW, ls="--")
    ax.text(10.28, 1.05, "Giám sát\nsâu", ha="center", fontsize=8, color=MUTED)
    # chú giải
    for k, (c, t) in enumerate([(BLUE, "Đường co (encoder)"), (AQUA, "Đường giãn (decoder)"), (ORANGE, "Kết nối tắt + Attention Gate"), (VIOLET, "Khối CBAM"), (YELLOW, "Đầu ra phụ (deep supervision)")]):
        ax.add_patch(Rectangle((0.05 + k * 2.45, -0.45), 0.22, 0.16, fc=c, ec="none")); ax.text(0.34 + k * 2.45, -0.37, t, fontsize=8, va="center", color=MUTED)
    save(fig, "fig_architecture.png")


def fig_ag():
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.set_xlim(0, 9); ax.set_ylim(0, 3.8); ax.axis("off")
    box(ax, 0.1, 2.5, 1.1, 0.6, "x\n(skip, H×W×Cₓ)", fc="#e4eefb", ec=BLUE, fs=8)
    box(ax, 0.1, 0.5, 1.1, 0.6, "g\n(tầng sâu hơn)", fc="#e9f7f1", ec=AQUA, fs=8)
    box(ax, 1.9, 2.5, 1.2, 0.6, "Wₓ: Conv1×1\n+ BN", fs=8); box(ax, 1.9, 0.5, 1.2, 0.6, "W_g: Conv1×1 + BN\n+ nội suy ×2", fs=8)
    box(ax, 3.7, 1.5, 0.7, 0.6, "+", fc="#fff", ec=INK, fs=14, bold=True)
    box(ax, 4.9, 1.5, 0.9, 0.6, "ReLU", fs=9); box(ax, 6.2, 1.5, 1.2, 0.6, "ψ: Conv1×1\n+ Sigmoid", fs=8)
    box(ax, 7.9, 1.5, 0.9, 0.6, "α\n(H×W×1)", fc="#fdeee6", ec=ORANGE, fs=8, bold=True)
    box(ax, 7.9, 3.0, 0.9, 0.6, "x̂ = α ⊙ x", fc="#fdeee6", ec=ORANGE, fs=8, bold=True)
    for p, q in [((1.2, 2.8), (1.9, 2.8)), ((3.1, 2.8), (4.05, 2.1)), ((1.2, 0.8), (1.9, 0.8)), ((3.1, 0.8), (4.05, 1.5)),
                 ((4.4, 1.8), (4.9, 1.8)), ((5.8, 1.8), (6.2, 1.8)), ((7.4, 1.8), (7.9, 1.8)), ((8.35, 2.1), (8.35, 3.0))]:
        arrow(ax, p, q)
    arrow(ax, (0.65, 3.1), (0.65, 3.6), c=BLUE); ax.plot([0.65, 7.9], [3.6, 3.6], color=BLUE, lw=1.2); arrow(ax, (7.9, 3.6), (7.9, 3.4), c=BLUE)
    ax.text(4.2, 3.72, "x được nhân với α", fontsize=8, color=BLUE, ha="center")
    save(fig, "fig_attention_gate.png")


def fig_cbam():
    fig, ax = plt.subplots(figsize=(9.5, 3.2)); ax.set_xlim(0, 9.5); ax.set_ylim(0, 3.2); ax.axis("off")
    box(ax, 0.1, 1.2, 1.0, 0.8, "F\nH×W×C", fc="#e4eefb", ec=BLUE, fs=9)
    box(ax, 1.7, 1.9, 1.5, 0.7, "AvgPool / MaxPool\n(toàn cục)", fs=8); box(ax, 1.7, 0.6, 1.5, 0.7, "MLP dùng chung\n(C→C/8→C)", fs=8)
    box(ax, 3.7, 1.2, 1.2, 0.8, "Mc = σ(·)\nchú ý kênh", fc="#efecfb", ec=VIOLET, fs=8, bold=True)
    box(ax, 5.3, 1.2, 1.2, 0.8, "F′ = Mc ⊙ F", fs=8)
    box(ax, 6.9, 1.9, 1.3, 0.7, "[AvgPool_c ; MaxPool_c]", fs=7); box(ax, 6.9, 0.6, 1.3, 0.7, "Conv 7×7\n+ Sigmoid", fs=8)
    box(ax, 8.3, 1.2, 1.1, 0.8, "F″ = Ms ⊙ F′", fc="#efecfb", ec=VIOLET, fs=8, bold=True)
    for p, q in [((1.1, 1.6), (1.7, 2.25)), ((2.45, 1.9), (2.45, 1.3)), ((3.2, 0.95), (3.7, 1.5)), ((4.9, 1.6), (5.3, 1.6)), ((6.5, 1.6), (6.9, 2.25)), ((7.55, 1.9), (7.55, 1.3)), ((8.2, 0.95), (8.55, 1.2))]:
        arrow(ax, p, q)
    save(fig, "fig_cbam.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
    steps = [("Dữ liệu\nBUSI / UDIAT\n(hoặc mô phỏng)", "#e4eefb", BLUE), ("Làm sạch\nlọc trùng, đệm viền,\nchuẩn hóa [0,1]", "#fff", LINE),
             ("Chia dữ liệu\n70/10/20\nphân tầng", "#fff", LINE), ("Tăng cường\nAlbumentations\n(không lật dọc)", "#fff", LINE),
             ("Huấn luyện\nBCE+Dice, Adam\ndừng sớm", "#e9f7f1", AQUA), ("Đánh giá\nDice, IoU, HD95\nWilcoxon", "#fdeee6", ORANGE), ("Ứng dụng web\nFlask + chỉ số\nhình thái", "#efecfb", VIOLET)]
    for i, (t, fc, ec) in enumerate(steps):
        x = 0.1 + i * 1.55
        box(ax, x, 0.7, 1.35, 1.2, t, fc=fc, ec=ec, fs=8)
        if i < len(steps) - 1: arrow(ax, (x + 1.35, 1.3), (x + 1.55, 1.3))
    save(fig, "fig_pipeline.png")


def fig_system():
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.set_xlim(0, 10.5); ax.set_ylim(-0.3, 4.6); ax.axis("off")
    box(ax, 0.1, 1.5, 2.0, 1.6, "Trình duyệt\nSPA HTML/CSS/JS\n9 trang · canvas\nbiểu đồ SVG\nsáng / tối", fc="#e4eefb", ec=BLUE, fs=8.5)
    box(ax, 3.0, 2.9, 2.4, 1.4, "Flask (app.py)\nphân đoạn · so sánh · độ bền\nhàng loạt · CSDL ảnh\nhuấn luyện · đánh giá · lịch sử", fc="#fff", ec=INK, fs=8)
    box(ax, 3.0, 0.5, 2.4, 1.3, "Tiền / hậu xử lý\nđệm viền · chuẩn hóa\nngưỡng · vùng liên thông\nchỉ số hình thái", fs=8)
    box(ax, 6.2, 2.9, 1.9, 1.4, "PyTorch\nM1–M4 (đồ án)\nR1, R2… (web)", fc="#e9f7f1", ec=AQUA, fs=8.5)
    box(ax, 6.2, 0.5, 1.9, 1.3, "Trainer chạy nền\nhuấn luyện · đánh giá\nđộ bền", fc="#e9f7f1", ec=AQUA, fs=8)
    box(ax, 8.7, 1.5, 1.7, 1.6, "SQLite app.db\nimages · runs\nevals · cases\n+ data/ (ảnh)", fc="#fdeee6", ec=ORANGE, fs=8)
    for p, q in [((2.1, 2.6), (3.0, 3.5)), ((4.2, 2.9), (4.2, 1.8)), ((5.4, 3.6), (6.2, 3.6)), ((7.15, 2.9), (7.15, 1.8)), ((8.1, 1.15), (8.7, 1.9)), ((8.1, 3.6), (8.7, 2.8)), ((5.4, 1.15), (6.2, 1.15))]:
        arrow(ax, p, q)
    ax.plot([9.55, 9.55, 1.1, 1.1], [1.5, 0.0, 0.0, 1.5], color=MUTED, lw=1.4); arrow(ax, (1.1, 0.15), (1.1, 1.5))
    ax.text(5.3, -0.22, "JSON + ảnh base64 trả về trình duyệt", fontsize=8, color=MUTED, ha="center")
    save(fig, "fig_system.png")


# ---- Hình: dữ liệu mô phỏng --------------------------------------------------------------
def fig_samples():
    X, Y, K = D.build_split(80, 1000)
    pick = [next(i for i in range(80) if K[i] == k) for k in ["benign"] * 1] + [i for i in range(80) if K[i] == "benign"][1:3] + \
           [i for i in range(80) if K[i] == "malignant"][:3] + [i for i in range(80) if K[i] == "normal"][:1]
    pick = pick[:6]
    names = {"benign": "Lành tính", "malignant": "Ác tính", "normal": "Bình thường"}
    fig, axs = plt.subplots(2, 6, figsize=(12, 4.3))
    for j, i in enumerate(pick):
        axs[0, j].imshow(X[i], cmap="gray", vmin=0, vmax=255); axs[0, j].set_title(names[K[i]], fontsize=9)
        ov = np.stack([X[i]] * 3, -1).astype(float) / 255; m = Y[i] > 0
        ov[m] = 0.55 * ov[m] + 0.45 * np.array([0.92, 0.41, 0.2])
        axs[1, j].imshow(ov); axs[1, j].contour(Y[i], [0.5], colors="#ffd43b", linewidths=0.8)
        for r in (0, 1): axs[r, j].axis("off")
    axs[0, 0].text(-0.05, 0.5, "Ảnh", transform=axs[0, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9)
    axs[1, 0].text(-0.05, 0.5, "Mặt nạ", transform=axs[1, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9)
    fig.tight_layout(); save(fig, "fig_data_samples.png")


def fig_domain():
    XA, YA, KA = D.build_split(40, 1000); XB, YB, KB = D.build_split(40, 130000, "B")
    fig, axs = plt.subplots(2, 5, figsize=(10.5, 4.4))
    ia = [i for i in range(40) if KA[i] != "normal"][:5]; ib = [i for i in range(40) if KB[i] != "normal"][:5]
    for j in range(5):
        axs[0, j].imshow(XA[ia[j]], cmap="gray", vmin=0, vmax=255); axs[1, j].imshow(XB[ib[j]], cmap="gray", vmin=0, vmax=255)
        for r in (0, 1): axs[r, j].axis("off")
    axs[0, 0].set_title("Miền A (huấn luyện/kiểm tra trong miền)", loc="left", fontsize=9)
    axs[1, 0].set_title("Miền B (khác thiết bị: độ lợi, tương phản, kích thước đốm)", loc="left", fontsize=9)
    fig.tight_layout(); save(fig, "fig_domain_shift.png")


def fig_stats():
    S = D.load_synthetic()
    area = S["Y_train"].sum((1, 2)) / S["Y_train"][0].size * 100; K = S["K_train"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    cnt = [int((K == k).sum()) for k in ["normal", "benign", "malignant"]]
    b = axs[0].bar(["Bình thường", "Lành tính", "Ác tính"], cnt, color=[MUTED, BLUE, ORANGE], width=0.6)
    for r, v in zip(b, cnt): axs[0].text(r.get_x() + r.get_width() / 2, v + 4, str(v), ha="center", fontsize=9)
    axs[0].set_title("Phân bố lớp (tập huấn luyện)", fontsize=10, loc="left"); axs[0].set_ylabel("Số ảnh")
    axs[1].hist(area[area > 0], bins=25, color=BLUE, edgecolor="white"); axs[1].set_title("Diện tích khối u (% ảnh)", fontsize=10, loc="left"); axs[1].set_xlabel("%"); axs[1].set_ylabel("Số ảnh")
    ys, xs = np.nonzero(S["Y_train"].sum(0)); heat = S["Y_train"].sum(0).astype(float)
    axs[2].imshow(heat, cmap="Blues"); axs[2].axis("off"); axs[2].set_title("Mật độ vị trí khối u", fontsize=10, loc="left")
    fig.tight_layout(); save(fig, "fig_data_stats.png")


def fig_aug():
    import albumentations as A
    S = D.load_synthetic(); i = int(np.nonzero(S["Y_train"].sum((1, 2)) > 300)[0][3]); x, y = S["X_train"][i], S["Y_train"][i]
    aug = D.make_augment(D.IMG); fig, axs = plt.subplots(2, 6, figsize=(12, 4.2))
    np.random.seed(3)
    axs[0, 0].imshow(x, cmap="gray"); axs[1, 0].imshow(y, cmap="gray"); axs[0, 0].set_title("Gốc", fontsize=9)
    for j in range(1, 6):
        r = aug(image=x, mask=y); axs[0, j].imshow(r["image"], cmap="gray"); axs[1, j].imshow(r["mask"], cmap="gray"); axs[0, j].set_title(f"Tăng cường {j}", fontsize=9)
    for a in axs.ravel(): a.axis("off")
    fig.tight_layout(); save(fig, "fig_augmentation.png")


if __name__ == "__main__":
    for f in [fig_arch, fig_ag, fig_cbam, fig_pipeline, fig_system, fig_samples, fig_domain, fig_stats, fig_aug]:
        f(); print("ok", f.__name__)
