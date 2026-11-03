"""Các mô hình phân đoạn: U-Net, Attention U-Net, U-Net + SE/CBAM và mô hình đề xuất.

Một lớp UNetFlex duy nhất, cấu hình bằng các cờ (Bảng 5.2 và 5.6 của đề cương):
  attn_gate   : Attention Gate trên các kết nối tắt            [Oktay 2018]
  bottleneck  : None | "se" | "cbam"  khối chú ý ở tầng đáy     [Hu 2018, Woo 2018]
  cbam_all    : chèn CBAM vào mọi khối tích chập (mô hình M3)
  deep_sup    : giám sát sâu bằng đầu ra phụ ở các tầng giải mã
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, cbam=False, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.body = nn.Sequential(*layers)
        self.att = CBAM(cout) if cbam else nn.Identity()

    def forward(self, x):
        return self.att(self.body(x))


class SE(nn.Module):
    """Squeeze-and-Excitation: tái hiệu chỉnh theo kênh."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean((2, 3)))
        return x * s[:, :, None, None]


class CBAM(nn.Module):
    """Chú ý kênh rồi chú ý không gian (tích chập 7x7)."""
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        mc = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))
        x = x * mc[:, :, None, None]
        sp = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.spatial(sp))


class AttentionGate(nn.Module):
    """alpha = sigmoid( psi( relu( Wx x + Wg g ) ) ); x_hat = alpha * x."""
    def __init__(self, cx, cg, cint):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(cx, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.wg = nn.Sequential(nn.Conv2d(cg, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.psi = nn.Sequential(nn.Conv2d(cint, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.last_alpha = None
        self.capture = False

    def forward(self, x, g):
        g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        a = self.psi(F.relu(self.wx(x) + self.wg(g), inplace=True))
        if self.capture:
            self.last_alpha = a.detach()
        return x * a


class UNetFlex(nn.Module):
    def __init__(self, in_ch=1, base=16, depth=4, attn_gate=False, bottleneck=None,
                 cbam_all=False, deep_sup=False, dropout=0.1):
        super().__init__()
        self.depth, self.deep_sup = depth, deep_sup
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        cin = in_ch
        for i in range(depth):
            self.enc.append(ConvBlock(cin, ch[i], cbam=cbam_all))
            cin = ch[i]
        self.pool = nn.MaxPool2d(2)
        self.bott = ConvBlock(ch[depth - 1], ch[depth], cbam=cbam_all, dropout=dropout)
        self.bott_att = {"se": SE, "cbam": CBAM}.get(bottleneck, None)
        self.bott_att = self.bott_att(ch[depth]) if self.bott_att else nn.Identity()
        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2))
            self.gates.append(AttentionGate(ch[i], ch[i + 1], max(ch[i] // 2, 4)) if attn_gate else nn.Identity())
            self.dec.append(ConvBlock(ch[i] * 2, ch[i], cbam=cbam_all))
        self.head = nn.Conv2d(ch[0], 1, 1)
        if deep_sup:
            self.aux = nn.ModuleList([nn.Conv2d(ch[i], 1, 1) for i in reversed(range(1, depth))])

    def attention_gates(self):
        return [g for g in self.gates if isinstance(g, AttentionGate)]

    def set_capture(self, flag=True):
        for g in self.attention_gates():
            g.capture = flag

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott_att(self.bott(x))
        aux = []
        for k, (up, gate, dec) in enumerate(zip(self.up, self.gates, self.dec)):
            skip = skips[self.depth - 1 - k]
            g = x
            u = up(x)
            s = gate(skip, g) if isinstance(gate, AttentionGate) else skip
            x = dec(torch.cat([u, s], 1))
            if self.deep_sup and self.training and k < self.depth - 1:
                aux.append(F.interpolate(self.aux[k](x), size=skips[0].shape[2:], mode="bilinear", align_corners=False))
        out = self.head(x)
        return (out, aux) if (self.deep_sup and self.training) else out


# Cấu hình theo Bảng 5.2 (M1–M4) và Bảng 5.6 (A0–A5)
CONFIGS = {
    "M1": dict(),                                                   # U-Net cơ sở  (= A0)
    "M2": dict(attn_gate=True),                                     # Attention U-Net (= A1)
    "M3": dict(cbam_all=True),                                      # U-Net + CBAM
    "M4": dict(attn_gate=True, bottleneck="cbam", deep_sup=True),   # mô hình đề xuất (= A5)
    "A2": dict(bottleneck="se"),
    "A3": dict(bottleneck="cbam"),
    "A4": dict(attn_gate=True, bottleneck="cbam"),
    # các tổ hợp bổ sung để hoàn thiện ma trận ablation 2×2×2 (AG × CBAM đáy × giám sát sâu)
    "D1": dict(deep_sup=True),
    "D2": dict(attn_gate=True, deep_sup=True),
    "D3": dict(bottleneck="cbam", deep_sup=True),
}
NAMES = {"M1": "U-Net (cơ sở)", "M2": "Attention U-Net", "M3": "U-Net + CBAM",
         "M4": "Mô hình đề xuất", "A2": "U-Net + SE (đáy)", "A3": "U-Net + CBAM (đáy)", "A4": "AG + CBAM (đáy)",
         "D1": "U-Net + giám sát sâu", "D2": "AG + giám sát sâu", "D3": "CBAM (đáy) + giám sát sâu"}


def build(name, base=16, **kw):
    cfg = dict(CONFIGS[name]); cfg.update(kw)
    return UNetFlex(base=base, **cfg)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def trace(model, x):
    """Lan truyền thuận có ghi lại các bước trung gian (dùng cho trang “Giải phẫu mô hình”).

    Trả dict:
      enc   : [N,C,H,W] đặc trưng sau mỗi khối mã hóa (độ phân giải cao → thấp)
      bott  : đặc trưng tầng đáy trước khối chú ý; bott_out sau khối chú ý
      cbam  : {"channel": [C], "spatial": [h,w]} nếu tầng đáy dùng CBAM; se: [C] nếu dùng SE
      gates : [(k, alpha)] alpha của Attention Gate trên từng kết nối tắt (k=0 là tầng sâu nhất)
      dec   : đặc trưng sau mỗi khối giải mã (thấp → cao)
      aux   : logit của các đầu ra phụ giám sát sâu (thấp → cao), đã nội suy về kích thước đầu vào
      out   : logit đầu ra chính
    """
    model.eval()
    r = dict(enc=[], gates=[], dec=[], aux=[], cbam=None, se=None)
    skips = []
    for blk in model.enc:
        x = blk(x); skips.append(x); r["enc"].append(x); x = model.pool(x)
    x = model.bott(x); r["bott"] = x
    ba = model.bott_att
    if isinstance(ba, CBAM):
        mc = torch.sigmoid(ba.mlp(x.mean((2, 3))) + ba.mlp(x.amax((2, 3))))
        xc = x * mc[:, :, None, None]
        sp = torch.sigmoid(ba.spatial(torch.cat([xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)], 1)))
        r["cbam"] = dict(channel=mc[0], spatial=sp[0, 0])
        x = xc * sp
    elif isinstance(ba, SE):
        s = ba.fc(x.mean((2, 3))); r["se"] = s[0]; x = x * s[:, :, None, None]
    r["bott_out"] = x
    for k, (up, gate, dec) in enumerate(zip(model.up, model.gates, model.dec)):
        skip = skips[model.depth - 1 - k]
        g = x; u = up(x)
        if isinstance(gate, AttentionGate):
            gg = F.interpolate(g, size=skip.shape[2:], mode="bilinear", align_corners=False)
            a = gate.psi(F.relu(gate.wx(skip) + gate.wg(gg)))
            r["gates"].append((k, a[0, 0])); s = skip * a
        else:
            s = skip
        x = dec(torch.cat([u, s], 1)); r["dec"].append(x)
        if model.deep_sup and k < model.depth - 1:
            r["aux"].append(F.interpolate(model.aux[k](x), size=skips[0].shape[2:], mode="bilinear", align_corners=False))
    r["out"] = model.head(x)
    return r
