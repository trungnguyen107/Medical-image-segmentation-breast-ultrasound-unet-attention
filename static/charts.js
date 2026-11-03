/* Thư viện biểu đồ SVG nhỏ gọn, không cần mạng (SonoSeg). */
window.Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const PAL = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];
  const color = (i) => css(PAL[i % PAL.length]) || "#3b82f6";
  const el = (tag, a = {}, p) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(+v) ? "–" : Number(v).toFixed(d).replace(".", ","));
  let tipEl;
  function tip(e, html) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "chart-tip"; document.body.appendChild(tipEl); }
    if (!html) { tipEl.classList.remove("on"); return; }
    tipEl.innerHTML = html; tipEl.classList.add("on");
    const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    tipEl.style.left = Math.max(6, Math.min(innerWidth - w - 6, e.clientX - w / 2)) + "px";
    tipEl.style.top = Math.max(6, e.clientY - h - 12) + "px";
  }
  const hover = (n, html) => { n.addEventListener("mousemove", (e) => tip(e, html)); n.addEventListener("mouseleave", () => tip(null)); };
  function ticks(lo, hi, n = 5) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.01) || mag * 10;
    const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function legend(target, series) {
    const lg = document.createElement("div"); lg.className = "chart-legend";
    lg.innerHTML = series.map((s, i) => `<span><i style="background:${s.color || color(i)}"></i>${s.name}</span>`).join("");
    target.appendChild(lg);
  }
  function frame(target, o) {
    target.innerHTML = "";
    if (o.legend !== false && o.series && o.series.length > 1) legend(target, o.series);
    const W = o.width || Math.max(300, Math.round(target.clientWidth || 600)), H = o.height || 260, m = { l: o.ml ?? 46, r: o.mr ?? 16, t: 14, b: o.xTitle ? 44 : 30 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img" }, target);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const y = (v) => m.t + ph - ((Math.max(o.yMin, Math.min(o.yMax, v)) - o.yMin) / (o.yMax - o.yMin)) * ph;
    for (const t of ticks(o.yMin, o.yMax, o.yTicks || 5)) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "c-grid" }, svg);
      el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", class: "c-tick" }, svg).textContent = fmt(t, o.tickDec ?? 2);
    }
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, class: "c-axis" }, svg);
    if (o.yTitle) el("text", { x: 11, y: m.t + ph / 2, transform: `rotate(-90 11 ${m.t + ph / 2})`, "text-anchor": "middle", class: "c-tick" }, svg).textContent = o.yTitle;
    if (o.xTitle) el("text", { x: m.l + pw / 2, y: H - 6, "text-anchor": "middle", class: "c-tick" }, svg).textContent = o.xTitle;
    if (o.ref !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: y(o.ref), y2: y(o.ref), class: "c-ref" }, svg);
    return { svg, W, H, m, pw, ph, y };
  }
  const roundTop = (x, yv, w, base, r) => { r = Math.min(r, w / 2, Math.max(0, base - yv)); return `M${x},${base}V${yv + r}Q${x},${yv} ${x + r},${yv}H${x + w - r}Q${x + w},${yv} ${x + w},${yv + r}V${base}Z`; };

  /* cột nhóm */
  function bar(target, o) {
    const { svg, m, pw, ph, y } = frame(target, o);
    const nc = o.categories.length, ns = o.series.length, gw = pw / nc, bw = Math.min(o.maxBar || 42, (gw * 0.76) / ns), gap = ns > 1 ? 3 : 0;
    o.categories.forEach((c, ci) => {
      const x0 = m.l + gw * ci + (gw - (bw + gap) * ns + gap) / 2;
      o.series.forEach((s, si) => {
        const v = s.values[ci]; if (v === null || v === undefined) return;
        const x = x0 + si * (bw + gap), yv = y(v), base = m.t + ph, col = (s.colors && s.colors[ci]) || s.color || color(si);
        const p = el("path", { d: roundTop(x, yv, bw, base, 5), fill: col, class: "c-bar", style: `animation-delay:${(ci * ns + si) * 25}ms` }, svg);
        const err = s.errors && s.errors[ci];
        hover(p, `<b>${s.name}</b><br>${c}: ${fmt(v, o.dec ?? 3)}${err ? " ± " + fmt(err, o.dec ?? 3) : ""}`);
        if (err) {
          const cx = x + bw / 2, a = y(v + err), b = y(v - err);
          el("path", { d: `M${cx},${a}V${b}M${cx - 4},${a}H${cx + 4}M${cx - 4},${b}H${cx + 4}`, class: "c-err" }, svg);
        }
        if (o.labels !== false && bw >= 20) el("text", { x: x + bw / 2, y: (err ? y(v + err) : yv) - 6, "text-anchor": "middle", class: "c-val" }, svg).textContent = fmt(v, o.dec ?? 3);
      });
      el("text", { x: m.l + gw * ci + gw / 2, y: m.t + ph + 18, "text-anchor": "middle", class: "c-cat" }, svg).textContent = c;
    });
  }

  /* đường */
  function line(target, o) {
    const { svg, m, pw, y } = frame(target, o);
    const n = o.x.length, xs = (i) => m.l + 12 + (i * (pw - 24)) / Math.max(1, n - 1);
    const every = Math.max(1, Math.ceil(n / (o.maxXLabels || 10)));
    o.x.forEach((xv, i) => { if (i % every === 0 || i === n - 1) el("text", { x: xs(i), y: m.t + (o.height || 260) - m.t - m.b + 18, "text-anchor": "middle", class: "c-cat" }, svg).textContent = o.xLabels ? o.xLabels[i] : xv; });
    o.series.forEach((s, si) => {
      const c = s.color || color(si), vals = s.values, pts = vals.map((v, i) => (v === null || v === undefined ? null : [xs(i), y(v)]));
      if (s.errors) {
        const up = vals.map((v, i) => [xs(i), y(v + s.errors[i])]), lo = vals.map((v, i) => [xs(i), y(v - s.errors[i])]).reverse();
        el("path", { d: "M" + up.concat(lo).map((p) => p.join(",")).join("L") + "Z", fill: c, opacity: 0.14 }, svg);
      }
      const d = pts.reduce((acc, p, i) => (p ? acc + (acc && pts[i - 1] ? "L" : "M") + p.join(",") : acc), "");
      const path = el("path", { d, fill: "none", stroke: c, "stroke-width": s.width || 2.2, "stroke-linejoin": "round", "stroke-linecap": "round", class: "c-line", "stroke-dasharray": s.dash || "" }, svg);
      try { const L = path.getTotalLength(); if (!s.dash) { path.style.strokeDasharray = L; path.style.strokeDashoffset = L; } } catch { /* */ }
      if (o.dots !== false && n <= 40) pts.forEach((p, i) => {
        if (!p) return;
        el("circle", { cx: p[0], cy: p[1], r: n > 20 ? 2.5 : 4, fill: c, class: "c-dot" }, svg);
        const hit = el("circle", { cx: p[0], cy: p[1], r: 10, fill: "transparent" }, svg);
        hover(hit, `<b>${s.name}</b><br>${o.xName || ""}${o.xLabels ? o.xLabels[i] : o.x[i]}: ${fmt(vals[i], o.dec ?? 3)}`);
      });
    });
  }

  /* bánh vòng */
  function donut(target, o) {
    target.innerHTML = "";
    const size = o.size || 180, R = size / 2 - 6, r0 = R * 0.64, total = o.values.reduce((a, b) => a + b, 0) || 1;
    const wrap = document.createElement("div"); wrap.className = "donut-wrap"; target.appendChild(wrap);
    const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, class: "donut", width: size, height: size }, wrap);
    let a0 = -Math.PI / 2;
    o.values.forEach((v, i) => {
      const a1 = a0 + (v / total) * Math.PI * 2 - 0.0001, large = a1 - a0 > Math.PI ? 1 : 0, cx = size / 2, cy = size / 2;
      const P = (a, rr) => [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
      const [x1, y1] = P(a0, R), [x2, y2] = P(a1, R), [x3, y3] = P(a1, r0), [x4, y4] = P(a0, r0);
      const p = el("path", { d: `M${x1},${y1}A${R},${R} 0 ${large} 1 ${x2},${y2}L${x3},${y3}A${r0},${r0} 0 ${large} 0 ${x4},${y4}Z`, fill: (o.colors && o.colors[i]) || color(i), class: "c-slice" }, svg);
      hover(p, `<b>${o.labels[i]}</b>: ${v} (${fmt((v / total) * 100, 1)}%)`);
      a0 = a1 + 0.0001;
    });
    const t = el("text", { x: size / 2, y: size / 2 - 2, "text-anchor": "middle", class: "donut-n" }, svg); t.textContent = o.center ?? total;
    const s = el("text", { x: size / 2, y: size / 2 + 16, "text-anchor": "middle", class: "c-tick" }, svg); s.textContent = o.centerLabel || "";
    const lg = document.createElement("div"); lg.className = "donut-legend";
    lg.innerHTML = o.labels.map((l, i) => `<div><i style="background:${(o.colors && o.colors[i]) || color(i)}"></i><span>${l}</span><b>${o.values[i]}</b></div>`).join("");
    wrap.appendChild(lg);
  }

  /* hộp + điểm (phân bố theo mô hình) */
  function box(target, o) {
    const { svg, m, pw, y } = frame(target, { ...o, legend: false });
    const n = o.groups.length, gw = pw / n;
    o.groups.forEach((g, gi) => {
      const v = [...g.values].sort((a, b) => a - b), q = (p) => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))];
      const cx = m.l + gw * gi + gw / 2, w = Math.min(46, gw * 0.42), c = g.color || color(gi);
      g.values.forEach((val, i) => {
        const jx = cx + ((((i * 9301 + 49297) % 233280) / 233280) - 0.5) * w * 1.3;
        el("circle", { cx: jx, cy: y(val), r: 2.2, fill: c, opacity: 0.35 }, svg);
      });
      el("line", { x1: cx, x2: cx, y1: y(q(0.05)), y2: y(q(0.95)), stroke: c, "stroke-width": 1.4 }, svg);
      const rect = el("rect", { x: cx - w / 2, y: y(q(0.75)), width: w, height: Math.max(1, y(q(0.25)) - y(q(0.75))), rx: 5, fill: c, "fill-opacity": 0.22, stroke: c, "stroke-width": 1.6 }, svg);
      el("line", { x1: cx - w / 2, x2: cx + w / 2, y1: y(q(0.5)), y2: y(q(0.5)), stroke: c, "stroke-width": 2.5 }, svg);
      hover(rect, `<b>${g.name}</b><br>trung vị ${fmt(q(0.5))} · Q1 ${fmt(q(0.25))} · Q3 ${fmt(q(0.75))}<br>n = ${v.length}`);
      el("text", { x: cx, y: (o.height || 260) - (o.xTitle ? 26 : 12), "text-anchor": "middle", class: "c-cat" }, svg).textContent = g.name;
    });
  }

  /* phân tán */
  function scatter(target, o) {
    const { svg, m, pw, ph, y } = frame(target, { ...o, legend: false });
    const xs = (v) => m.l + ((v - o.xMin) / (o.xMax - o.xMin)) * pw;
    for (const t of ticks(o.xMin, o.xMax, 5)) el("text", { x: xs(t), y: m.t + ph + 18, "text-anchor": "middle", class: "c-cat" }, svg).textContent = fmt(t, o.xDec ?? 0);
    o.points.forEach((p, i) => {
      const c = p.color || color(i);
      const d = el("circle", { cx: xs(p.x), cy: y(p.y), r: p.r || 7, fill: c, "fill-opacity": 0.85, stroke: css("--surface"), "stroke-width": 2, class: "c-dot" }, svg);
      hover(d, `<b>${p.label}</b><br>${o.xName}: ${fmt(p.x, o.xDec ?? 1)}<br>${o.yName}: ${fmt(p.y, 3)}`);
      el("text", { x: xs(p.x) + ((i % 2) ? -10 : 10), y: y(p.y) + ((i % 2) ? -10 : 16), "text-anchor": (i % 2) ? "end" : "start", class: "c-val" }, svg).textContent = p.label;
    });
  }

  /* biểu đồ tần suất */
  function hist(target, o) {
    const bins = o.bins || 10, lo = o.min, hi = o.max, w = (hi - lo) / bins, cnt = Array(bins).fill(0);
    o.values.forEach((v) => { cnt[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)))]++; });
    bar(target, { categories: cnt.map((_, i) => fmt(lo + i * w, o.dec ?? 2)), series: [{ name: o.name || "Số ảnh", values: cnt, color: o.color || color(0) }], yMin: 0, yMax: Math.max(1, Math.ceil(Math.max(...cnt) * 1.15)), dec: 0, tickDec: 0, height: o.height || 220, legend: false, labels: o.labels ?? true, maxBar: 60, xTitle: o.xTitle });
  }

  /* đồng hồ vòng cung (Dice) */
  function gauge(target, v, label) {
    target.innerHTML = "";
    const R = 52, C = 2 * Math.PI * R, pct = Math.max(0, Math.min(1, v || 0));
    target.innerHTML = `<svg viewBox="0 0 130 130" class="gauge"><circle cx="65" cy="65" r="${R}" class="g-bg"/><circle cx="65" cy="65" r="${R}" class="g-fg" style="stroke-dasharray:${C};stroke-dashoffset:${C}" data-off="${C * (1 - pct)}"/><text x="65" y="66" text-anchor="middle" class="g-v">${fmt(v, 3)}</text><text x="65" y="86" text-anchor="middle" class="g-l">${label}</text></svg>`;
    requestAnimationFrame(() => requestAnimationFrame(() => { const f = target.querySelector(".g-fg"); f.style.strokeDashoffset = f.dataset.off; }));
  }

  /* biểu đồ điểm ngang (Cleveland dot plot): mỗi hàng một hạng mục, mỗi chuỗi một chấm (± sai số).
     Dùng khi các giá trị gần nhau – trục không cần bắt đầu từ 0 mà vẫn trung thực. */
  function dot(target, o) {
    target.innerHTML = "";
    if (o.legend !== false && o.series.length > 1) legend(target, o.series);
    const rows = o.categories.length, rowH = o.rowH || 34, W = o.width || Math.max(300, Math.round(target.clientWidth || 600));
    const m = { l: o.ml ?? 92, r: 52, t: o.dotLabels ? 14 : 6, b: o.xTitle ? 44 : 28 }, H = m.t + m.b + rows * rowH, pw = W - m.l - m.r;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img" }, target);
    const x = (v) => m.l + ((Math.max(o.xMin, Math.min(o.xMax, v)) - o.xMin) / (o.xMax - o.xMin)) * pw;
    for (const t of ticks(o.xMin, o.xMax, o.xTicks || 5)) {
      el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "c-grid" }, svg);
      el("text", { x: x(t), y: H - m.b + 17, "text-anchor": "middle", class: "c-tick" }, svg).textContent = fmt(t, o.tickDec ?? 2);
    }
    if (o.xTitle) el("text", { x: m.l + pw / 2, y: H - 2, "text-anchor": "middle", class: "c-tick" }, svg).textContent = o.xTitle;
    const ns = o.series.length;
    o.categories.forEach((c, ri) => {
      const cy = m.t + ri * rowH + rowH / 2;
      if (ri % 2 === 0) el("rect", { x: m.l, y: cy - rowH / 2, width: pw, height: rowH, class: "c-band" }, svg);
      el("text", { x: m.l - 10, y: cy + 4, "text-anchor": "end", class: "c-cat" }, svg).textContent = c;
      const vals = o.series.map((s) => s.values[ri]).filter((v) => v !== null && v !== undefined);
      if (vals.length > 1) el("line", { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: cy, y2: cy, class: "c-conn" }, svg);
      o.series.forEach((s, si) => {
        const v = s.values[ri]; if (v === null || v === undefined) return;
        const col = (s.colors && s.colors[ri]) || s.color || color(si), dy = ns > 1 && o.offset ? (si - (ns - 1) / 2) * 7 : 0, e = s.errors && s.errors[ri];
        if (e) el("path", { d: `M${x(v - e)},${cy + dy}H${x(v + e)}M${x(v - e)},${cy + dy - 4}V${cy + dy + 4}M${x(v + e)},${cy + dy - 4}V${cy + dy + 4}`, stroke: col, "stroke-width": 1.4, fill: "none", opacity: 0.7 }, svg);
        const d = el("circle", { cx: x(v), cy: cy + dy, r: o.r || 6, fill: col, stroke: css("--surface"), "stroke-width": 2, class: "c-dot" }, svg);
        hover(d, `<b>${s.name}</b><br>${c}: ${fmt(v, o.dec ?? 3)}${e ? " ± " + fmt(e, o.dec ?? 3) : ""}`);
        if (o.dotLabels) el("text", { x: x(v), y: cy + dy - 10, "text-anchor": "middle", class: "c-val" }, svg).textContent = fmt(v, o.dec ?? 3);
      });
      if (o.valueLabel !== false) {
        const last = o.series[ns - 1].values[ri];
        if (ns === 1 && last !== null && last !== undefined) el("text", { x: W - m.r + 8, y: cy + 4, class: "c-val" }, svg).textContent = fmt(last, o.dec ?? 3);
      }
    });
  }
  return { bar, line, donut, box, scatter, hist, gauge, dot, color, fmt };
})();
