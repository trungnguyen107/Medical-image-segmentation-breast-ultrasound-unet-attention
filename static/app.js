/* SonoSeg – giao diện một trang */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const F = (v, d = 3) => (v === null || v === undefined || Number.isNaN(+v) ? "–" : Number(v).toFixed(d).replace(".", ","));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const icon = (n) => `<svg><use href="#i-${n}"/></svg>`;
  const KIND = { benign: "Lành tính", malignant: "Ác tính", normal: "Bình thường", unknown: "Chưa rõ", real: "Ảnh thật" };
  const SPLIT = { train: "Huấn luyện", val: "Kiểm định", test: "Kiểm tra (A)", ext: "Thiết bị khác (B)", real: "Ảnh thật (STU)", upload: "Tải lên" };
  const KIND_TXT = { speckle: ["Nhiễu đốm nhân", "σ"], gauss: ["Nhiễu Gauss cộng", "σ"], contrast: ["Giảm tương phản", "hệ số"] };
  const LEVELS = { speckle: [0, 0.1, 0.2, 0.3, 0.4], gauss: [0, 0.05, 0.1, 0.15, 0.2], contrast: [1, 0.8, 0.6, 0.4, 0.2] };
  const S = { models: [], R: null, input: null, res: null, layer: "overlay", gate: "mean", split: 0.5, imgs: {}, robKind: "speckle", trainPoll: null };
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* */ } } };

  // ------------------------------------------------------------------ tiện ích
  async function api(url, opt = {}) {
    if (opt.json) { opt.method = opt.method || "POST"; opt.headers = { "Content-Type": "application/json" }; opt.body = JSON.stringify(opt.json); delete opt.json; }
    const res = await fetch(url, opt);
    let j; try { j = await res.json(); } catch { throw new Error(`Máy chủ trả lỗi ${res.status}.`); }
    if (!res.ok || (j && j.error)) throw new Error((j && j.error) || `Lỗi ${res.status}`);
    return j;
  }
  function toast(msg, kind = "", title = "") {
    const t = document.createElement("div"); t.className = "toast " + kind;
    t.innerHTML = `<div>${title ? `<b>${esc(title)}</b>` : ""}${esc(msg)}</div>`;
    $("#toasts").appendChild(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 300); }, kind === "err" ? 5200 : 3200);
  }
  const busy = (b, on) => { if (!b) return; b.classList.toggle("busy", on); b.disabled = on; };
  const loadImg = (src) => new Promise((ok, no) => { const im = new Image(); im.onload = () => ok(im); im.onerror = no; im.src = src; });
  const toBlob = (u) => fetch(u).then((r) => r.blob());
  const download = (href, name) => { const a = document.createElement("a"); a.href = href; a.download = name; document.body.appendChild(a); a.click(); a.remove(); };
  const kpi = (label, val, sub = "", ic = "") => `<div class="kpi"><div class="k-l">${ic ? icon(ic) : ""}${label}</div><div class="k-v">${val}</div>${sub ? `<div class="k-s">${sub}</div>` : ""}</div>`;
  const mcol = (id) => { const i = S.models.findIndex((m) => m.id === id); return Charts.color(i < 0 ? 7 : i); };
  const mname = (id) => (S.models.find((m) => m.id === id) || {}).name || id;
  const mtag = (id) => `<span class="mtag"><i style="background:${mcol(id)}"></i>${esc(id)}</span>`;

  function modal(html) { $("#modalBody").innerHTML = html; $("#modal").hidden = false; }
  const closeModal = () => { $("#modal").hidden = true; $("#modalBody").innerHTML = ""; };
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal" || e.target.closest("[data-close]")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeModal(); $("#stage").classList.remove("fs"); fitCanvas(); } });
  const zoom = (src, cap) => modal(`<img class="zoom" src="${src}" alt=""><p class="cap">${esc(cap || "")}</p>`);
  document.addEventListener("click", (e) => { const im = e.target.closest(".cm img, .figs img, .figw img, img.cons, .tbl img"); if (im) zoom(im.src, im.alt); });
  function confirmBox(text, okText = "Đồng ý") {
    return new Promise((res) => {
      modal(`<h3 style="margin:0 30px 10px 0">Xác nhận</h3><p class="muted">${esc(text)}</p><div class="row gap" style="justify-content:flex-end;margin-top:18px"><button class="btn" data-close>Hủy</button><button class="btn primary" id="cfOk">${esc(okText)}</button></div>`);
      $("#cfOk").onclick = () => { closeModal(); res(true); };
      $("#modal").addEventListener("click", function h(e) { if (e.target.id === "modal" || e.target.closest("[data-close]")) { $("#modal").removeEventListener("click", h); res(false); } });
    });
  }

  // ------------------------------------------------------------------ giao diện sáng / tối
  const sysDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
  const curTheme = () => document.documentElement.dataset.theme || (sysDark() ? "dark" : "light");
  function setTheme(t) {
    document.documentElement.dataset.theme = t; store.set("theme", t);
    $$(".theme-sw button").forEach((b) => b.classList.toggle("on", b.dataset.themeSet === t));
    rerender();
  }
  $$(".theme-sw button").forEach((b) => b.addEventListener("click", () => setTheme(b.dataset.themeSet)));
  $("#themeQuick").addEventListener("click", () => setTheme(curTheme() === "dark" ? "light" : "dark"));
  $$(".theme-sw button").forEach((b) => b.classList.toggle("on", b.dataset.themeSet === curTheme()));

  // ------------------------------------------------------------------ điều hướng
  const PAGES = {
    dash: ["Tổng quan", "Bắt đầu từ đây"],
    anatomy: ["Bên trong mô hình đề xuất", "Theo dõi một ảnh đi qua bộ mã hóa, CBAM, Attention Gate và các đầu ra giám sát sâu"],
    ablation: ["Ablation", "Bật/tắt từng thành phần của M4 để đo đóng góp của nó"],
    seg: ["Phân đoạn", "Phân đoạn một ảnh, xem attention và chỉ số, so sánh và thử độ bền"],
    batch: ["Xử lý hàng loạt", "Phân đoạn nhiều ảnh cùng lúc, thống kê và xuất ZIP"],
    history: ["Lịch sử ca", "Các ca phân đoạn đã lưu trong cơ sở dữ liệu"],
    data: ["Cơ sở dữ liệu ảnh", "Bộ ảnh mẫu kèm mặt nạ chuẩn và ảnh do bạn thêm"],
    train: ["Huấn luyện", "Huấn luyện mô hình trên bộ ảnh mẫu ngay trong trình duyệt"],
    eval: ["Đánh giá & so sánh", "So sánh các mô hình trên tập kiểm tra bằng bảng và biểu đồ"],
    exp: ["Kết quả thực nghiệm", "Ảnh siêu âm thật (R1–R2) và thí nghiệm E1–E6 trên dữ liệu mô phỏng"],
    model: ["Mô hình & phương pháp", "Kiến trúc U-Net, Attention Gate, CBAM và luồng xử lý"],
  };
  const RENDER = {};
  let page = null;
  function go(p) { if (!PAGES[p]) p = "dash"; if (location.hash !== "#" + p) location.hash = p; else show(p); }
  function show(p) {
    page = p;
    $$(".page").forEach((x) => x.classList.toggle("on", x.id === "p-" + p));
    $$(".nav nav a").forEach((a) => a.classList.toggle("on", a.dataset.page === p));
    $("#pageTitle").textContent = PAGES[p][0]; $("#pageSub").textContent = PAGES[p][1];
    document.title = `${PAGES[p][0]} · SonoSeg`;
    $("#nav").classList.remove("open"); $("#scrim").classList.remove("on");
    window.scrollTo({ top: 0 });
    if (RENDER[p]) RENDER[p]();
    if (p === "seg") requestAnimationFrame(fitCanvas);
  }
  function rerender() { if (page && RENDER[page]) RENDER[page](true); if (page === "seg") draw(); }
  window.addEventListener("hashchange", () => show(location.hash.slice(1) || "dash"));
  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-go]"); if (!g) return;
    e.preventDefault();
    if (g.dataset.demo) { go("seg"); setTimeout(quickDemo, 150); } else go(g.dataset.go);
  });
  $("#menuBtn").addEventListener("click", () => { $("#nav").classList.add("open"); $("#scrim").classList.add("on"); });
  $("#scrim").addEventListener("click", () => { $("#nav").classList.remove("open"); $("#scrim").classList.remove("on"); });
  window.addEventListener("scroll", () => $(".topbar").classList.toggle("scrolled", scrollY > 4), { passive: true });

  // ------------------------------------------------------------------ mô hình & máy chủ
  async function loadModels() {
    S.models = await api("/api/models");
    const opts = (sel) => {
      const cur = sel.value;
      const b = S.models.filter((m) => m.source === "builtin"), rl = S.models.filter((m) => m.source === "real"), t = S.models.filter((m) => m.source === "trained");
      sel.innerHTML = `<optgroup label="Mô hình của đồ án">${b.map((m) => `<option value="${m.id}">${m.id} – ${esc(m.name)}</option>`).join("")}</optgroup>` +
        (rl.length ? `<optgroup label="Tinh chỉnh trên ảnh thật (STU)">${rl.map((m) => `<option value="${m.id}">${m.id} – ${esc(m.name)}</option>`).join("")}</optgroup>` : "") +
        (t.length ? `<optgroup label="Huấn luyện trên web">${t.map((m) => `<option value="${m.id}">${m.id} – ${esc(m.name)}</option>`).join("")}</optgroup>` : "");
      sel.value = S.models.some((m) => m.id === cur) ? cur : (S.models.find((m) => m.id === "M4") || S.models[0] || {}).id;
    };
    [$("#model"), $("#bModel")].forEach(opts);
    modelDesc();
    return S.models;
  }
  function modelDesc() {
    const m = S.models.find((x) => x.id === $("#model").value);
    $("#modelDesc").textContent = m ? `${m.desc} ${F(m.params / 1e6, 2)} triệu tham số · Dice kiểm định ${F(m.val_dice, 3)}${m.eval && m.eval.test ? " · Dice kiểm tra " + F(m.eval.test, 3) : ""}.` : "";
  }
  $("#model").addEventListener("change", modelDesc);
  async function health() {
    try {
      const h = await api("/api/health"); const s = $("#srv"); s.className = "srv ok";
      $("span", s).textContent = `${h.models} mô hình · CPU ${h.threads} luồng`;
      $("#navTrain").hidden = !h.training;
      return h;
    } catch { $("#srv").className = "srv err"; $("#srv span").textContent = "Mất kết nối máy chủ"; return null; }
  }
  async function updateNavCases() { try { const s = await api("/api/cases?per=1"); $("#navCases").textContent = s.total || ""; } catch { /* */ } }

  // ================================================================== TỔNG QUAN
  RENDER.dash = async () => {
    try {
      const [st] = await Promise.all([api("/api/stats"), S.models.length ? null : loadModels()]);
      const nImg = st.images.reduce((a, b) => a + b.n, 0), byKind = {};
      st.images.forEach((r) => (byKind[r.kind] = (byKind[r.kind] || 0) + r.n));
      const ev = st.evals.filter((e) => e.split === "test"), best = ev.reduce((a, b) => (b.dice > (a ? a.dice : -1) ? b : a), null);
      const runsDone = st.runs.filter((r) => r.status === "done").reduce((a, b) => a + b.n, 0), runsAll = st.runs.reduce((a, b) => a + b.n, 0);
      const m4 = st.evals.find((e) => e.model_id === "M4" && e.split === "test"), m4b = st.evals.find((e) => e.model_id === "M4" && e.split === "ext");
      const stat = (v, l, ic) => `<div class="st">${icon(ic)}<b>${v}</b><span>${l}</span></div>`;
      $("#dashKpi").innerHTML = stat(nImg, `ảnh mẫu · ${(byKind.benign || 0) + (byKind.malignant || 0)} có khối u`, "db") + stat(S.models.length, "mô hình sẵn sàng", "net") +
        stat(m4 ? F(m4.dice, 3) : "–", `Dice của M4${m4b ? " · thiết bị khác " + F(m4b.dice, 3) : ""}`, "chart") + stat(st.cases.n || 0, `ca đã phân đoạn${runsAll ? ` · ${runsAll} lần huấn luyện` : ""}`, "clock");
      const get = (id, sp) => (st.evals.find((e) => e.model_id === id && e.split === sp) || {});
      const lb = S.models.filter((m) => m.source === "builtin").map((m) => ({ m, a: get(m.id, "test"), b: get(m.id, "ext"), c: get(m.id, "real") })).filter((x) => x.a.dice !== undefined).sort((x, y) => y.a.dice - x.a.dice);
      const barc = (v, c) => v === undefined || v === null ? "–" : `<div class="ibar"><i style="width:${Math.max(0, Math.min(1, v)) * 70}%;background:${c}"></i><b>${F(v, 3)}</b></div>`;
      $("#dashModels").innerHTML = lb.length ? `<div class="tbl-wrap"><table class="tbl lead"><thead><tr><th>Mô hình</th><th>Cùng thiết bị</th><th>Thiết bị khác</th>${lb.some((x) => x.c.dice !== undefined) ? "<th>Ảnh thật</th>" : ""}<th class="n">ms/ảnh</th></tr></thead><tbody>` +
        lb.map((x) => `<tr class="${x.m.id === "M4" ? "m4" : ""}"><td><div class="mn">${mtag(x.m.id)}<span>${esc(x.m.type === "M4" ? "Đề xuất" : x.m.name.replace(" (cơ sở)", ""))}</span></div></td><td>${barc(x.a.dice, Charts.color(0))}</td><td>${barc(x.b.dice, Charts.color(1))}</td>${lb.some((y) => y.c.dice !== undefined) ? `<td>${barc(x.c.dice, Charts.color(2))}</td>` : ""}<td class="n">${F(x.a.ms, 1)}</td></tr>`).join("") + `</tbody></table></div><p class="muted sm" style="margin:10px 0 0">Dice trên ảnh có khối u: 80 ảnh cùng thiết bị, 60 ảnh thiết bị khác${lb.some((y) => y.c.dice !== undefined) ? ", 42 ảnh siêu âm thật STU (mô hình chưa từng thấy ảnh thật)" : ""}. Bốn mô hình ngang nhau khi cùng thiết bị; khác biệt nằm ở ảnh lạ.</p>`
        : '<div class="empty"><p>Chưa có kết quả đánh giá.</p></div>';
      $("#dashRecent").innerHTML = st.recent.length ? st.recent.slice(0, 4).map((c) => `<div class="rc" data-go="history"><img src="/data/${c.thumb}" alt="" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'noimg',textContent:'—'}))"><div><b title="${esc(c.image_label)}">${esc(c.image_label)}</b><span>${c.model_id} · ${c.dice !== null ? "Dice " + F(c.dice, 3) : F(c.area_ratio, 2) + "% diện tích"}</span></div></div>`).join("")
        : `<div class="empty"><svg><use href="#i-clock"/></svg><p>Chưa có ca nào. Bấm <b>Chạy thử ngay</b> để phân đoạn ảnh đầu tiên.</p></div>`;
      heroArt();
    } catch (e) { toast(e.message, "err"); }
  };
  let heroDone = false;
  async function heroArt() {
    if (heroDone) return; heroDone = true;
    try {
      const im = await api("/api/images?split=test&kind=malignant&per=1&page=3"), row = im.rows[0]; if (!row) return;
      const f = new FormData(); f.append("image_id", row.id); f.append("model", "M4"); f.append("save", "0");
      const r = await api("/api/segment", { method: "POST", body: f });
      [[r.image, "#ha0"], [r.attention || r.prob, "#ha1"], [r.overlay, "#ha2"]].forEach(([src, sel], i) => {
        const d = $(sel); d.classList.remove("skeleton"); d.style.backgroundImage = `url(${src})`; d.style.animationDelay = i * 120 + "ms"; d.classList.add("in");
      });
      if (r.metrics.dice !== undefined) $("#ha2c").textContent = `Kết quả · Dice ${F(r.metrics.dice, 3)}`;
    } catch { heroDone = false; }
  }

  // ================================================================== PHÂN ĐOẠN
  const cv = $("#cv"), ctx = cv.getContext("2d", { willReadFrequently: true });
  $$("#srcTabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#srcTabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".src").forEach((s) => (s.hidden = s.dataset.src !== b.dataset.src));
    if (b.dataset.src === "db") loadPick();
  }));
  async function loadPick() {
    const g = $("#pickGrid"); g.innerHTML = Array(12).fill('<div class="skeleton" style="aspect-ratio:1"></div>').join("");
    try {
      const j = await api(`/api/images?split=${$("#pickSplit").value}&kind=${$("#pickKind").value}&per=40`);
      g.innerHTML = j.rows.length ? j.rows.map((r) => `<button data-id="${r.id}" title="${esc(r.filename.split("/").pop())} · ${KIND[r.kind] || ""}"><img src="${r.url}" loading="lazy" alt=""><i class="k-${r.kind}"></i></button>`).join("") : '<p class="muted sm" style="grid-column:1/-1">Không có ảnh.</p>';
      $$("button", g).forEach((b) => b.addEventListener("click", () => { $$("button", g).forEach((x) => x.classList.toggle("on", x === b)); setDbInput(j.rows.find((r) => r.id === +b.dataset.id)); }));
    } catch (e) { g.innerHTML = `<p class="muted sm">${esc(e.message)}</p>`; }
  }
  $("#pickSplit").addEventListener("change", loadPick); $("#pickKind").addEventListener("change", loadPick);

  async function setInput(inp) {
    S.input = inp; S.res = null; S.imgs = {}; S.gate = "mean";
    S.imgs.image = await loadImg(inp.preview);
    if (inp.gtUrl) S.imgs.gt = await loadImg(inp.gtUrl);
    inp.w = S.imgs.image.width; inp.h = S.imgs.image.height;
    S.gtEdge = S.imgs.gt ? edgesOf(S.imgs.gt) : null; S.predEdge = null; S.maskBits = null;
    $("#cur").hidden = false; $("#curImg").src = inp.preview; $("#curName").textContent = inp.label;
    $("#curMeta").textContent = `${inp.w}×${inp.h} px · ${inp.gtUrl ? "có mặt nạ chuẩn" : "không có mặt nạ chuẩn"}`;
    $("#emptyStage").hidden = true;
    ["#run", "#cmpRun", "#robRun", "#robSweep"].forEach((s) => ($(s).disabled = false));
    setLayer(S.layer === "split" ? "split" : "image");
    $("#resBox").innerHTML = `<div class="empty"><svg><use href="#i-play"/></svg><p>Bấm <b>Phân đoạn</b> hoặc phím Enter để chạy mô hình ${esc($("#model").value)}.</p></div>`;
    $("#resAct").hidden = true; $("#runInfo").textContent = "";
    $("#cmpGrid").innerHTML = ""; $("#cmpMore").hidden = true; $("#robGrid").innerHTML = ""; $("#robChart").innerHTML = "";
  }
  function setFileInput(blob, label, gtBlob) {
    return setInput({ type: "file", blob, gtBlob: gtBlob || null, label, preview: URL.createObjectURL(blob), gtUrl: gtBlob ? URL.createObjectURL(gtBlob) : null });
  }
  function setDbInput(r) {
    return setInput({ type: "db", id: r.id, label: r.filename.split("/").pop(), preview: r.url, gtUrl: r.mask_url, kind: r.kind });
  }
  const drop = $("#drop"), file = $("#file");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => { if (file.files[0]) setFileInput(file.files[0], file.files[0].name, null); file.value = ""; });
  function dnd(zone, cb) {
    ["dragover", "dragenter"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
    zone.addEventListener("drop", (e) => cb([...e.dataTransfer.files].filter((f) => /^image\//.test(f.type) || /\.(png|jpe?g|bmp)$/i.test(f.name))));
  }
  dnd(drop, (fs) => {
    if (!fs.length) return;
    const img = fs.find((f) => !/[_-](mask|gt)\./i.test(f.name)) || fs[0], gt = fs.find((f) => f !== img && /[_-](mask|gt)\./i.test(f.name));
    setFileInput(img, img.name, gt || null);
  });
  $("#gtFile").addEventListener("change", async (e) => {
    const f = e.target.files[0]; e.target.value = ""; if (!f) return;
    if (!S.input) { toast("Hãy chọn ảnh trước, sau đó mới tải mặt nạ chuẩn.", "err"); return; }
    S.input.gtBlob = f; S.input.gtUrl = URL.createObjectURL(f);
    S.imgs.gt = await loadImg(S.input.gtUrl); S.gtEdge = edgesOf(S.imgs.gt);
    $(".file-line").classList.add("has"); $("#gtName").textContent = f.name;
    $("#curMeta").textContent = `${S.input.w}×${S.input.h} px · có mặt nạ chuẩn`; draw();
    toast("Đã thêm mặt nạ chuẩn. Bấm Phân đoạn để tính Dice.", "ok");
  });
  $("#curClear").addEventListener("click", () => {
    S.input = null; S.res = null; S.imgs = {}; $("#cur").hidden = true; $("#emptyStage").hidden = false; cv.width = 1; cv.height = 1;
    ["#run", "#cmpRun", "#robRun", "#robSweep"].forEach((s) => ($(s).disabled = true));
    $("#resBox").innerHTML = '<div class="empty"><svg><use href="#i-dice"/></svg><p>Chỉ số sẽ hiện ở đây sau khi phân đoạn.</p></div>'; $("#resAct").hidden = true;
    $(".file-line").classList.remove("has"); $("#gtName").textContent = "Mặt nạ chuẩn (tùy chọn)";
  });
  $$("[data-sim]").forEach((b) => b.addEventListener("click", async () => {
    busy(b, true);
    try {
      const r = await api(`/api/sample?type=${b.dataset.sim}&domain=${$("#simB").checked ? "B" : "A"}&seed=${Math.floor(Math.random() * 1e6)}`);
      await setFileInput(await toBlob(r.image), `Mô phỏng ${KIND[r.kind].toLowerCase()} · miền ${r.domain} · #${r.seed}`, await toBlob(r.gt));
      runSeg();
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  }));
  async function quickDemo() {
    try {
      const j = await api(`/api/images?split=test&kind=${Math.random() < 0.5 ? "malignant" : "benign"}&per=1&page=${1 + Math.floor(Math.random() * 20)}`);
      if (j.rows[0]) { await setDbInput(j.rows[0]); runSeg(); }
    } catch (e) { toast(e.message, "err"); }
  }
  $("#quickDemo").addEventListener("click", quickDemo);

  // ---- vẽ khung xem
  function edgesOf(im) {
    const t = document.createElement("canvas"); t.width = S.input.w; t.height = S.input.h;
    const x = t.getContext("2d", { willReadFrequently: true }); x.imageSmoothingEnabled = false; x.drawImage(im, 0, 0, t.width, t.height);
    const d = x.getImageData(0, 0, t.width, t.height).data, w = t.width, h = t.height, m = new Uint8Array(w * h), e = [];
    for (let i = 0; i < m.length; i++) m[i] = d[i * 4] > 127 ? 1 : 0;
    for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) { const i = y * w + xx; if (m[i] && (xx === 0 || y === 0 || xx === w - 1 || y === h - 1 || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w])) e.push(i); }
    return { m, e };
  }
  function paintEdge(img, ed, rgb) {
    const w = S.input.w, h = S.input.h, t = Math.max(1, Math.round(Math.max(w, h) / 320));
    for (const i of ed) {
      const x0 = i % w, y0 = (i / w) | 0;
      for (let dy = 0; dy < t; dy++) for (let dx = 0; dx < t; dx++) {
        const xx = Math.min(w - 1, x0 + dx), yy = Math.min(h - 1, y0 + dy), k = (yy * w + xx) * 4;
        img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2];
      }
    }
  }
  function composeOverlay() {
    const w = S.input.w, h = S.input.h; ctx.drawImage(S.imgs.image, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h), a = +$("#op").value / 100;
    if (S.maskBits) for (let i = 0; i < S.maskBits.length; i++) if (S.maskBits[i]) { const k = i * 4; d.data[k] = d.data[k] * (1 - a) + 255 * a; d.data[k + 1] = d.data[k + 1] * (1 - a) + 120 * a; d.data[k + 2] = d.data[k + 2] * (1 - a) + 80 * a; }
    if (S.gtEdge && $("#showGt").checked) paintEdge(d, S.gtEdge.e, [90, 230, 110]);
    if (S.predEdge) paintEdge(d, S.predEdge, [255, 220, 40]);
    return d;
  }
  function draw() {
    if (!S.input || !S.imgs.image) return;
    const w = S.input.w, h = S.input.h; cv.width = w; cv.height = h;
    const L = S.layer, has = !!S.res;
    if (L === "image" || (!has && L !== "split")) {
      ctx.drawImage(S.imgs.image, 0, 0, w, h);
      if (S.gtEdge && $("#showGt").checked) { const d = ctx.getImageData(0, 0, w, h); paintEdge(d, S.gtEdge.e, [90, 230, 110]); ctx.putImageData(d, 0, 0); }
    } else if (L === "overlay") ctx.putImageData(composeOverlay(), 0, 0);
    else if (L === "mask") ctx.drawImage(S.imgs.mask, 0, 0, w, h);
    else if (L === "prob") ctx.drawImage(S.imgs.prob, 0, 0, w, h);
    else if (L === "unc") ctx.drawImage(S.imgs.unc, 0, 0, w, h);
    else if (L === "att") { const im = S.gate === "mean" ? S.imgs.att : S.imgs["g" + S.gate]; if (im) ctx.drawImage(im, 0, 0, w, h); else ctx.drawImage(S.imgs.image, 0, 0, w, h); }
    else if (L === "split") {
      const right = has ? composeOverlay() : null;
      ctx.drawImage(S.imgs.image, 0, 0, w, h);
      if (right) { const sx = Math.round(S.split * w), left = ctx.getImageData(0, 0, sx || 1, h); ctx.putImageData(right, 0, 0); ctx.putImageData(left, 0, 0); }
    }
    $("#splitH").hidden = L !== "split";
    if (L === "split") $("#splitH").style.left = `calc(${cvBox().left}px + ${S.split * cvBox().width}px)`;
    $("#gates").hidden = !(L === "att" && has && S.res.gates.length);
    fitCanvas();
  }
  function cvBox() { const s = $("#stage").getBoundingClientRect(), c = cv.getBoundingClientRect(); return { left: c.left - s.left, width: c.width }; }
  function fitCanvas() {
    const st = $("#stage"); if (!S.input || !st.offsetWidth) return;
    const W = st.clientWidth, H = st.clientHeight, s = Math.min(W / S.input.w, H / S.input.h);
    cv.style.width = Math.floor(S.input.w * s) + "px"; cv.style.height = Math.floor(S.input.h * s) + "px";
    if (S.layer === "split") $("#splitH").style.left = `${(W - S.input.w * s) / 2 + S.split * S.input.w * s}px`;
  }
  let rzT; window.addEventListener("resize", () => { fitCanvas(); clearTimeout(rzT); rzT = setTimeout(() => { if (["dash", "eval", "exp"].includes(page) && $("#modal").hidden) RENDER[page](true); }, 300); });
  function setLayer(l) {
    S.layer = l; $$("#layerCtl button").forEach((b) => b.classList.toggle("on", b.dataset.layer === l)); draw();
  }
  $$("#layerCtl button").forEach((b) => b.addEventListener("click", () => {
    if (!S.res && !["image", "split"].includes(b.dataset.layer)) { toast("Hãy chạy phân đoạn trước.", ""); return; }
    if (b.dataset.layer === "att" && S.res && !S.res.attention) { toast("Mô hình này không có Attention Gate – hiển thị bản đồ xác suất.", ""); setLayer("prob"); return; }
    setLayer(b.dataset.layer);
  }));
  $("#showGt").addEventListener("change", draw);
  $("#op").addEventListener("input", (e) => { $("#opv").textContent = e.target.value + "%"; if (S.layer === "overlay" || S.layer === "split") draw(); });
  $("#thr").addEventListener("input", (e) => ($("#thrv").textContent = F(e.target.value, 2)));
  $("#zoomBtn").addEventListener("click", () => { $("#stage").classList.toggle("fs"); fitCanvas(); });
  // kéo thanh trước/sau
  (() => {
    let drag = false; const st = $("#stage");
    const mv = (e) => { if (!drag) return; const c = cv.getBoundingClientRect(), x = (e.touches ? e.touches[0].clientX : e.clientX) - c.left; S.split = Math.max(0, Math.min(1, x / c.width)); draw(); };
    st.addEventListener("mousedown", (e) => { if (S.layer === "split") { drag = true; mv(e); } });
    st.addEventListener("touchstart", (e) => { if (S.layer === "split") { drag = true; mv(e); } }, { passive: true });
    window.addEventListener("mousemove", mv); window.addEventListener("touchmove", mv, { passive: true });
    window.addEventListener("mouseup", () => (drag = false)); window.addEventListener("touchend", () => (drag = false));
  })();

  function form(extra = {}) {
    const f = new FormData(), i = S.input;
    if (i.type === "db") { f.append("image_id", i.id); if (i.gtBlob) f.append("gt", i.gtBlob, "gt.png"); }
    else { f.append("image", i.blob, "image.png"); f.append("label", i.label); if (i.gtBlob) f.append("gt", i.gtBlob, "gt.png"); }
    f.append("model", $("#model").value); f.append("threshold", $("#thr").value); f.append("post", $("#post").checked ? "1" : "0");
    f.append("px_mm", $("#pxmm").value.trim()); f.append("save", $("#save").checked ? "1" : "0");
    for (const k in extra) f.append(k, extra[k]);
    return f;
  }
  async function runSeg() {
    if (!S.input) return;
    const b = $("#run"); busy(b, true); $("#stageLoad").hidden = false;
    try {
      const r = await api("/api/segment", { method: "POST", body: form() });
      S.res = r;
      const [mask, prob, unc, att, ...gates] = await Promise.all([loadImg(r.mask), loadImg(r.prob), loadImg(r.uncertainty), r.attention ? loadImg(r.attention) : null, ...r.gates.map(loadImg)]);
      Object.assign(S.imgs, { mask, prob, unc, att }); gates.forEach((g, i) => (S.imgs["g" + i] = g));
      const ed = edgesOf(mask); S.maskBits = ed.m; S.predEdge = ed.e;
      $("#gates").innerHTML = r.gates.length ? [`<button data-g="mean" class="${S.gate === "mean" ? "on" : ""}"><img src="${r.attention}"><span>TB</span></button>`, ...r.gates.map((g, i) => `<button data-g="${i}" class="${S.gate == i ? "on" : ""}" title="Attention Gate ${i + 1}${i === 0 ? " (sâu nhất)" : i === 3 ? " (nông nhất)" : ""}"><img src="${g}"><span>AG${i + 1}</span></button>`)].join("") : "";
      $$("#gates button").forEach((x) => x.addEventListener("click", () => { S.gate = x.dataset.g; $$("#gates button").forEach((y) => y.classList.toggle("on", y === x)); draw(); }));
      $('#layerCtl [data-layer="att"]').disabled = !r.attention;
      if (S.layer === "image" || (S.layer === "att" && !r.attention)) S.layer = "overlay";
      setLayer(S.layer);
      renderResult(r);
      if (r.case_id) updateNavCases();
    } catch (e) { toast(e.message, "err", "Không phân đoạn được"); }
    $("#stageLoad").hidden = true; busy(b, false);
  }
  $("#run").addEventListener("click", runSeg);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || page !== "seg" || $("#run").disabled || !$("#modal").hidden) return;
    if (e.target.closest("button, a, select, textarea, .drop")) return;
    e.preventDefault(); runSeg();
  });

  function shapeWord(m) {
    if (!m.detected) return "–";
    return m.solidity < 0.85 || m.circularity < 0.6 ? "Không đều" : m.aspect_ratio >= 1.35 ? "Bầu dục, bờ đều" : "Gần tròn, bờ đều";
  }
  const LV = { high: ["ok", "check", "Độ tin cậy cao"], medium: ["warn", "help", "Độ tin cậy trung bình"], low: ["bad", "x", "Độ tin cậy thấp"], none: ["", "help", "Không có vùng khối u"] };
  const meter = (label, v, txt, good, hint) => `<div class="meter ${good}"><div class="mt-h"><span>${label}</span><b>${txt}</b></div><div class="bar"><i style="width:0" data-w="${Math.max(0, Math.min(1, v)) * 100}%"></i></div>${hint ? `<small>${hint}</small>` : ""}</div>`;
  const mBox = (label, val, bar) => `<div class="m"><span>${label}</span><b>${val}</b>${bar !== undefined && bar !== null ? `<div class="bar"><i style="width:0" data-w="${Math.max(0, Math.min(1, bar)) * 100}%"></i></div>` : ""}</div>`;
  function confText(c, m) {
    if (!m.detected) return c.weak ? "Có vùng mô hình nghi ngờ yếu (xác suất cao nhất " + F(c.max_prob, 2) + ") nhưng dưới ngưỡng. Thử hạ ngưỡng hoặc xem lớp Xác suất." : "Mô hình không thấy vùng nào giống khối u (xác suất cao nhất " + F(c.max_prob, 2) + ").";
    if (c.level === "high") return "Vùng khối u rõ, kết quả gần như không đổi khi thay ngưỡng. Có thể dùng trực tiếp các số đo kích thước.";
    if (c.level === "medium") return "Biên khối u chưa rõ ở một số đoạn – kích thước có thể sai lệch vài điểm ảnh. Xem lớp <b>Độ không chắc</b> để biết đoạn nào.";
    return "Kết quả phụ thuộc nhiều vào ngưỡng hoặc mô hình chưa chắc chắn. Nên kiểm tra lại bằng mắt, so sánh với mô hình khác.";
  }
  function renderResult(r) {
    const m = r.metrics, c = r.confidence || { level: m.detected ? "high" : "none" }, hasGt = m.dice !== undefined, mm = m.area_mm2 !== undefined, L = c.loc;
    const lv = LV[c.level] || LV.none;
    let h = `<div class="res-head"><span class="chip ${m.detected ? "run" : ""}">${m.detected ? "Phát hiện vùng khối u" : "Không phát hiện khối u"}</span><span class="conf ${lv[0]}">${icon(lv[1])}${lv[2]}</span></div>`;
    if (m.detected) {
      h += `<div class="facts">
        <div><span>Kích thước</span><b>${mm ? F(m.max_diameter_mm, 1) + " mm" : F(m.max_diameter_px, 0) + " px"}</b><small>${L ? (mm ? `${F(L.bw_mm, 1)} × ${F(L.bh_mm, 1)} mm` : `${L.bw} × ${L.bh} px`) + " (rộng × cao)" : "đường kính lớn nhất"}</small></div>
        <div><span>Diện tích</span><b>${mm ? F(m.area_mm2, 1) + " mm²" : F(m.area_ratio, 1) + "%"}</b><small>${mm ? F(m.area_ratio, 1) + "% khung ảnh" : "của khung ảnh · nhập mm/px để đổi ra mm"}</small></div>
        <div><span>Vị trí</span><b>${L ? `${L.horiz[0].toUpperCase() + L.horiz.slice(1)}, ${L.depth === "vừa" ? "sâu vừa" : L.depth}` : "–"}</b><small>${L ? (L.depth_mm != null ? `bờ trên cách mặt da ${F(L.depth_mm, 1)} mm` : `tâm ở ${F(L.cx, 0)}% ngang, ${F(L.cy, 0)}% dọc`) : ""}</small></div>
        <div><span>Hình dạng</span><b>${shapeWord(m)}</b><small>tròn ${F(m.circularity, 2)} · đặc ${F(m.solidity, 2)} · trục ${F(m.aspect_ratio, 2)}</small></div>
      </div>`;
      h += `<div class="sec-t">Mức độ chắc chắn</div>
        ${meter("Xác suất trung bình trong vùng", c.prob_in, F(c.prob_in, 2), c.prob_in >= 0.85 ? "g" : c.prob_in >= 0.7 ? "y" : "r")}
        ${meter("Ổn định khi đổi ngưỡng 0,3 ↔ 0,7", c.stability, F(c.stability, 2), c.stability >= 0.9 ? "g" : c.stability >= 0.75 ? "y" : "r")}
        ${meter("Điểm ảnh lưỡng lự ở biên", Math.min(1, c.border / 60), F(c.border, 0) + "% diện tích", c.border <= 25 ? "g" : c.border <= 45 ? "y" : "r")}`;
    }
    h += `<div class="note">${confText(c, m)}</div>`;
    const leak = /-STU$/.test(r.model) && /^stu_/.test(r.label || "");
    if (hasGt && leak) h += `<div class="note warn-note">Ảnh này thuộc 42 ảnh thật mà <b>${esc(r.model)}</b> đã học, nên Dice dưới đây cao hơn thực tế. Độ chính xác khách quan của mô hình là Dice kiểm định chéo ≈ ${F((S.models.find((m) => m.id === r.model) || {}).val_dice, 3)} (trang Kết quả thực nghiệm).</div>`;
    if (hasGt) h += `<div class="sec-t">Đối chiếu mặt nạ chuẩn</div><div class="res-top"><div id="gauge"></div><div class="status"><span class="chip ${m.dice >= 0.8 ? "ok" : m.dice >= 0.5 ? "warn" : "bad"}">${m.dice >= 0.8 ? "Trùng khớp tốt" : m.dice >= 0.5 ? "Trùng khớp một phần" : "Sai lệch lớn"}</span><p>IoU ${F(m.iou, 3)} · HD95 ${m.hd95_px === null ? "–" : F(m.hd95_px, 1) + " px"}</p><p>Precision ${F(m.precision, 3)} · Recall ${F(m.recall, 3)}</p></div></div>`;
    h += `<div class="res-foot">${icon("clock")}Suy luận ${F(r.inference_ms, 0)} ms trên CPU · ${esc(r.model)} · ngưỡng ${F(r.threshold, 2)}</div>`;
    $("#resBox").innerHTML = h; $("#resAct").hidden = false;
    $("#runInfo").textContent = r.case_id ? `đã lưu ca #${r.case_id}` : "";
    if (hasGt) Charts.gauge($("#gauge"), m.dice, "DICE");
    requestAnimationFrame(() => requestAnimationFrame(() => $$("#resBox .bar i").forEach((i) => (i.style.width = i.dataset.w))));
  }
  const base = () => (S.input.label || "anh").replace(/[^\p{L}\p{N}_-]+/gu, "_").slice(0, 40);
  $("#dlMask").addEventListener("click", () => download(S.res.mask, `${base()}_${S.res.model}_mask.png`));
  $("#dlOver").addEventListener("click", () => { const l = S.layer; S.layer = "overlay"; draw(); download(cv.toDataURL("image/png"), `${base()}_${S.res.model}_overlay.png`); S.layer = l; draw(); });
  $("#dlCsv").addEventListener("click", () => {
    const r = S.res, m = r.metrics, keys = ["detected", "area_px", "area_ratio", "area_mm2", "max_diameter_px", "max_diameter_mm", "circularity", "solidity", "aspect_ratio", "dice", "iou", "precision", "recall", "hd95_px"];
    const c = r.confidence || {}, ck = ["level", "prob_in", "stability", "border"];
    const rows = [["image", "model", "threshold", ...keys, ...ck.map((k) => "conf_" + k), "inference_ms"], [S.input.label, r.model, r.threshold, ...keys.map((k) => m[k] ?? ""), ...ck.map((k) => c[k] ?? ""), r.inference_ms]];
    download(URL.createObjectURL(new Blob(["﻿" + rows.map((x) => x.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n")], { type: "text/csv" })), `${base()}_${r.model}.csv`);
  });
  $("#print").addEventListener("click", () => {
    const r = S.res, m = r.metrics, row = (a, b) => `<tr><th>${a}</th><td>${b}</td></tr>`, l0 = S.layer;
    S.layer = "image"; draw(); const o1 = cv.toDataURL(); S.layer = "overlay"; draw(); const o2 = cv.toDataURL(); S.layer = r.attention ? "att" : "prob"; draw(); const o3 = cv.toDataURL(); S.layer = l0; draw();
    $("#report").innerHTML = `<div class="rh"><div><h1>Phiếu kết quả phân đoạn khối u trên ảnh siêu âm vú</h1><div>SonoSeg · U-Net kết hợp Attention</div></div><div>${new Date().toLocaleString("vi-VN")}${r.case_id ? "<br>Mã ca: #" + r.case_id : ""}</div></div>
      <h2>Thông tin xử lý</h2><table>${row("Ảnh", esc(S.input.label) + ` (${r.size[0]}×${r.size[1]} px)`)}${row("Mô hình", `${esc(r.model)} – ${esc(mname(r.model))}`)}${row("Ngưỡng / hậu xử lý", `${F(r.threshold, 2)} / ${$("#post").checked ? "giữ vùng liên thông lớn nhất" : "không"}`)}${row("Kích thước điểm ảnh", $("#pxmm").value.trim() ? esc($("#pxmm").value) + " mm/px" : "không khai báo")}</table>
      <h2>Hình ảnh</h2><div class="imgs"><figure><img src="${o1}"><figcaption>Ảnh gốc${S.gtEdge ? " (viền xanh: mặt nạ chuẩn)" : ""}</figcaption></figure><figure><img src="${o2}"><figcaption>Vùng dự đoán (viền vàng)</figcaption></figure><figure><img src="${o3}"><figcaption>${r.attention ? "Bản đồ attention" : "Bản đồ xác suất"}</figcaption></figure></div>
      <h2>Chỉ số</h2><table>${row("Phát hiện khối u", m.detected ? "Có" : "Không")}${m.detected ? row("Diện tích", `${F(m.area_ratio, 2)}% ảnh (${m.area_px} px)${m.area_mm2 !== undefined ? ` ≈ ${F(m.area_mm2, 1)} mm²` : ""}`) + row("Đường kính lớn nhất", `${F(m.max_diameter_px, 1)} px${m.max_diameter_mm !== undefined ? ` ≈ ${F(m.max_diameter_mm, 1)} mm` : ""}`) + row("Độ tròn / độ đặc / tỉ lệ trục", `${F(m.circularity, 2)} / ${F(m.solidity, 2)} / ${F(m.aspect_ratio, 2)}`) + (r.confidence && r.confidence.loc ? row("Vị trí", `${r.confidence.loc.horiz}, độ sâu ${r.confidence.loc.depth}${r.confidence.loc.depth_mm != null ? ` (bờ trên cách mặt da ${F(r.confidence.loc.depth_mm, 1)} mm)` : ""}`) + row("Độ tin cậy", `${LV[r.confidence.level][2]} · xác suất TB ${F(r.confidence.prob_in, 2)} · ổn định ngưỡng ${F(r.confidence.stability, 2)}`) : "") : ""}
      ${m.dice !== undefined ? row("Dice / IoU", `${F(m.dice, 3)} / ${F(m.iou, 3)}`) + row("Precision / Recall / HD95", `${F(m.precision, 3)} / ${F(m.recall, 3)} / ${m.hd95_px === null ? "–" : F(m.hd95_px, 1) + " px"}`) : ""}${row("Thời gian suy luận", F(r.inference_ms, 1) + " ms (CPU)")}</table>
      <p>Hình dạng: <b>${shapeWord(m)}</b>. ${LV[(r.confidence || {}).level || "none"][2]}: ${confText(r.confidence || {}, m)}</p><p class="foot">Kết quả do mô hình học sâu tạo ra, phục vụ nghiên cứu và minh họa; mô hình được huấn luyện trên ảnh siêu âm mô phỏng. Phiếu không có giá trị chẩn đoán và không thay thế ý kiến của bác sĩ chuyên khoa.</p>`;
    setTimeout(() => window.print(), 100);
  });

  // ---- công cụ: so sánh / độ bền
  $$("#toolTabs button").forEach((b) => b.addEventListener("click", () => { $$("#toolTabs button").forEach((x) => x.classList.toggle("on", x === b)); $$(".tool").forEach((t) => (t.hidden = t.dataset.tool !== b.dataset.tool)); }));
  $("#cmpRun").addEventListener("click", async () => {
    const b = $("#cmpRun"); busy(b, true);
    try {
      const f = form({ models: S.models.map((m) => m.id).join(","), save: "0" }); f.delete("model");
      const r = await api("/api/compare", { method: "POST", body: f }), hasGt = r.rows[0] && r.rows[0].metrics.dice !== undefined;
      const best = hasGt ? Math.max(...r.rows.map((x) => x.metrics.dice ?? -1)) : null;
      $("#cmpGrid").innerHTML = r.rows.map((x, i) => `<div class="cm${hasGt && x.metrics.dice === best ? " win" : ""}" style="animation-delay:${i * 50}ms"><img src="${x.overlay}" alt="${esc(x.id)} – ${esc(x.name)}"><div class="cb"><b title="${esc(x.name)}"><i class="dot" style="background:${mcol(x.id)}"></i>${esc(x.id)} <span class="muted ell" style="font-weight:500">${esc(x.name)}</span></b>${hasGt ? `Dice <b style="display:inline">${F(x.metrics.dice, 3)}</b> · IoU ${F(x.metrics.iou, 3)}<br>` : ""}${x.metrics.detected ? `${F(x.metrics.area_ratio, 2)}% · độ tròn ${F(x.metrics.circularity, 2)}` : "Không phát hiện"} · ${F(x.inference_ms, 1)} ms</div></div>`).join("");
      $("#cmpMore").hidden = false; $("#consImg").src = r.consensus;
      $("#consTxt").innerHTML = `Các mô hình thống nhất trên <b>${F(r.agreement, 1)}%</b> số điểm ảnh; trong vùng ít nhất một mô hình cho là khối u, <b>${F(r.full_agreement_in_union, 1)}%</b> được tất cả cùng xác định. Màu sáng: nhiều mô hình đồng ý.`;
      const vals = r.rows.map((x) => (hasGt ? x.metrics.dice ?? 0 : x.metrics.area_ratio));
      $("#cmpChartT").textContent = hasGt ? "Dice theo mô hình" : "Diện tích dự đoán (% ảnh)";
      const lo = hasGt ? Math.max(0, Math.floor((Math.min(...vals) - 0.04) * 20) / 20) : 0, hi = hasGt ? 1 : Math.max(1, Math.ceil(Math.max(...vals) * 1.3));
      Charts.dot($("#cmpChart"), { categories: r.rows.map((x) => x.id), series: [{ name: hasGt ? "Dice" : "Diện tích (%)", values: vals, colors: r.rows.map((x) => mcol(x.id)) }], xMin: lo, xMax: hi, dec: hasGt ? 3 : 2, ml: 50, rowH: 38, xTitle: hasGt ? "Dice so với mặt nạ chuẩn" : "% diện tích ảnh", legend: false, r: 7 });
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });
  const lvLabel = () => { const L = LEVELS[S.robKind][+$("#robLevel").value]; return `${KIND_TXT[S.robKind][1]} = ${F(L, 2)}${(S.robKind === "contrast" ? L === 1 : L === 0) ? " (gốc)" : ""}`; };
  $("#robLevel").addEventListener("input", () => ($("#robLv").textContent = lvLabel()));
  $$("#robKind button").forEach((b) => b.addEventListener("click", () => { S.robKind = b.dataset.kind; $$("#robKind button").forEach((x) => x.classList.toggle("on", x === b)); $("#robLv").textContent = lvLabel(); }));
  $("#robRun").addEventListener("click", async () => {
    const b = $("#robRun"); busy(b, true);
    try {
      const L = LEVELS[S.robKind][+$("#robLevel").value];
      const r = await api("/api/robustness", { method: "POST", body: form({ kind: S.robKind, level: L, models: S.models.filter((m) => m.source === "builtin").map((m) => m.id).join(",") }) });
      $("#robGrid").innerHTML = `<div class="cm"><img src="${r.image}" alt="Ảnh sau biến đổi"><div class="cb"><b>Ảnh sau biến đổi</b>${KIND_TXT[r.kind][0]} · ${lvLabel()}${r.psnr ? ` · PSNR ${F(r.psnr, 1)} dB` : ""}</div></div>` +
        r.rows.map((x, i) => `<div class="cm" style="animation-delay:${(i + 1) * 50}ms"><img src="${x.overlay}" alt="${x.id}"><div class="cb"><b><i class="dot" style="background:${mcol(x.id)}"></i>${x.id}</b>${x.metrics.dice !== undefined ? `Dice <b style="display:inline">${F(x.metrics.dice, 3)}</b>` : x.metrics.detected ? `${F(x.metrics.area_ratio, 2)}% diện tích` : "không phát hiện"}</div></div>`).join("");
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });
  $("#robSweep").addEventListener("click", async () => {
    const b = $("#robSweep"); busy(b, true);
    try {
      const r = await api("/api/robustness", { method: "POST", body: form({ kind: S.robKind, mode: "sweep", models: S.models.filter((m) => m.source === "builtin").map((m) => m.id).join(",") }) });
      const ids = Object.keys(r.series), all = ids.flatMap((k) => r.series[k]), isD = r.metric === "dice";
      Charts.line($("#robChart"), { x: r.levels, xLabels: r.levels.map((v) => F(v, 2)), series: ids.map((k) => ({ name: k, values: r.series[k], color: mcol(k) })), yMin: isD ? Math.max(0, Math.floor((Math.min(...all) - 0.03) * 20) / 20) : 0, yMax: isD ? 1 : Math.ceil(Math.max(...all) * 1.2 + 0.1), height: 240, xTitle: KIND_TXT[r.kind][0] + " (" + KIND_TXT[r.kind][1] + ")", yTitle: isD ? "Dice" : "% diện tích", dec: 3 });
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });

  // ================================================================== HÀNG LOẠT
  let bFiles = [];
  function setBatch(fs, note = "") {
    bFiles = fs; const masks = fs.filter((f) => /[_-](mask|gt|label)\.[a-z]+$/i.test(f.name)).length;
    $("#bInfo").textContent = fs.length ? `${fs.length - masks} ảnh, ${masks} mặt nạ chuẩn${note}.` : "Chưa chọn tệp.";
    $("#bRun").disabled = !(fs.length - masks);
  }
  $("#bfiles").addEventListener("change", (e) => setBatch([...e.target.files]));
  $("#bdrop").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#bfiles").click(); } });
  dnd($("#bdrop"), (fs) => setBatch(fs));
  $("#bDemo").addEventListener("click", async () => {
    const b = $("#bDemo"); busy(b, true);
    try {
      const j = await api("/api/images?split=test&per=24&page=" + (1 + Math.floor(Math.random() * 3))), fs = [];
      await Promise.all(j.rows.map(async (r) => {
        const n = r.filename.split("/").pop().replace(".png", "");
        fs.push(new File([await toBlob(r.url)], n + ".png", { type: "image/png" }));
        if (r.mask_url) fs.push(new File([await toBlob(r.mask_url)], n + "_mask.png", { type: "image/png" }));
      }));
      setBatch(fs, " (lấy từ tập kiểm tra trong CSDL)");
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });
  $("#bRun").addEventListener("click", async () => {
    const b = $("#bRun"); busy(b, true); $("#bProg").hidden = false;
    try {
      const f = new FormData(); bFiles.forEach((x) => f.append("images", x, x.name));
      f.append("model", $("#bModel").value); f.append("threshold", $("#thr").value); f.append("post", "1");
      const r = await api("/api/batch", { method: "POST", body: f }), s = r.summary;
      $("#bEmpty").hidden = true; $("#bRes").hidden = false; $("#bZip").href = r.zip; $("#bZip").setAttribute("download", "");
      $("#bKpi").innerHTML = kpi("Ảnh xử lý được", `${s.ok}/${s.n}`, "", "image") + kpi("Phát hiện khối u", s.detected, "", "scan") + kpi("Dice trung bình", s.mean_dice === null ? "–" : F(s.mean_dice, 3), s.mean_dice === null ? "cần mặt nạ chuẩn" : `± ${F(s.std_dice, 3)} · ${s.with_gt} ảnh`, "dice") + kpi("Thời gian / ảnh", F(s.mean_ms, 1) + " ms", r.model, "clock");
      $("#bCharts").hidden = false;
      if (s.dices.length) Charts.hist($("#bHist"), { values: s.dices, min: 0, max: 1, bins: 10, dec: 1, color: Charts.color(0), xTitle: "Dice", labels: true });
      else $("#bHist").innerHTML = '<div class="empty sm"><p>Cần mặt nạ chuẩn để tính Dice.</p></div>';
      Charts.hist($("#bArea"), { values: s.areas, min: 0, max: Math.max(5, Math.ceil(Math.max(...s.areas, 1))), bins: 8, dec: 1, color: Charts.color(2), xTitle: "% diện tích ảnh" });
      $("#bTbl").innerHTML = `<thead><tr><th></th><th>Tệp</th><th>Kết quả</th><th class="n">Diện tích %</th><th class="n">Đường kính px</th><th class="n">Độ tròn</th><th class="n">Dice</th><th class="n">IoU</th><th class="n">ms</th></tr></thead><tbody>` +
        r.rows.map((x) => x.ok ? `<tr><td><img src="${x.thumb}" alt="${esc(x.name)}"></td><td><b>${esc(x.name)}</b></td><td>${x.metrics.detected ? '<span class="chip run">Có khối u</span>' : '<span class="chip">Không phát hiện</span>'}</td><td class="n">${F(x.metrics.area_ratio, 2)}</td><td class="n">${F(x.metrics.max_diameter_px, 1)}</td><td class="n">${F(x.metrics.circularity, 2)}</td><td class="n">${F(x.metrics.dice, 3)}</td><td class="n">${F(x.metrics.iou, 3)}</td><td class="n">${F(x.inference_ms, 1)}</td></tr>`
          : `<tr><td></td><td>${esc(x.name)}</td><td colspan="7" style="color:var(--bad-2)">${esc(x.error)}</td></tr>`).join("") + "</tbody>";
      toast(`Đã xử lý ${s.ok} ảnh.`, "ok");
    } catch (e) { toast(e.message, "err"); }
    $("#bProg").hidden = true; busy(b, false); b.disabled = !bFiles.length;
  });
  RENDER.batch = () => { if (!S.models.length) loadModels(); };

  // ================================================================== LỊCH SỬ
  let hPage = 1;
  RENDER.history = async () => {
    try {
      const st = await api("/api/stats");
      $("#hKpi").innerHTML = kpi("Tổng số ca", st.cases.n || 0, "", "clock") + kpi("Ca phát hiện khối u", st.cases.det || 0, "", "scan") + kpi("Dice trung bình", st.cases.dice ? F(st.cases.dice, 3) : "–", "các ca có mặt nạ chuẩn", "dice") + kpi("Thời gian TB", st.cases.ms ? F(st.cases.ms, 1) + " ms" : "–", "mỗi ca", "cpu");
      const bm = st.cases_by_model;
      if (bm.length) Charts.bar($("#hByModel"), { categories: bm.map((x) => x.model_id), series: [{ name: "Số ca", values: bm.map((x) => x.n), colors: bm.map((x) => mcol(x.model_id)) }], yMin: 0, yMax: Math.max(4, Math.ceil(Math.max(...bm.map((x) => x.n)) * 1.2)), dec: 0, tickDec: 0, height: 220, legend: false });
      else $("#hByModel").innerHTML = '<div class="empty sm"><p>Chưa có ca nào.</p></div>';
      const all = await api("/api/cases?per=200"), d = all.rows.filter((x) => x.dice !== null).map((x) => x.dice);
      if (d.length) Charts.hist($("#hDice"), { values: d, min: 0, max: 1, bins: 10, dec: 1, color: Charts.color(0), xTitle: "Dice", height: 220 });
      else $("#hDice").innerHTML = '<div class="empty sm"><p>Chưa có ca nào có mặt nạ chuẩn.</p></div>';
      const sel = $("#hModel"), cur = sel.value;
      sel.innerHTML = '<option value="">Mọi mô hình</option>' + S.models.map((m) => `<option value="${m.id}">${m.id}</option>`).join(""); sel.value = cur;
      loadCases();
    } catch (e) { toast(e.message, "err"); }
  };
  async function loadCases() {
    const j = await api(`/api/cases?page=${hPage}&per=15&model=${$("#hModel").value}&q=${encodeURIComponent($("#hQ").value.trim())}`);
    $("#hTbl").innerHTML = `<thead><tr><th></th><th>#</th><th>Thời gian</th><th>Ảnh</th><th>Mô hình</th><th class="n">Diện tích %</th><th class="n">Đường kính</th><th class="n">Độ tròn</th><th class="n">Dice</th><th class="n">IoU</th><th class="n">ms</th><th></th></tr></thead><tbody>` +
      (j.rows.length ? j.rows.map((c) => `<tr><td>${c.thumb ? `<img src="/data/${c.thumb}" alt="${esc(c.image_label)}">` : ""}</td><td class="muted">${c.id}</td><td>${esc(c.created_at.replace("T", " "))}</td><td><b>${esc(c.image_label)}</b></td><td>${mtag(c.model_id)}</td><td class="n">${F(c.area_ratio, 2)}</td><td class="n">${c.diameter_mm !== null ? F(c.diameter_mm, 1) + " mm" : F(c.diameter_px, 1) + " px"}</td><td class="n">${F(c.circularity, 2)}</td><td class="n">${F(c.dice, 3)}</td><td class="n">${F(c.iou, 3)}</td><td class="n">${F(c.ms, 1)}</td><td><div class="acts"><button class="icon-btn sm" data-del="${c.id}" title="Xóa"><svg><use href="#i-trash"/></svg></button></div></td></tr>`).join("")
        : `<tr><td colspan="12"><div class="empty"><svg><use href="#i-clock"/></svg><p>Không có ca phù hợp.</p></div></td></tr>`) + "</tbody>";
    $$("#hTbl [data-del]").forEach((b) => b.addEventListener("click", async () => { await api(`/api/cases/${b.dataset.del}`, { method: "DELETE" }); toast("Đã xóa ca #" + b.dataset.del, "ok"); RENDER.history(); updateNavCases(); }));
    pager($("#hPager"), j.total, j.per, hPage, (p) => { hPage = p; loadCases(); });
  }
  let hT; $("#hQ").addEventListener("input", () => { clearTimeout(hT); hT = setTimeout(() => { hPage = 1; loadCases(); }, 250); });
  $("#hModel").addEventListener("change", () => { hPage = 1; loadCases(); });
  $("#hClear").addEventListener("click", async () => { if (await confirmBox("Xóa toàn bộ lịch sử ca phân đoạn? Thao tác không hoàn tác được.", "Xóa hết")) { await api("/api/cases/clear", { method: "POST" }); toast("Đã xóa lịch sử.", "ok"); RENDER.history(); updateNavCases(); } });
  function pager(el, total, per, cur, cb) {
    const n = Math.ceil(total / per); if (n <= 1) { el.innerHTML = total ? `<span>${total} mục</span>` : ""; return; }
    const pages = [...new Set([1, cur - 1, cur, cur + 1, n].filter((p) => p >= 1 && p <= n))].sort((a, b) => a - b);
    let h = `<button ${cur === 1 ? "disabled" : ""} data-p="${cur - 1}">‹</button>`, prev = 0;
    pages.forEach((p) => { if (p - prev > 1) h += "<span>…</span>"; h += `<button class="${p === cur ? "on" : ""}" data-p="${p}">${p}</button>`; prev = p; });
    h += `<button ${cur === n ? "disabled" : ""} data-p="${cur + 1}">›</button><span>${total} mục</span>`;
    el.innerHTML = h; $$("button", el).forEach((b) => b.addEventListener("click", () => cb(+b.dataset.p)));
  }

  // ================================================================== CSDL
  let gPage = 1;
  RENDER.data = async () => {
    try {
      const ds = await api("/api/datasets"), tot = ds.reduce((a, b) => a + b.n, 0), by = (k, v) => ds.filter((r) => r[k] === v).reduce((a, b) => a + b.n, 0);
      $("#dKpi").innerHTML = kpi("Tổng số ảnh", tot, "đều lưu trong data/ và app.db", "db") + kpi("Huấn luyện", by("split", "train"), "miền A", "cpu") + kpi("Kiểm định", by("split", "val"), "chọn epoch tốt nhất", "check") + kpi("Kiểm tra (A)", by("split", "test"), "cùng thiết bị", "chart") + kpi("Thiết bị khác (B)", by("split", "ext"), "đánh giá khái quát", "scan") + kpi("Tải lên", by("split", "upload"), "ảnh của người dùng", "upload");
      const kinds = ["benign", "malignant", "normal", "unknown"].filter((k) => by("kind", k));
      Charts.donut($("#dDonut"), { labels: kinds.map((k) => KIND[k]), values: kinds.map((k) => by("kind", k)), colors: kinds.map((k, i) => Charts.color(k === "unknown" ? 7 : i)), center: tot, centerLabel: "ảnh" });
      const splits = ["train", "val", "test", "ext", "upload"].filter((s) => by("split", s));
      Charts.bar($("#dBars"), { categories: splits.map((s) => SPLIT[s]), series: kinds.map((k, i) => ({ name: KIND[k], values: splits.map((s) => ds.filter((r) => r.split === s && r.kind === k).reduce((a, b) => a + b.n, 0)), color: Charts.color(k === "unknown" ? 7 : i) })), yMin: 0, yMax: Math.ceil(Math.max(...splits.map((s) => Math.max(...kinds.map((k) => ds.filter((r) => r.split === s && r.kind === k).reduce((a, b) => a + b.n, 0))))) * 1.15 / 10) * 10, dec: 0, tickDec: 0, height: 250 });
      loadGallery();
    } catch (e) { toast(e.message, "err"); }
  };
  async function loadGallery() {
    const g = $("#gallery"); g.innerHTML = Array(16).fill('<div class="skeleton" style="aspect-ratio:1"></div>').join("");
    const j = await api(`/api/images?split=${$("#fSplit").value}&kind=${$("#fKind").value}&page=${gPage}&per=48`), sm = $("#fMask").checked;
    g.innerHTML = j.rows.length ? j.rows.map((r, i) => `<div class="gi" data-id="${r.id}" style="animation-delay:${Math.min(i, 24) * 15}ms"><img src="${r.url}" loading="lazy" alt="">${sm && r.mask_url ? `<img class="mk" src="${r.mask_url}" loading="lazy" alt="">` : ""}<div class="tagb"><span class="k-${r.kind}">${KIND[r.kind] || r.kind}</span><span>${r.split}</span></div><div class="nm">${esc(r.filename.split("/").pop())}</div></div>`).join("")
      : '<div class="empty" style="grid-column:1/-1"><svg><use href="#i-image"/></svg><p>Không có ảnh phù hợp bộ lọc.</p></div>';
    $$(".gi", g).forEach((el) => el.addEventListener("click", () => imageDetail(j.rows.find((r) => r.id === +el.dataset.id))));
    pager($("#gPager"), j.total, j.per, gPage, (p) => { gPage = p; loadGallery(); });
  }
  ["#fSplit", "#fKind", "#fMask"].forEach((s) => $(s).addEventListener("change", () => { gPage = 1; loadGallery(); }));
  function imageDetail(r) {
    modal(`<h3 style="margin:0 40px 14px 0">${esc(r.filename.split("/").pop())}</h3><div class="det"><div><h4>Ảnh</h4><img src="${r.url}" alt=""></div><div><h4>Mặt nạ chuẩn</h4>${r.mask_url ? `<img src="${r.mask_url}" alt="">` : '<div class="empty"><p>Không có mặt nạ.</p></div>'}</div>
      <div><dl><dt>Mã</dt><dd>#${r.id}</dd><dt>Bộ</dt><dd>${esc(r.dataset)}</dd><dt>Tập</dt><dd>${SPLIT[r.split] || r.split}</dd><dt>Loại</dt><dd>${KIND[r.kind] || r.kind}</dd><dt>Miền</dt><dd>${esc(r.domain)}</dd><dt>Kích thước</dt><dd>${r.width}×${r.height}</dd><dt>Khối u</dt><dd>${r.tumor_ratio === null ? "–" : F(r.tumor_ratio * 100, 2) + "% ảnh"}</dd><dt>Nguồn</dt><dd>${esc(r.source)}</dd></dl>
      <button class="btn primary block" id="dSeg"><svg><use href="#i-scan"/></svg>Phân đoạn ảnh này</button>
      ${r.source === "upload" ? `<label class="fld" style="margin-top:12px">Đưa vào tập<select class="sel" id="dSplit"><option value="upload">Chỉ lưu (tải lên)</option><option value="train">Huấn luyện</option><option value="val">Kiểm định</option><option value="test">Kiểm tra</option></select></label><button class="btn danger block" id="dDel"><svg><use href="#i-trash"/></svg>Xóa ảnh</button>` : '<p class="muted sm" style="margin-top:12px">Ảnh thuộc bộ ảnh mẫu, được giữ cố định để kết quả đánh giá có thể so sánh.</p>'}</div></div>`);
    $("#dSeg").onclick = async () => { closeModal(); go("seg"); await setDbInput(r); runSeg(); };
    if ($("#dDel")) $("#dDel").onclick = async () => { await api(`/api/images/${r.id}`, { method: "DELETE" }).then(() => { closeModal(); toast("Đã xóa ảnh.", "ok"); RENDER.data(); }).catch((e) => toast(e.message, "err")); };
    if ($("#dSplit")) { $("#dSplit").value = r.split; $("#dSplit").onchange = async (e) => { try { await api(`/api/images/${r.id}/split`, { json: { split: e.target.value } }); toast("Đã cập nhật tập.", "ok"); RENDER.data(); } catch (er) { toast(er.message, "err"); } }; }
  }
  $("#upOpen").addEventListener("click", () => {
    modal(`<h3 style="margin:0 40px 6px 0">Thêm ảnh vào cơ sở dữ liệu</h3><p class="muted sm">Chọn ảnh siêu âm; mặt nạ chuẩn (nếu có) đặt cùng tên với hậu tố <code>_mask</code>. Ảnh có mặt nạ có thể đưa vào tập huấn luyện hoặc kiểm tra.</p>
      <label class="drop" id="upDrop" tabindex="0" style="margin:12px 0"><input type="file" id="upFiles" multiple accept="image/png,image/jpeg,image/bmp" hidden><svg><use href="#i-upload"/></svg><b>Chọn ảnh</b><small id="upInfo">Chưa chọn tệp</small></label>
      <label class="fld">Loại tổn thương<select class="sel" id="upKind"><option value="unknown">Chưa rõ</option><option value="benign">Lành tính</option><option value="malignant">Ác tính</option><option value="normal">Bình thường</option></select></label>
      <div class="row gap" style="justify-content:flex-end"><button class="btn" data-close>Hủy</button><button class="btn primary" id="upGo" disabled><svg><use href="#i-upload"/></svg>Thêm vào CSDL</button></div>`);
    let fs = [];
    const set = (x) => { fs = x; $("#upInfo").textContent = fs.length ? `${fs.length} tệp đã chọn` : "Chưa chọn tệp"; $("#upGo").disabled = !fs.length; };
    $("#upFiles").onchange = (e) => set([...e.target.files]); dnd($("#upDrop"), set);
    $("#upGo").onclick = async () => {
      const b = $("#upGo"); busy(b, true);
      try { const f = new FormData(); fs.forEach((x) => f.append("images", x, x.name)); f.append("kind", $("#upKind").value); const r = await api("/api/images/upload", { method: "POST", body: f }); closeModal(); toast(`Đã thêm ${r.added} ảnh vào CSDL.`, "ok"); $("#fSplit").value = "upload"; gPage = 1; RENDER.data(); }
      catch (e) { toast(e.message, "err"); busy(b, false); }
    };
  });

  // ================================================================== HUẤN LUYỆN
  const ARCH = [["M1", "U-Net", "Mốc so sánh, không attention"], ["M2", "Attention U-Net", "Attention Gate trên kết nối tắt"], ["M3", "U-Net + CBAM", "CBAM ở mọi khối tích chập"], ["M4", "Mô hình đề xuất", "AG + CBAM đáy + giám sát sâu"]];
  $("#arch").innerHTML = ARCH.map(([k, n, d], i) => `<label><input type="radio" name="arch" value="${k}" ${k === "M4" ? "checked" : ""}><b>${k} · ${n}</b><span>${d}</span></label>`).join("");
  const eta = () => {
    const n = +$("#tN").value, sz = +$("#tSize").value, bs = +$("#tBase").value, ep = +$("#tEpochs").value, a = $("input[name=arch]:checked").value;
    const sec = 6.5 * (n / 240) * Math.pow(sz / 96, 2) * Math.pow(bs / 8, 1.6) * (a === "M1" ? 0.75 : 1) + 1.2;
    $("#tEta").textContent = `Ước tính khoảng ${sec * ep < 90 ? Math.round(sec * ep) + " giây" : Math.round((sec * ep) / 60) + " phút"} trên CPU (≈ ${F(sec, 1)} giây/epoch). Sau khi xong, mô hình được đánh giá tự động trên tập kiểm tra.`;
  };
  $("#tEpochs").addEventListener("input", (e) => { $("#epv").textContent = e.target.value; eta(); });
  $("#tN").addEventListener("input", (e) => { $("#ntv").textContent = e.target.value; eta(); });
  ["#tSize", "#tBase"].forEach((s) => $(s).addEventListener("change", eta)); $("#arch").addEventListener("change", eta); eta();
  $("#tStart").addEventListener("click", async () => {
    const b = $("#tStart"); busy(b, true);
    try {
      const r = await api("/api/train", { json: { model_type: $("input[name=arch]:checked").value, epochs: +$("#tEpochs").value, n_train: +$("#tN").value, lr: +$("#tLr").value, bs: +$("#tBs").value, size: +$("#tSize").value, base: +$("#tBase").value, augment: $("#tAug").checked, name: $("#tName").value } });
      toast(`Đã bắt đầu lần huấn luyện #${r.run_id}.`, "ok"); S.watch = r.run_id; pollTrain(true);
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });
  $("#tStop").addEventListener("click", async () => { await api("/api/train/stop", { method: "POST" }); toast("Đang dừng… trọng số tốt nhất sẽ được giữ lại.", ""); });
  const STATUS = { queued: ["Đang chờ", "run"], running: ["Đang chạy", "run"], done: ["Hoàn tất", "ok"], stopped: ["Đã dừng", "warn"], failed: ["Lỗi", "bad"] };
  let lastRunsSig = "";
  async function pollTrain(force) {
    let j; try { j = await api("/api/runs"); } catch { return; }
    $("#navTrain").hidden = !j.busy;
    const live = j.runs.find((r) => r.status === "running" || r.status === "queued") || j.runs.find((r) => r.id === S.watch) || j.runs[0];
    if (page === "train") {
      renderLive(live, j.busy);
      const sig = JSON.stringify(j.runs.map((r) => [r.id, r.status, r.epoch, r.test_dice]));
      if (force || sig !== lastRunsSig) { lastRunsSig = sig; renderRuns(j.runs); }
    }
    const wasBusy = S.trainBusy; S.trainBusy = j.busy;
    if (wasBusy && !j.busy) { await loadModels(); const d = j.runs[0]; if (d) toast(d.status === "done" ? `Lần huấn luyện #${d.id} hoàn tất – Dice kiểm tra ${F(d.test_dice, 3)}. Mô hình R${d.id} đã sẵn sàng.` : d.message, d.status === "done" ? "ok" : "err", "Huấn luyện"); }
    clearTimeout(S.trainPoll);
    if (j.busy || page === "train") S.trainPoll = setTimeout(pollTrain, j.busy ? 1500 : 6000);
  }
  function renderLive(r, isBusy) {
    $("#tStop").disabled = !isBusy; $("#tStart").disabled = isBusy;
    $("#live").classList.toggle("running", !!isBusy);
    if (!r) { $("#tKpi").innerHTML = ""; $("#tLoss").innerHTML = $("#tDice").innerHTML = '<div class="empty sm"><p>Chưa có lần huấn luyện nào.</p></div>'; return; }
    const st = STATUS[r.status] || [r.status, ""]; $("#tStatus").className = "chip " + st[1]; $("#tStatus").textContent = `#${r.id} · ${st[0]}`;
    const pct = Math.round((r.progress || 0) * 100); $("#tBar").style.width = pct + "%"; $("#tPct").textContent = pct + "%";
    const h = r.history || [], el = r.duration_s || 0, per = h.length ? h[h.length - 1].t / h.length : null, left = per && r.status === "running" ? per * (r.epochs - h.length) : null;
    $("#tMsg").textContent = `${r.name} · ${r.message || ""}`;
    $("#tKpi").innerHTML = kpi("Epoch", `${r.epoch || 0}/${r.epochs}`) + kpi("Dice kiểm định tốt nhất", F(r.best_val_dice, 3), r.best_epoch ? "epoch " + r.best_epoch : "") + kpi("Dice tập kiểm tra", F(r.test_dice, 3), r.status === "done" ? "đánh giá tự động" : "sau khi xong") + kpi("Thời gian", el < 90 ? Math.round(el) + " giây" : F(el / 60, 1) + " phút", left ? `còn khoảng ${Math.round(left)} giây` : "");
    if (h.length) {
      const x = h.map((e) => e.epoch);
      Charts.line($("#tLoss"), { x, series: [{ name: "Loss", values: h.map((e) => e.loss), color: Charts.color(1) }], yMin: 0, yMax: Math.ceil(Math.max(...h.map((e) => e.loss)) * 10 + 1) / 10, height: 200, dec: 4, xTitle: "Epoch", legend: false, tickDec: 1 });
      Charts.line($("#tDice"), { x, series: [{ name: "Dice kiểm định", values: h.map((e) => e.val_dice), color: Charts.color(0) }], yMin: 0, yMax: 1, height: 200, dec: 3, xTitle: "Epoch", legend: false, tickDec: 1 });
    } else { $("#tLoss").innerHTML = $("#tDice").innerHTML = `<div class="empty sm"><p>${r.status === "running" || r.status === "queued" ? "Đang chạy epoch đầu tiên…" : "Không có dữ liệu."}</p></div>`; }
  }
  function renderRuns(runs) {
    $("#runTbl").innerHTML = `<thead><tr><th>#</th><th>Tên</th><th>Kiến trúc</th><th>Trạng thái</th><th class="n">Epoch</th><th class="n">Ảnh</th><th class="n">Dice KĐ</th><th class="n">Dice KT</th><th class="n">Thời gian</th><th></th></tr></thead><tbody>` +
      (runs.length ? runs.map((r) => { const st = STATUS[r.status] || [r.status, ""]; const usable = ["done", "stopped"].includes(r.status);
        return `<tr><td class="muted">${r.id}</td><td><b>${esc(r.name)}</b><div class="muted sm">${r.size}px · ${r.base} kênh · lr ${r.lr} · batch ${r.bs}${r.augment ? " · tăng cường" : ""}</div></td><td>${r.model_type}</td><td><span class="chip ${st[1]}">${st[0]}</span></td><td class="n">${r.epoch}/${r.epochs}</td><td class="n">${r.n_train}</td><td class="n">${F(r.best_val_dice, 3)}</td><td class="n"><b>${F(r.test_dice, 3)}</b></td><td class="n">${r.duration_s ? (r.duration_s < 90 ? Math.round(r.duration_s) + " s" : F(r.duration_s / 60, 1) + " ph") : "–"}</td>
        <td><div class="acts">${usable ? `<button class="btn sm" data-use="R${r.id}" title="Dùng để phân đoạn">Dùng</button><button class="btn sm" data-cmp="R${r.id}" title="So sánh với các mô hình khác">So sánh</button>` : ""}${r.status === "running" || r.status === "queued" ? "" : `<button class="icon-btn sm" data-rdel="${r.id}" title="Xóa"><svg><use href="#i-trash"/></svg></button>`}</div></td></tr>`; }).join("")
        : `<tr><td colspan="10"><div class="empty"><svg><use href="#i-cpu"/></svg><p>Chưa có lần huấn luyện nào. Chọn cấu hình bên trái rồi bấm <b>Bắt đầu huấn luyện</b> (gợi ý: M4, 8–12 epoch, khoảng 1–2 phút).</p></div></td></tr>`) + "</tbody>";
    $$("#runTbl [data-use]").forEach((b) => b.addEventListener("click", async () => { await loadModels(); $("#model").value = b.dataset.use; modelDesc(); go("seg"); toast(`Đã chọn mô hình ${b.dataset.use}.`, "ok"); }));
    $$("#runTbl [data-cmp]").forEach((b) => b.addEventListener("click", async () => { await loadModels(); S.evalSel = null; go("eval"); }));
    $$("#runTbl [data-rdel]").forEach((b) => b.addEventListener("click", async () => { if (!(await confirmBox(`Xóa lần huấn luyện #${b.dataset.rdel} và trọng số của nó?`, "Xóa"))) return; try { await api(`/api/runs/${b.dataset.rdel}`, { method: "DELETE" }); toast("Đã xóa.", "ok"); await loadModels(); pollTrain(true); } catch (e) { toast(e.message, "err"); } }));
  }
  RENDER.train = () => { eta(); pollTrain(true); };

  // ================================================================== ĐÁNH GIÁ
  RENDER.eval = async (force) => {
    if (!S.models.length || !force) await loadModels();
    if (!S.evalSel) S.evalSel = new Set(S.models.map((m) => m.id));
    $("#eModels").innerHTML = S.models.map((m) => `<span class="mchip ${S.evalSel.has(m.id) ? "on" : ""}" data-id="${m.id}" style="--c:${mcol(m.id)}"><i></i>${m.id}<small>${esc(m.name)}</small></span>`).join("");
    $$("#eModels .mchip").forEach((c) => c.addEventListener("click", () => { const id = c.dataset.id; S.evalSel.has(id) ? S.evalSel.delete(id) : S.evalSel.add(id); if (!S.evalSel.size) S.evalSel.add(id); c.classList.toggle("on", S.evalSel.has(id)); drawEval(); }));
    await ensureEvals(); drawEval();
  };
  async function ensureEvals() {
    S.evals = await api("/api/evals");
    const miss = []; S.models.forEach((m) => ["test", "ext"].forEach((sp) => { if (!S.evals.some((e) => e.model_id === m.id && e.split === sp)) miss.push([m.id, sp]); }));
    if (miss.length) {
      $("#eNote").textContent = "Đang đánh giá các mô hình còn thiếu…";
      for (const sp of ["test", "ext"]) { const ids = miss.filter((x) => x[1] === sp).map((x) => x[0]); if (ids.length) await api("/api/evaluate", { json: { models: ids, split: sp } }); }
      S.evals = await api("/api/evals");
    }
  }
  $("#eRun").addEventListener("click", async () => {
    const b = $("#eRun"); busy(b, true);
    try { await api("/api/evaluate", { json: { models: [...S.evalSel], split: $("#eSplit").value } }); S.evals = await api("/api/evals"); drawEval(); toast("Đã đánh giá lại.", "ok"); } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });
  $("#eSplit").addEventListener("change", drawEval);
  function drawEval() {
    if (!S.evals) return;
    const sp = $("#eSplit").value, ids = S.models.map((m) => m.id).filter((id) => S.evalSel.has(id));
    const E = (id, s = sp) => S.evals.find((e) => e.model_id === id && e.split === s);
    const rows = ids.map((id) => ({ id, e: E(id), m: S.models.find((x) => x.id === id) })).filter((x) => x.e);
    const cols = [["dice", "Dice", 3, 1], ["iou", "IoU", 3, 1], ["precision", "Precision", 3, 1], ["recall", "Recall", 3, 1], ["hd95", "HD95 (px)", 2, -1], ["fp_rate", "Báo nhầm ở ảnh bình thường", 3, -1], ["ms", "ms/ảnh", 1, -1]];
    const best = {}; cols.forEach(([k, , , dir]) => { const v = rows.map((r) => r.e[k]).filter((x) => x !== null); best[k] = v.length ? (dir > 0 ? Math.max(...v) : Math.min(...v)) : null; });
    $("#eNote").textContent = `${rows[0] ? rows[0].e.n : 0} ảnh · ô xanh là giá trị tốt nhất`;
    $("#eTbl").innerHTML = `<thead><tr><th>Mô hình</th>${cols.map(([, l]) => `<th class="n">${l}</th>`).join("")}<th class="n">Tham số</th><th class="n">Đánh giá lúc</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${mtag(r.id)} <span class="muted sm">${esc(r.m.name)}</span></td>${cols.map(([k, , d]) => `<td class="n ${r.e[k] !== null && r.e[k] === best[k] && rows.length > 1 ? "best" : ""}">${F(r.e[k], d)}${k === "dice" ? `<span class="muted sm"> ± ${F(r.e.dice_sd, 3)}</span>` : ""}</td>`).join("")}<td class="n">${r.m.params ? F(r.m.params / 1e6, 3) + " M" : "–"}</td><td class="n muted sm">${esc((r.e.created_at || "").replace("T", " ").slice(5, 16))}</td></tr>`).join("") + "</tbody>";
    const spName = sp === "test" ? "tập kiểm tra (A)" : "thiết bị khác (B)";
    const dom = ids.map((id) => ({ id, a: E(id, "test"), b: E(id, "ext") })).filter((x) => x.a || x.b);
    const allD = dom.flatMap((x) => [x.a && x.a.dice, x.b && x.b.dice]).filter((v) => v !== undefined && v !== null);
    Charts.dot($("#eDomain"), { categories: dom.map((x) => x.id), series: [{ name: "Cùng thiết bị (A)", values: dom.map((x) => (x.a ? x.a.dice : null)), color: Charts.color(0) }, { name: "Thiết bị khác (B)", values: dom.map((x) => (x.b ? x.b.dice : null)), color: Charts.color(1) }], xMin: Math.max(0, Math.floor((Math.min(...allD, 0.9) - 0.04) * 20) / 20), xMax: 1, xTitle: "Dice trung bình", ml: 60, valueLabel: false, dotLabels: true, rowH: 42 });
    const bx = rows.map((r) => ({ name: r.id, values: r.e.per_image.filter((p) => p.kind !== "normal").map((p) => p.dice), color: mcol(r.id) })), bmin = Math.min(...bx.flatMap((g) => g.values), 1);
    $("#eBoxSub").textContent = spName; $("#eKindSub").textContent = spName;
    Charts.box($("#eBox"), { groups: bx, yMin: Math.max(0, Math.floor((bmin - 0.02) * 10) / 10), yMax: 1, height: 250, tickDec: 1, yTitle: "Dice" });
    const kd = rows.map((r) => r.e.by_kind), kv = kd.flatMap((k) => [k.benign, k.malignant]).filter((v) => v !== undefined);
    Charts.dot($("#eKind"), { categories: rows.map((r) => r.id), series: [{ name: "Lành tính", values: kd.map((k) => k.benign ?? null), color: Charts.color(2) }, { name: "Ác tính", values: kd.map((k) => k.malignant ?? null), color: Charts.color(1) }], xMin: Math.max(0, Math.floor((Math.min(...kv, 0.9) - 0.04) * 20) / 20), xMax: 1, xTitle: "Dice trung bình", ml: 60, valueLabel: false, dotLabels: true, rowH: 42 });
    drawCurves(ids);
  }
  async function drawCurves(ids) {
    if (!S.R) S.R = await api("/api/results").catch(() => null);
    const runs = (await api("/api/runs")).runs, ser = [];
    ids.forEach((id) => {
      if (id.startsWith("R")) { const r = runs.find((x) => "R" + x.id === id); if (r && r.history.length) ser.push({ name: id, values: r.history.map((h) => h.val_dice), color: mcol(id) }); }
      else if (S.R && S.R.curves && S.R.curves[id]) ser.push({ name: id + " (đồ án, 3 hạt)", values: S.R.curves[id].val, color: mcol(id) });
    });
    if (!ser.length) { $("#eCurves").innerHTML = '<div class="empty sm"><p>Không có đường cong.</p></div>'; return; }
    const n = Math.max(...ser.map((s) => s.values.length));
    ser.forEach((s) => { while (s.values.length < n) s.values.push(null); });
    Charts.line($("#eCurves"), { x: [...Array(n).keys()].map((i) => i + 1), series: ser, yMin: 0, yMax: 1, height: 260, xTitle: "Epoch", yTitle: "Dice kiểm định", dec: 3, tickDec: 1, maxXLabels: 13 });
  }
  $("#eRob").addEventListener("click", async () => {
    const b = $("#eRob"); busy(b, true);
    try {
      const k = $("#eRobKind").value, r = await api("/api/evaluate/robustness", { json: { models: [...S.evalSel], kind: k } }), ids = Object.keys(r.series), all = ids.flatMap((i) => r.series[i]);
      Charts.line($("#eRobChart"), { x: r.levels, xLabels: r.levels.map((v) => F(v, 2)), series: ids.map((i) => ({ name: i, values: r.series[i], color: mcol(i) })), yMin: Math.max(0, Math.floor((Math.min(...all) - 0.05) * 10) / 10), yMax: 1, height: 250, xTitle: KIND_TXT[k][0] + " (" + KIND_TXT[k][1] + ")", yTitle: "Dice", dec: 3 });
    } catch (e) { toast(e.message, "err"); }
    busy(b, false);
  });

  // ================================================================== THỰC NGHIỆM ĐỒ ÁN
  RENDER.exp = async () => {
    if (!S.R) S.R = await api("/api/results").catch(() => null);
    const R = S.R, box = $("#expBox");
    if (!R || !R.tables) { box.innerHTML = '<div class="card"><div class="empty"><p>Chưa có results/results.json.</p></div></div>'; return; }
    const MK = ["M1", "M2", "M3", "M4"], M = Object.fromEntries(R.main.map((m) => [m.key, m])), d = R.data, C = (k) => Charts.color(MK.indexOf(k));
    if (S.real === undefined) S.real = await api("/api/real").catch(() => ({}));
    const RL = S.real && S.real.cv ? S.real : null;
    const realCard = RL ? `<div class="card real-card"><div class="card-h"><h3>Ảnh siêu âm thật – ${esc(RL.dataset.name)} (${RL.dataset.n} ảnh)</h3><span class="badge ok">Dữ liệu thật</span></div>
      <p class="muted sm" style="margin:0 0 12px">${esc(RL.dataset.ref)}. R1: mô hình chỉ học trên ảnh mô phỏng, chạy thẳng trên ảnh thật. R2: kiểm định chéo 5 phần, tinh chỉnh trên 4/5 số ảnh thật và đánh giá trên phần còn lại; ${RL.setup.seeds.length} hạt ngẫu nhiên, ${RL.setup.epochs} epoch.</p>
      <div class="grid g2" style="margin:0"><div><h4>R1 – Chưa từng thấy ảnh thật</h4><div id="xr1" class="chart"></div></div><div><h4>R2 – Sau khi tinh chỉnh trên ảnh thật</h4><div id="xr2" class="chart"></div></div></div>
      <div class="tbl-wrap" style="margin-top:12px"><table class="tbl"><thead><tr><th>So sánh (Wilcoxon, Dice từng ảnh)</th><th class="n">Chênh Dice</th><th class="n">Ảnh tốt hơn</th><th class="n">p</th></tr></thead><tbody>${RL.tests.map((t) => `<tr><td>${esc(t.a)} vs ${esc(t.b)}</td><td class="n">${t.diff >= 0 ? "+" : "−"}${F(Math.abs(t.diff), 3)}</td><td class="n">${t.wins}/${t.n}</td><td class="n ${t.p < 0.05 ? "best" : ""}">${F(t.p, 3)}</td></tr>`).join("")}</tbody></table></div>
      <p class="muted sm">M4-R: M4 với công thức huấn luyện cải tiến (giám sát sâu giảm dần, tăng cường đặc thù siêu âm, AdamW + cosine, lật ngang khi suy luận). M1-R: U-Net với cùng công thức, để tách tác dụng của công thức khỏi kiến trúc. p &lt; 0,05 được tô xanh. Có 8 phép so sánh trên cùng 42 ảnh: nếu hiệu chỉnh Bonferroni (ngưỡng 0,006) thì chưa phép nào đạt, nên cần thêm ảnh thật để khẳng định.</p></div>` : "";
    box.innerHTML = realCard + `<div class="card"><div class="card-h"><h3>Bối cảnh</h3><span class="badge warn">${esc(R.meta.data_note)}</span></div><p class="muted sm" style="margin:0 0 14px">${esc(R.meta.setup)}</p>
      <div class="kpis small" style="margin:0">${kpi("Ảnh huấn luyện", d.train.n)}${kpi("Kiểm định", d.val.n)}${kpi("Kiểm tra (A)", d.test.n)}${kpi("Thiết bị khác (B)", d.ext.n)}${kpi("Kích thước", `${R.config.size}×${R.config.size}`)}${kpi("Epoch tối đa", R.config.epochs)}</div></div>
      <div class="grid g2"><div class="card"><div class="card-h"><h3>E1 – So sánh bốn mô hình</h3></div><div id="x1" class="chart"></div><p class="muted sm">Trung bình ± độ lệch chuẩn qua 3 hạt. Dice chênh nhau không quá 0,002; kiểm định Wilcoxon không có ý nghĩa.</p></div>
      <div class="card"><div class="card-h"><h3>Kiểm định Wilcoxon (Dice từng ảnh)</h3></div><div class="tbl-wrap"><table class="tbl"><thead><tr><th>So sánh</th><th class="n">Chênh Dice</th><th class="n">Ảnh tốt hơn</th><th class="n">p</th></tr></thead><tbody>${R.wilcoxon.map((w) => `<tr><td>${w.a} vs ${w.b}</td><td class="n">${w.mean_diff >= 0 ? "+" : ""}${F(w.mean_diff, 4)}</td><td class="n">${w.wins}/${w.n}</td><td class="n">${F(w.p, 3)}</td></tr>`).join("")}</tbody></table></div><h4 style="margin-top:16px">HD95 (điểm ảnh)</h4><div id="x1b" class="chart"></div></div>
      <div class="card"><div class="card-h"><h3>E2 – Ablation</h3></div><div id="x2" class="chart"></div><p class="muted sm">A0 U-Net · A1 +AG · A2 +SE đáy · A3 +CBAM đáy · A4 AG+CBAM · A5 = M4.</p></div>
      <div class="card"><div class="card-h"><h3>E4 – Khái quát hóa sang thiết bị khác</h3></div><div id="x4" class="chart"></div></div>
      <div class="card"><div class="card-h"><h3>E3 – Nhiễu đốm</h3></div><div id="x3a" class="chart"></div></div><div class="card"><div class="card-h"><h3>E3 – Nhiễu Gauss</h3></div><div id="x3b" class="chart"></div></div>
      <div class="card"><div class="card-h"><h3>E3 – Giảm tương phản</h3></div><div id="x3c" class="chart"></div></div><div class="card"><div class="card-h"><h3>E5 – Dice theo nhóm</h3></div><div id="x5" class="chart"></div></div>
      <div class="card"><div class="card-h"><h3>Tỷ số attention trong / ngoài khối u</h3></div><div id="xa" class="chart"></div><p class="muted sm">&gt; 1: cổng tập trung vào khối u. Cổng sâu (AG1, AG2) của M4 tập trung vào khối u; cổng nông chú ý biên và nền.</p></div>
      <div class="card"><div class="card-h"><h3>E6 – Chi phí tính toán</h3></div><div class="tbl-wrap"><table class="tbl"><thead><tr><th>Mô hình</th><th class="n">Tham số</th><th class="n">GFLOPs</th><th class="n">ms/ảnh</th><th class="n">Huấn luyện</th></tr></thead><tbody>${R.efficiency.filter((e) => MK.includes(e.key)).map((e) => `<tr><td>${mtag(e.key)} ${esc(e.name)}</td><td class="n">${e.params.toLocaleString("vi-VN")}</td><td class="n">${F(e.gflops, 3)}</td><td class="n">${F(e.ms, 1)}</td><td class="n">${R.train_info[e.key] ? F(R.train_info[e.key].time[0] / 60, 1) + " ph" : "–"}</td></tr>`).join("")}</tbody></table></div></div></div>
      ${R.tables.map((t) => `<div class="card"><div class="card-h"><h3>${esc(t.title)}</h3></div><div class="tbl-wrap"><table class="tbl"><thead><tr>${t.columns.map((c, i) => `<th class="${i ? "n" : ""}">${esc(c)}</th>`).join("")}</tr></thead><tbody>${t.rows.map((r, ri) => `<tr>${r.map((c, i) => `<td class="${i ? "n" : ""} ${t.best === ri ? "best" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>`).join("")}
      <div class="card"><div class="card-h"><h3>Hình trong báo cáo</h3></div><div class="figs">${R.figures.map((f) => `<figure><img src="/results/figures/${f.file}" alt="${esc(f.title)}" loading="lazy"><figcaption>${esc(f.title)}</figcaption></figure>`).join("")}</div></div>`;
    if (RL) {
      const z = ["M1", "M2", "M3", "M4"].filter((k) => RL.zero_shot[k]);
      Charts.dot($("#xr1"), { categories: z, series: [{ name: "Dice", values: z.map((k) => RL.zero_shot[k].mean), colors: z.map((k) => (k === "M4" ? Charts.color(0) : Charts.color(7))) }], xMin: 0.3, xMax: 0.8, ml: 50, rowH: 32, legend: false, tickDec: 1, xTitle: "Dice trung bình trên 42 ảnh" });
      const c = ["M1", "M2", "M3", "M4", "M4-scratch", "M1-R", "M4-R"].filter((k) => RL.cv[k]), lo = Math.floor(Math.min(...c.map((k) => RL.cv[k].mean)) * 20) / 20 - 0.05;
      Charts.dot($("#xr2"), { categories: c, series: [{ name: "Dice", values: c.map((k) => RL.cv[k].mean), errors: c.map((k) => RL.cv[k].sd_seed), colors: c.map((k) => (k.startsWith("M4") ? Charts.color(0) : Charts.color(7))) }], xMin: Math.max(0, lo), xMax: Math.min(1, Math.ceil(Math.max(...c.map((k) => RL.cv[k].mean)) * 20) / 20 + 0.02), ml: 90, rowH: 30, legend: false, tickDec: 2, xTitle: "Dice (± SD giữa các hạt)" });
    }
    Charts.dot($("#x1"), { categories: ["Dice", "IoU", "Precision", "Recall"], series: MK.map((k) => ({ name: `${k} – ${M[k].name}`, values: [M[k].dice[0], M[k].iou[0], M[k].precision[0], M[k].recall[0]], errors: [M[k].dice[1], M[k].iou[1], M[k].precision[1], M[k].recall[1]], color: C(k) })), xMin: 0.85, xMax: 1, offset: true, rowH: 44, ml: 80, valueLabel: false, xTitle: "Giá trị (TB ± SD qua 3 hạt)" });
    Charts.bar($("#x1b"), { categories: MK, series: [{ name: "HD95", values: MK.map((k) => M[k].hd95[0]), errors: MK.map((k) => M[k].hd95[1]), colors: MK.map(C) }], yMin: 0, yMax: 2.5, dec: 2, height: 190, legend: false });
    Charts.dot($("#x2"), { categories: R.ablation.map((a) => `${a.code} · ${a.desc === "—" ? "U-Net" : a.desc}`), series: [{ name: "Dice", values: R.ablation.map((a) => a.dice), colors: R.ablation.map((a, i) => (i === 5 ? C("M4") : Charts.color(0))) }], xMin: 0.925, xMax: 0.945, tickDec: 3, ml: 190, xTitle: "Dice (một hạt ngẫu nhiên)" });
    Charts.dot($("#x4"), { categories: MK.map((k) => `${k} · ${M[k].name}`), series: [{ name: "Cùng thiết bị (A)", values: R.external.map((e) => e.dice_in[0]), errors: R.external.map((e) => e.dice_in[1]), color: Charts.color(0) }, { name: "Thiết bị khác (B)", values: R.external.map((e) => e.dice_ext[0]), errors: R.external.map((e) => e.dice_ext[1]), color: Charts.color(1) }], xMin: 0.7, xMax: 1, ml: 160, rowH: 40, offset: true, valueLabel: false, xTitle: "Dice (TB ± SD qua 3 hạt)" });
    [["speckle", "#x3a"], ["gauss", "#x3b"], ["contrast", "#x3c"]].forEach(([k, id]) => { const rb = R.robustness[k]; Charts.line($(id), { x: rb.levels, xLabels: rb.levels.map((v) => F(v, 2)), series: MK.map((m) => ({ name: m, values: rb[m].map((v) => v[0]), errors: rb[m].map((v) => v[1]), color: C(m) })), yMin: k === "contrast" ? 0 : 0.8, yMax: 1, height: 230, xTitle: KIND_TXT[k][1] }); });
    const G = R.groups.rows; Charts.dot($("#x5"), { categories: ["Lành tính", "Ác tính", "U nhỏ", "U vừa", "U lớn"], series: G.map((g) => ({ name: g.key, values: [g.benign, g.malignant, g.small, g.medium, g.large], color: C(g.key) })), xMin: 0.88, xMax: 0.97, ml: 90, offset: true, valueLabel: false, xTitle: "Dice" });
    Charts.bar($("#xa"), { categories: ["AG1 (sâu)", "AG2", "AG3", "AG4 (nông)"], series: Object.keys(R.attention).map((k, i) => ({ name: k === "A4" ? "A4 (AG + CBAM đáy)" : k, values: R.attention[k], color: k === "A4" ? Charts.color(6) : C(k) })), yMin: 0, yMax: 2.2, dec: 2, tickDec: 1, height: 240, ref: 1 });
  };

  // ================================================================== MÔ HÌNH
  RENDER.model = async () => {
    if (!S.models.length) await loadModels();
    if (!S.R) S.R = await api("/api/results").catch(() => null);
    const R = S.R || {};
    $("#modelBox").innerHTML = `<div class="card"><div class="card-h"><h3>Các mô hình trong hệ thống</h3><span class="muted sm">${S.models.length} mô hình</span></div><div class="mcards">${S.models.map((m) => `<div class="mcard" style="--c:${mcol(m.id)}"><h4>${m.id} · ${esc(m.name)}</h4><p>${esc(m.desc)}</p><dl><dt>Kiến trúc</dt><dd>${m.type}</dd><dt>Tham số</dt><dd>${m.params ? m.params.toLocaleString("vi-VN") : "–"}</dd><dt>Đầu vào</dt><dd>${m.size}×${m.size}, ${m.base} kênh</dd><dt>Dice kiểm định</dt><dd>${F(m.val_dice, 3)}</dd><dt>Dice kiểm tra A / B</dt><dd>${F(m.eval.test, 3)} / ${F(m.eval.ext, 3)}</dd><dt>Nguồn</dt><dd>${m.source === "builtin" ? "đồ án" : m.source === "real" ? "tinh chỉnh trên ảnh thật" : "huấn luyện web"}</dd></dl></div>`).join("")}</div></div>
      <div class="card"><div class="card-h"><h3>Luồng xử lý một ảnh</h3></div><div class="pipe">${["Ảnh xám", "Đệm viền giữ tỉ lệ", "Thu nhỏ, chuẩn hóa [0, 1]", "U-Net + Attention", "Sigmoid → xác suất", "Ngưỡng", "Vùng liên thông lớn nhất", "Trả về kích thước gốc", "Chỉ số & lưu CSDL"].map((s) => `<span>${s}</span>`).join(icon("arrow"))}</div></div>
      <div class="card figw"><div class="card-h"><h3>Kiến trúc mô hình đề xuất M4</h3></div><img src="/results/figures/fig_architecture.png" alt="Kiến trúc M4"><p class="muted sm">Attention Gate trên 4 kết nối tắt, CBAM ở tầng đáy, đầu ra phụ cho giám sát sâu (chỉ dùng khi huấn luyện). Số kênh trên hình theo cấu hình 256×256; trọng số đi kèm dùng ${S.models[0] ? S.models[0].size : 96}×${S.models[0] ? S.models[0].size : 96}, ${S.models[0] ? S.models[0].base : 8} kênh cơ sở.</p></div>
      <div class="grid g2"><div class="card figw"><div class="card-h"><h3>Attention Gate</h3></div><img src="/results/figures/fig_attention_gate.png" alt="Attention Gate"><p class="sm">α = σ(ψ(ReLU(W<sub>x</sub>x + W<sub>g</sub>g))), x̂ = α·x: tín hiệu g từ tầng sâu quyết định vùng nào của đặc trưng kết nối tắt x được giữ lại.</p></div>
      <div class="card figw"><div class="card-h"><h3>CBAM</h3></div><img src="/results/figures/fig_cbam.png" alt="CBAM"><p class="sm">Chú ý kênh (gộp trung bình, cực đại → MLP) rồi chú ý không gian (gộp theo kênh → tích chập 7×7).</p></div></div>
      <div class="card"><div class="card-h"><h3>Cơ sở dữ liệu SQLite (data/app.db)</h3></div><div class="tbl-wrap"><table class="tbl"><thead><tr><th>Bảng</th><th>Nội dung</th></tr></thead><tbody><tr><td><b>images</b></td><td>Ảnh và mặt nạ chuẩn: bộ, tập (train/val/test/ext/upload), loại tổn thương, miền thiết bị, kích thước, tỷ lệ khối u</td></tr><tr><td><b>runs</b></td><td>Các lần huấn luyện: cấu hình, trạng thái, tiến trình, lịch sử loss/Dice theo epoch, Dice kiểm định và kiểm tra, đường dẫn trọng số</td></tr><tr><td><b>evals</b></td><td>Kết quả đánh giá mỗi mô hình trên mỗi tập: Dice ± SD, IoU, Precision, Recall, HD95, tỷ lệ dương tính giả, Dice từng ảnh</td></tr><tr><td><b>cases</b></td><td>Lịch sử ca phân đoạn: ảnh, mô hình, ngưỡng, chỉ số hình thái, Dice/IoU (nếu có mặt nạ), thời gian, ảnh thu nhỏ</td></tr></tbody></table></div></div>`;
  };


  // ================================================================== DẢI CHỌN ẢNH (dùng chung)
  S.srcSplit = { an: "test", ab: "test" };
  async function strip(box, onPick) {
    const key = box.id === "anStrip" ? "an" : "ab", sp = S.srcSplit[key];
    box.innerHTML = Array(10).fill('<button class="skeleton" disabled></button>').join("");
    try {
      const j = await api(`/api/images?split=${sp}&per=14&page=${1 + Math.floor(Math.random() * (sp === "real" ? 3 : 5))}`);
      if (!j.rows.length && sp === "real") { box.innerHTML = '<p class="muted sm">Chưa có ảnh thật trong CSDL. Chạy <code>python manage.py import-stu</code> (cần mạng) rồi tải lại trang.</p>'; return []; }
      const rows = j.rows.filter((r) => r.kind !== "normal").slice(0, 10);
      rows.forEach((r) => { if (r.kind === "unknown") r.kind = "real"; });
      box.innerHTML = rows.map((r) => `<button data-id="${r.id}" title="${esc(KIND[r.kind] || r.kind)} · ${esc(r.filename.split("/").pop())}"><img src="${r.url}" alt="" loading="lazy"><i class="kd-${r.kind}"></i></button>`).join("");
      $$("button", box).forEach((b) => b.addEventListener("click", () => { $$("button", box).forEach((x) => x.classList.toggle("on", x === b)); onPick({ id: +b.dataset.id }); }));
      return rows;
    } catch (e) { box.innerHTML = `<p class="muted sm">${esc(e.message)}</p>`; return []; }
  }
  $$(".src-sw").forEach((sw) => $$("button", sw).forEach((b) => b.addEventListener("click", () => {
    $$("button", sw).forEach((x) => x.classList.toggle("on", x === b)); const k = sw.dataset.for; S.srcSplit[k] = b.dataset.split;
    const pick = k === "an" ? (inp) => { S.an.input = inp; anRun(); } : abPick;
    if (k === "an") { const want = b.dataset.split === "real" && S.models.some((m) => m.id === "M4-STU") ? "M4-STU" : "M4"; $("#anModel").value = want; anFlags(); }
    strip($(k === "an" ? "#anStrip" : "#abStrip"), pick).then((rows) => { if (rows.length) $(k === "an" ? "#anStrip button" : "#abStrip button").click(); });
  })));
  const inputForm = (inp) => { const f = new FormData(); if (inp.id) f.append("image_id", inp.id); else f.append("image", inp.file, inp.file.name); return f; };
  const FLAGS = { M1: [0, 0, 0], M2: [1, 0, 0], M3: [0, 2, 0], M4: [1, 1, 1], A3: [0, 1, 0], A4: [1, 1, 0], D1: [0, 0, 1], D2: [1, 0, 1], D3: [0, 1, 1] };
  const flagsOf = (id) => { const m = S.models.find((x) => x.id === id); return FLAGS[m ? m.type : id] || [0, 0, 0]; };
  const tile = (src, t, sub = "", extra = "") => `<figure class="tile"><img src="${src}" alt="${esc(t)}"><figcaption><b>${t}</b>${sub ? `<span>${sub}</span>` : ""}${extra}</figcaption></figure>`;
  document.addEventListener("click", (e) => { const im = e.target.closest(".tile img"); if (im) zoom(im.src, im.alt); });

  // ================================================================== BÊN TRONG M4
  S.an = { input: null, res: null, view: "overlay" };
  RENDER.anatomy = async (re) => {
    if (!S.models.length) await loadModels();
    const sel = $("#anModel"), cur = sel.value || "M4";
    sel.innerHTML = S.models.map((m) => `<option value="${m.id}">${m.id} – ${esc(m.name)}</option>`).join("");
    sel.value = S.models.some((m) => m.id === cur) ? cur : "M4";
    anFlags();
    if (re && S.an.res) return anRender();
    if (!$("#anStrip").children.length || !re) {
      const rows = await strip($("#anStrip"), (inp) => { S.an.input = inp; anRun(); });
      if (!S.an.input && rows.length) { const b = $("#anStrip button:nth-child(2)") || $("#anStrip button"); b.click(); }
    }
  };
  function anFlags() {
    const f = flagsOf($("#anModel").value);
    const chip = (on, t) => `<span class="fl ${on ? "on" : ""}">${icon(on ? "check" : "x")}${t}</span>`;
    $("#anFlags").innerHTML = chip(f[0], "Attention Gate") + chip(f[1], f[1] === 2 ? "CBAM (mọi khối)" : "CBAM tầng đáy") + chip(f[2], "Giám sát sâu");
  }
  $("#anModel").addEventListener("change", () => { anFlags(); if (S.an.input) anRun(); });
  $("#anRun").addEventListener("click", () => (S.an.input ? anRun() : toast("Hãy chọn một ảnh.", "err")));
  $("#anMore").addEventListener("click", () => strip($("#anStrip"), (inp) => { S.an.input = inp; anRun(); }));
  $("#anFile").addEventListener("change", (e) => { const f = e.target.files[0]; if (!f) return; $$("#anStrip button").forEach((x) => x.classList.remove("on")); S.an.input = { file: f }; anRun(); e.target.value = ""; });
  $$("#anPipe li").forEach((li) => li.addEventListener("click", () => { const t = $("#an-" + li.dataset.sec); if (t) t.scrollIntoView({ behavior: "smooth", block: "start" }); }));

  async function anRun() {
    const b = $("#anRun"); busy(b, true);
    $("#anBody").classList.add("dim");
    try {
      const f = inputForm(S.an.input); f.append("model", $("#anModel").value); f.append("save", "0");
      S.an.res = await api("/api/anatomy", { method: "POST", body: f });
      anRender();
    } catch (e) { toast(e.message, "err"); } finally { busy(b, false); $("#anBody").classList.remove("dim"); }
  }

  function chanChart(box, vals) {
    const v = [...vals].sort((a, b) => b - a), W = Math.max(320, box.clientWidth || 520), H = 150, m = { l: 34, r: 10, t: 10, b: 26 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b, bw = pw / v.length, y = (t) => m.t + ph * (1 - t);
    let g = "";
    [0, 0.5, 1].forEach((t) => { g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" class="c-grid"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" class="c-tick">${F(t, 1)}</text>`; });
    v.forEach((x, i) => { g += `<rect x="${(m.l + i * bw + 0.3).toFixed(2)}" y="${y(x).toFixed(2)}" width="${Math.max(0.6, bw - 0.6).toFixed(2)}" height="${(ph * x).toFixed(2)}" fill="${x > 0.5 ? "var(--c1)" : "var(--c8)"}" opacity="${x > 0.5 ? 1 : 0.55}"/>`; });
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0.5)}" y2="${y(0.5)}" stroke="var(--c2)" stroke-dasharray="4 3" stroke-width="1.4"/>`;
    g += `<text x="${m.l + pw / 2}" y="${H - 4}" text-anchor="middle" class="c-tick">${v.length} kênh, sắp xếp theo trọng số giảm dần</text>`;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart-svg" role="img" aria-label="Trọng số chú ý kênh">${g}</svg>`;
  }

  function anRender() {
    const d = S.an.res; if (!d) return;
    const f = d.flags, refTxt = d.gt ? "mặt nạ chuẩn" : "vùng mô hình dự đoán";
    $$("#anPipe li").forEach((li) => li.classList.toggle("off", (li.dataset.sec === "cbam" && !d.cbam) || (li.dataset.sec === "gate" && !d.gates.length) || (li.dataset.sec === "ds" && d.heads.length < 2)));
    const head = (n, id, t, p, tech = "") => `<div class="an-h" id="an-${id}"><b class="n">${n}</b><div><h3>${t}</h3><p>${p}</p>${tech ? `<details class="tech"><summary>Chi tiết kỹ thuật</summary><p>${tech}</p></details>` : ""}</div></div>`;
    const absent = (t) => `<div class="absent">${icon("x")}<span>${t}</span></div>`;
    let h = "";
    const gMax = d.gates.length ? d.gates.reduce((a, b) => ((b.focus || 0) > (a.focus || 0) ? b : a)) : null, hd = d.heads;
    if (/-STU$/.test(d.model) && /^stu_/.test(d.label || "")) h += `<div class="note warn-note" style="margin:0 0 14px">${esc(d.model)} đã được tinh chỉnh trên chính 42 ảnh thật này, nên Dice ở đây cao hơn thực tế (Dice kiểm định chéo ≈ 0,86). Trang này dùng để xem cách mô hình hoạt động; muốn so sánh khách quan, chọn M1–M4 (chưa từng thấy ảnh thật).</div>`;
    h += `<div class="an-sum">
      <div><span>Kết quả</span><b>${d.metrics.dice !== undefined ? "Dice " + F(d.metrics.dice, 3) : d.metrics.detected ? "Có khối u" : "Không có khối u"}</b><small>${F(d.metrics.area_ratio, 1)}% diện tích · ${F(d.ms, 0)} ms</small></div>
      <div class="${d.cbam ? "" : "off"}"><span>CBAM giữ lại</span><b>${d.cbam ? `${d.cbam.hi}/${d.cbam.n} kênh` : "không có"}</b><small>${d.cbam ? "trọng số trên 0,5" : "mô hình không dùng CBAM tầng đáy"}</small></div>
      <div class="${gMax ? "" : "off"}"><span>Attention Gate</span><b>${gMax && gMax.focus !== undefined ? "×" + F(gMax.focus, 1) + " vào khối u" : gMax ? "4 cổng" : "không có"}</b><small>${gMax && gMax.focus !== undefined ? `mạnh nhất ở tầng ${gMax.level} (${gMax.res})` : "lọc kết nối tắt"}</small></div>
      <div class="${hd.length > 1 ? "" : "off"}"><span>Giám sát sâu</span><b>${hd.length > 1 && hd[0].dice != null ? `${F(hd[0].dice, 2)} → ${F(hd[hd.length - 1].dice, 2)}` : hd.length > 1 ? `${hd.length} đầu ra` : "không có"}</b><small>${hd.length > 1 ? "Dice từ 12×12 đến 96×96" : "chỉ một đầu ra"}</small></div>
    </div>`;
    // 1. mã hóa
    h += `<section class="card an-sec">${head(1, "enc", "Mô hình nhìn thấy gì ở từng tầng?", "Càng xuống sâu, ảnh càng nhỏ nhưng mô hình càng “hiểu” rộng hơn: tầng đầu bắt kết cấu nhiễu đốm, tầng cuối đã khoanh được vùng khối u. Màu nóng là nơi mô hình phản ứng mạnh.", `Ảnh ${esc(d.input_res)} qua 4 khối tích chập (Conv–BN–ReLU ×2), mỗi khối max-pool giảm một nửa kích thước và tăng gấp đôi số kênh. Bản đồ là trung bình |x| theo kênh, chuẩn hóa về 0–1.`)}
      <div class="tiles t4">${d.enc.map((e) => tile(e.img, `Tầng ${e.level}`, `${e.res} · ${e.ch} kênh`)).join("")}</div></section>`;
    // 2. CBAM
    h += `<section class="card an-sec">${head(2, "cbam", "CBAM chọn lọc thông tin nào?", "Ở tầng sâu nhất, CBAM làm hai việc: giữ lại những “kênh” đặc trưng có ích và làm yếu phần còn lại, rồi chỉ ra vùng không gian đáng chú ý. Kết quả là đặc trưng sau CBAM tập trung hơn vào khối u.", `Tầng đáy ${esc(d.bott.res)}, ${d.bott.ch} kênh. Chú ý kênh: σ(MLP(AvgPool) + MLP(MaxPool)); chú ý không gian: σ(Conv7×7([Avg; Max] theo kênh)). Hai trọng số được nhân lần lượt vào đặc trưng.`)}`;
    if (d.cbam) {
      h += `<div class="an-cbam"><div class="tiles t3">${tile(d.bott.before, "Trước CBAM", "đặc trưng tầng đáy")}${tile(d.cbam.spatial, "Chú ý không gian", d.cbam.spatial_res + " → phóng to")}${tile(d.bott.after, "Sau CBAM", "đặc trưng đã hiệu chỉnh")}</div>
        <div><h4>Trọng số chú ý kênh</h4><div id="anChan" class="chart"></div>
        <p class="note-s"><b>${d.cbam.hi}/${d.cbam.n}</b> kênh có trọng số trên 0,5 (đường đứt). Trung bình ${F(d.cbam.mean, 3)}, độ lệch chuẩn ${F(d.cbam.sd, 3)} – CBAM giữ một nhóm kênh và làm yếu phần còn lại, thay vì coi mọi kênh như nhau.</p></div></div>`;
    } else h += absent(f.cbam ? "M3 đặt CBAM trong mọi khối tích chập; trang này chỉ hiển thị CBAM ở tầng đáy như trong M4." : `${esc(d.model)} không có CBAM ở tầng đáy – đặc trưng đi thẳng sang bộ giải mã.`);
    h += `</section>`;
    // 3. Attention Gate
    h += `<section class="card an-sec">${head(3, "gate", "Attention Gate cho thông tin nào đi qua?", `Mỗi cổng quyết định phần nào của ảnh được chuyển thẳng sang bộ giải mã: đỏ là cho qua, xanh là chặn. Con số “×” cho biết cổng ưu tiên vùng khối u gấp bao nhiêu lần so với nền (so với ${refTxt}).`, "α = σ(ψ(ReLU(W<sub>x</sub>x + W<sub>g</sub>g))), x̂ = α·x. Tín hiệu g lấy từ tầng sâu hơn được nội suy lên kích thước của x; ψ là tích chập 1×1 + BN.")}`;
    if (d.gates.length) {
      h += `<div class="tiles t4">${d.gates.map((g) => tile(g.img, `AG tầng ${g.level}`, `${g.res}`, g.focus !== undefined ? `<span class="foc"><em>trong ${F(g.inside, 2)}</em><em>nền ${F(g.outside, 2)}</em><strong class="${g.focus >= 1 ? "up" : "dn"}">×${F(g.focus, 2)}</strong></span>` : "")).join("")}</div>
        ${d.gates[0].focus !== undefined ? `<div class="an-row"><div id="anGate" class="chart"></div><p class="note-s">“×” là tỉ lệ α trung bình trong khối u so với nền. Lớn hơn 1: cổng ưu tiên vùng khối u. ${(() => { const up = d.gates.filter((g) => g.focus >= 1.1).map((g) => "tầng " + g.level), dn = d.gates.filter((g) => g.focus < 0.9).map((g) => "tầng " + g.level); return (up.length ? `Ở ảnh này cổng ưu tiên vùng khối u tại ${up.join(", ")} (độ phân giải thấp, nơi mô hình định vị khối u).` : "Ở ảnh này không cổng nào ưu tiên rõ vùng khối u.") + (dn.length ? ` Tại ${dn.join(", ")} tỉ lệ nhỏ hơn 1: cổng chặn bớt chi tiết bên trong khối u – phần đã được tầng sâu định vị – và để lại thông tin độ phân giải cao ở viền và nền.` : ""); })()}</p></div>` : ""}`;
    } else h += absent(`${esc(d.model)} không có Attention Gate – đặc trưng kết nối tắt được nối nguyên vẹn sang bộ giải mã, kể cả nhiễu nền.`);
    h += `</section>`;
    // 4. giám sát sâu
    h += `<section class="card an-sec">${head(4, "ds", "Kết quả được tinh chỉnh ra sao?", "Bộ giải mã phóng to dần ảnh. M4 đặt thêm một “đầu ra phụ” ở mỗi tầng để giám sát quá trình học; nhìn vào đó thấy mô hình định vị khối u từ rất sớm (12×12) rồi chỉnh biên ở các tầng sau.", "Mỗi tầng giải mã có tích chập 1×1 làm đầu ra phụ. Hàm mất mát khi huấn luyện: L = L<sub>chính</sub> + 0,5·L<sub>1</sub> + 0,25·L<sub>2</sub> + 0,125·L<sub>3</sub> (mỗi L = 0,5·BCE + 0,5·Dice). Khi suy luận chỉ dùng đầu ra chính.")}`;
    if (d.heads.length > 1) {
      h += `<div class="seg-ctl sm" id="anView"><button data-v="overlay" class="${S.an.view === "overlay" ? "on" : ""}">Đường viền</button><button data-v="prob" class="${S.an.view === "prob" ? "on" : ""}">Xác suất</button></div>
        <div class="tiles t4">${d.heads.map((x) => tile(S.an.view === "prob" ? x.prob : x.overlay, x.name + (x.main ? "" : ""), `${x.res}${x.weight ? " · trọng số loss " + F(x.weight, x.weight === 1 ? 0 : 3) : ""}`, x.dice !== null ? `<span class="dv ${x.main ? "main" : ""}">Dice ${F(x.dice, 3)}</span>` : `<span class="dv">${F(x.area, 1)}% diện tích</span>`)).join("")}</div>
        ${d.gt ? `<div class="an-row"><div id="anHeads" class="chart"></div><p class="note-s">${(() => { const h = d.heads, mx = Math.max(...h.map((x) => x.dice)), best = h.find((x) => x.dice === mx); if (mx < 0.5) return "Cả bốn đầu ra đều chưa khớp mặt nạ chuẩn – mô hình chưa nhận ra khối u trên ảnh này (thường gặp khi mô hình học trên ảnh mô phỏng gặp ảnh thật; thử mô hình M4-STU)."; return (h[0].dice >= 0.7 ? `Ngay từ đầu ra 12×12, mô hình đã định vị được khối u (Dice ${F(h[0].dice, 3)}). ` : `Đầu ra 12×12 mới khoanh vùng thô (Dice ${F(h[0].dice, 3)}). `) + (best.main ? "Các tầng sau tinh chỉnh dần biên và đầu ra chính tốt nhất." : `Ở ảnh này ${best.name.toLowerCase()} (${best.res}) có Dice cao nhất – đầu ra phụ không nhất thiết kém hơn; vai trò chính của chúng là đưa gradient vào các tầng sâu khi huấn luyện.`); })()} Viền xanh lá: mặt nạ chuẩn; viền vàng: dự đoán.</p></div>` : ""}`;
    } else h += absent(`${esc(d.model)} không dùng giám sát sâu – chỉ có một đầu ra ở độ phân giải đầy đủ.`) + `<div class="tiles t4">${tile(d.heads[0].overlay, "Đầu ra chính", d.heads[0].res)}</div>`;
    h += `</section>`;
    // 5. kết quả
    const m = d.metrics, main = d.heads[d.heads.length - 1];
    h += `<section class="card an-sec">${head(5, "out", "Kết quả cuối cùng", `Toàn bộ quá trình trên mất ${F(d.ms, 0)} ms trên CPU.`, "Xác suất ≥ 0,5 → mặt nạ; giữ vùng liên thông lớn nhất; bỏ phần đệm viền và trả về kích thước gốc.")}
      <div class="an-out"><div class="tiles t2">${tile(main.overlay, "Phân đoạn", d.gt ? "xanh lá: chuẩn · vàng: dự đoán" : "vàng: dự đoán")}${tile(main.prob, "Bản đồ xác suất", "0 → 1")}</div>
      <div><div class="kpis small">${m.dice !== undefined ? kpi("Dice", F(m.dice, 3)) + kpi("IoU", F(m.iou, 3)) + kpi("HD95", F(m.hd95_px, 1) + " px") : ""}${kpi("Diện tích", F(m.area_ratio, 1) + "%")}${kpi("Độ tròn", F(m.circularity, 2))}${kpi("Độ đặc", F(m.solidity, 2))}</div>
      <div class="row gap" style="margin-top:14px"><button class="btn" id="anToAbl">${icon("toggle")}Thử ablation trên ảnh này</button><button class="btn" id="anCmp">${icon("layers")}Xem cùng ảnh với ${d.model === "M4" ? "M1 (U-Net)" : "M4"}</button></div></div></div></section>`;
    $("#anBody").innerHTML = h;
    if (d.cbam) chanChart($("#anChan"), d.cbam.channel);
    if (d.gates.length && d.gates[0].focus !== undefined) Charts.dot($("#anGate"), { categories: d.gates.map((g) => `AG tầng ${g.level} · ${g.res}`), series: [{ name: "Trong khối u", values: d.gates.map((g) => g.inside), color: Charts.color(0) }, { name: "Nền", values: d.gates.map((g) => g.outside), color: Charts.color(7) }], xMin: 0, xMax: 1, ml: 130, valueLabel: false, dotLabels: true, dec: 2, rowH: 40, xTitle: "α trung bình" });
    if (d.gt && d.heads.length > 1) { const ds = d.heads.map((x) => x.dice), lo = Math.max(0, Math.floor((Math.min(...ds) - 0.02) * 50) / 50), hi = Math.min(1, Math.ceil((Math.max(...ds) + 0.01) * 50) / 50); Charts.dot($("#anHeads"), { categories: d.heads.map((x) => `${x.name} · ${x.res}`), series: [{ name: "Dice", values: ds, colors: d.heads.map((x) => (x.main ? Charts.color(0) : Charts.color(7))) }], xMin: lo, xMax: hi, ml: 150, rowH: 34, xTitle: "Dice so với mặt nạ chuẩn", legend: false }); }
    $$("#anView button").forEach((b) => b.addEventListener("click", () => { S.an.view = b.dataset.v; anRender(); }));
    if (S.an.io) S.an.io.disconnect();
    S.an.io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { const id = e.target.querySelector(".an-h").id.slice(3); $$("#anPipe li").forEach((li) => li.classList.toggle("cur", li.dataset.sec === id)); } }), { rootMargin: "-45% 0px -50% 0px" });
    $$("#anBody .an-sec").forEach((x) => S.an.io.observe(x));
    $("#anToAbl").onclick = () => { S.ab.input = S.an.input; S.ab.img = null; go("ablation"); };
    $("#anCmp").onclick = () => { $("#anModel").value = d.model === "M4" ? "M1" : "M4"; anFlags(); anRun(); };
  }

  // ================================================================== ABLATION
  S.ab = { rows: null, f: { ag: 1, cbam: 1, ds: 1 }, input: null, img: null };
  const abKey = (f) => `${f.ag}${f.cbam}${f.ds}`;
  const abRow = (f = S.ab.f) => (S.ab.rows || []).find((r) => `${r.ag}${r.cbam}${r.ds}` === abKey(f));
  const ABL_LBL = (r) => [r.ag && "AG", r.cbam && "CBAM", r.ds && "GSS"].filter(Boolean).join(" + ") || "U-Net gốc";
  RENDER.ablation = async (re) => {
    if (!S.ab.rows) {
      $("#abSel").innerHTML = `<div class="loadline"><i></i>Đang đánh giá các tổ hợp trên tập kiểm tra (chỉ lần đầu)…</div>`;
      try { S.ab.rows = await api("/api/ablation"); } catch (e) { $("#abSel").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    }
    const ref = $("#abRef"), cur = ref.value || "M1";
    ref.innerHTML = S.ab.rows.map((r) => `<option value="${r.id}">${r.id} · ${ABL_LBL(r)}</option>`).join(""); ref.value = cur;
    abRender();
    if (re && S.ab.img) return abImgRender();
    if (S.ab.input && !S.ab.img) { abImg(); if (!$("#abStrip").children.length) strip($("#abStrip"), abPick); return; }
    if (!$("#abStrip").children.length) { const rows = await strip($("#abStrip"), abPick); if (!S.ab.input && rows.length) $("#abStrip button").click(); }
  };
  function abPick(inp) { S.ab.input = inp; abImg(); }
  $("#abMore").addEventListener("click", () => strip($("#abStrip"), abPick));
  $("#abRef").addEventListener("change", () => S.ab.input && abImg());
  $$("#abSw input").forEach((i) => i.addEventListener("change", () => { S.ab.f[i.dataset.f] = i.checked ? 1 : 0; abRender(); if (S.ab.img) abImgRender(); }));
  function abSet(r) { S.ab.f = { ag: r.ag, cbam: r.cbam, ds: r.ds }; $$("#abSw input").forEach((i) => (i.checked = !!S.ab.f[i.dataset.f])); abRender(); if (S.ab.img) abImgRender(); }

  function abRender() {
    const rows = S.ab.rows; if (!rows) return;
    const sel = abRow(), base = rows.find((r) => r.id === "M1");
    const d = (a, b) => (a == null || b == null ? null : a - b);
    const sgn = (v, n = 3) => (v == null ? "–" : `<span class="${v > 0.0005 ? "up" : v < -0.0005 ? "dn" : "eq"}">${v >= 0.0005 ? "+" : v <= -0.0005 ? "−" : "±"}${F(Math.abs(v) < 0.0005 ? 0 : Math.abs(v), n)}</span>`);
    const val = (r, sp) => (r && r[sp] ? r[sp].dice : null);
    // thẻ tổ hợp đang chọn
    if (sel) {
      const contrib = ["ag", "cbam", "ds"].filter((k) => S.ab.f[k]).map((k) => { const o = abRow({ ...S.ab.f, [k]: 0 }); return [k, d(val(sel, "test"), val(o, "test")), d(val(sel, "ext"), val(o, "ext")), o]; });
      const NM = { ag: "Attention Gate", cbam: "CBAM tầng đáy", ds: "Giám sát sâu" };
      $("#abSel").innerHTML = `<div class="card-h"><h3>${sel.id === "M4" ? "Mô hình đề xuất M4" : `Tổ hợp ${esc(sel.id)}`}</h3><span class="mtag">${esc(ABL_LBL(sel))}</span></div>
        <div class="kpis small">${kpi("Dice (A)", F(val(sel, "test"), 3), "so với U-Net " + sgn(d(val(sel, "test"), val(base, "test"))))}${kpi("Dice (B)", F(val(sel, "ext"), 3), "so với U-Net " + sgn(d(val(sel, "ext"), val(base, "ext"))))}${kpi("Tham số", (sel.params / 1000).toFixed(1).replace(".", ",") + "k", "U-Net " + (base.params / 1000).toFixed(1).replace(".", ",") + "k")}${kpi("HD95 (A)", F(sel.test && sel.test.hd95, 2) + " px")}</div>
        ${contrib.length ? `<h4 style="margin-top:14px">Dice thay đổi bao nhiêu nếu bỏ riêng từng thành phần</h4><div class="contrib">${contrib.map(([k, a, b, o]) => `<div><b>${NM[k]}</b><span>→ ${esc(o ? o.id : "?")}</span><em>Dice A ${sgn(a == null ? null : -a)}</em><em>Dice B ${sgn(b == null ? null : -b)}</em></div>`).join("")}</div>` : `<p class="muted sm" style="margin-top:12px">Đây là U-Net gốc – mốc để so sánh. Bật một thành phần để xem nó thay đổi gì.</p>`}`;
    }
    // kết luận nhanh: tác dụng trung bình khi thêm mỗi thành phần (trung bình trên 4 cặp tổ hợp chỉ khác thành phần đó)
    const eff = (k, sp) => { const ds = []; rows.forEach((r) => { if (r[k]) return; const o = abRow({ ag: r.ag, cbam: r.cbam, ds: r.ds, [k]: 1 }); if (o && val(o, sp) != null && val(r, sp) != null) ds.push(val(o, sp) - val(r, sp)); }); return ds.length ? [ds.reduce((a, b) => a + b, 0) / ds.length, ds.filter((x) => x > 0).length, ds.length] : [null, 0, 0]; };
    const verdict = (v, tol, w, n) => (v == null ? ["", "–"] : Math.abs(v) < tol ? ["eq", "Không đáng kể"] : (v > 0 ? w : n - w) < Math.ceil(n * 0.75) ? ["mix", "Không nhất quán"] : v > 0 ? ["up", "Có ích"] : ["dn", "Làm giảm"]);
    const NM2 = { ag: ["Attention Gate", "Lọc nhiễu ở kết nối tắt"], cbam: ["CBAM tầng đáy", "Chọn kênh và vùng ở tầng sâu"], ds: ["Giám sát sâu", "Hỗ trợ học ở các tầng giải mã"] };
    $("#abSum").innerHTML = `<div class="card-h"><h3>Kết luận nhanh: mỗi thành phần đóng góp gì?</h3><span class="muted sm">Trung bình trên 4 cặp tổ hợp chỉ khác đúng thành phần đó</span></div>
      <div class="verd">${["ag", "cbam", "ds"].map((k) => { const [a, wa, na] = eff(k, "test"), [b, wb, nb] = eff(k, "ext"), va = verdict(a, 0.005, wa, na), vb = verdict(b, 0.02, wb, nb), num = (v, vd) => `<strong class="${vd[0] === "up" ? "up" : vd[0] === "dn" ? "dn" : "eq"}">${v == null ? "–" : (v >= 0.0005 ? "+" : v <= -0.0005 ? "−" : "±") + F(Math.abs(v) < 0.0005 ? 0 : Math.abs(v), 3)}</strong>`; return `<div><b>${NM2[k][0]}</b><small>${NM2[k][1]}</small>
        <div class="vr"><span>Cùng thiết bị</span><em class="vtag ${va[0]}">${va[1]}</em>${num(a, va)}</div><div class="vr"><span>Thiết bị khác</span><em class="vtag ${vb[0]}">${vb[1]}</em>${num(b, vb)}</div><small class="faint">tăng Dice ở ${wa}/${na} cặp (A), ${wb}/${nb} cặp (B)</small></div>`; }).join("")}</div>
      <p class="note-s">Dưới ±0,005 Dice trên tập A (±0,02 trên tập B) được coi là không đáng kể – nhỏ hơn mức dao động giữa hai lần huấn luyện cùng cấu hình. “Không nhất quán”: trung bình vượt ngưỡng nhưng chưa tới 3/4 số cặp cùng chiều – thường do một tổ hợp lệch mạnh.</p>`;
    // bảng
    const bestA = Math.max(...rows.map((r) => val(r, "test") || 0)), bestB = Math.max(...rows.map((r) => val(r, "ext") || 0));
    const ck = (v) => (v ? `<span class="ok-ic">${icon("check")}</span>` : `<span class="faint">–</span>`);
    $("#abTbl").innerHTML = `<thead><tr><th>AG</th><th>CBAM</th><th>GSS</th><th>Mã</th><th class="n">Dice A</th><th class="n">Dice B</th><th class="n">Δ A</th><th class="n">Δ B</th><th class="n">Báo cáo E3</th></tr></thead><tbody>${rows.map((r) => `<tr data-id="${r.id}" class="${sel && r.id === sel.id ? "picked" : ""} ${r.id === "M4" ? "m4" : ""}"><td>${ck(r.ag)}</td><td>${ck(r.cbam)}</td><td>${ck(r.ds)}</td><td><b>${r.id}</b></td><td class="n ${val(r, "test") === bestA ? "best" : ""}">${F(val(r, "test"), 3)}</td><td class="n ${val(r, "ext") === bestB ? "best" : ""}">${F(val(r, "ext"), 3)}</td><td class="n">${r.id === "M1" ? "mốc" : sgn(d(val(r, "test"), val(base, "test")))}</td><td class="n">${r.id === "M1" ? "mốc" : sgn(d(val(r, "ext"), val(base, "ext")))}</td><td class="n">${r.report ? `${r.report.code} · ${F(r.report.dice, 3)}` : `<span class="faint">bổ sung</span>`}</td></tr>`).join("")}</tbody>`;
    $$("#abTbl tbody tr").forEach((tr) => tr.addEventListener("click", () => abSet(rows.find((r) => r.id === tr.dataset.id))));
    $("#abNote").innerHTML = `AG: Attention Gate · CBAM: CBAM tầng đáy · GSS: giám sát sâu. Mỗi tổ hợp được huấn luyện một lần với cùng cấu hình (96×96, 25 epoch, hạt 0) và đánh giá trên ${rows[0].test ? rows[0].test.n || 80 : 80} ảnh kiểm tra (A) và ảnh thiết bị khác (B) của CSDL. Cột “Báo cáo” là Dice của thí nghiệm E3 trong chương 4; ba tổ hợp chỉ có giám sát sâu kèm một thành phần là phần bổ sung trên web. Δ là chênh lệch so với U-Net gốc (M1). Trên tập A, chênh lệch vài phần nghìn nằm trong dao động giữa các lần huấn luyện (thí nghiệm E1 không thấy khác biệt có ý nghĩa thống kê); trên thiết bị khác (B), một lần huấn luyện có thể lệch nhiều hơn, nên cần nhiều hạt ngẫu nhiên trước khi kết luận.`;
    // biểu đồ: hai trục riêng cho A và B
    const cats = rows.map((r) => `${r.id} · ${ABL_LBL(r)}`), col = rows.map((r) => (sel && r.id === sel.id ? Charts.color(1) : r.id === "M4" ? Charts.color(0) : Charts.color(7)));
    const rng = (vs) => { const v = vs.filter((x) => x != null); return [Math.floor((Math.min(...v) - 0.005) * 100) / 100, Math.ceil((Math.max(...v) + 0.005) * 100) / 100]; };
    const box = $("#abChart"); box.innerHTML = `<h4>Cùng thiết bị (A)</h4><div id="abA"></div><h4 style="margin-top:10px">Thiết bị khác (B)</h4><div id="abB"></div>`;
    [["test", "#abA"], ["ext", "#abB"]].forEach(([sp, id]) => { const vs = rows.map((r) => val(r, sp)), [lo, hi] = rng(vs); Charts.dot($(id), { categories: cats, series: [{ name: "Dice", values: vs, colors: col }], xMin: lo, xMax: hi, ml: 150, rowH: 26, r: 5.5, legend: false, tickDec: 2 }); });
  }

  async function abImg() {
    $("#abImg").innerHTML = `<div class="loadline"><i></i>Đang phân đoạn bằng 8 tổ hợp…</div>`;
    try {
      const f = inputForm(S.ab.input); f.append("ref", $("#abRef").value || "M1");
      S.ab.img = await api("/api/ablation/image", { method: "POST", body: f }); abImgRender();
    } catch (e) { $("#abImg").innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }
  function abImgRender() {
    const d = S.ab.img, sel = abRow(); if (!d || !sel) return;
    const ds = d.rows.map((r) => r.metrics.dice).filter((x) => x != null), best = ds.length ? Math.max(...ds) : null;
    const df = d.diffs[sel.id], R = d.ref;
    $("#abImg").innerHTML = `<div class="ab-img"><div class="cmp-grid ab-grid">${d.rows.map((r) => `<div class="cm ${r.id === sel.id ? "picked" : ""} ${r.metrics.dice === best ? "win" : ""}" data-id="${r.id}"><img src="${r.overlay}" alt="${r.id}"><div class="cb"><b>${r.id}<small>${ABL_LBL(r)}</small></b>${r.metrics.dice != null ? `Dice <strong>${F(r.metrics.dice, 3)}</strong>` : `${F(r.metrics.area_ratio, 1)}% diện tích`}</div></div>`).join("")}</div>
      <div class="ab-diff">${df ? `<h4>${esc(sel.id)} so với ${esc(R)}</h4><img src="${df.img}" alt="Bản đồ khác biệt ${sel.id} – ${R}" class="cons"><div class="legend-s"><span><i style="background:#3cc83c"></i>${sel.id} đúng, ${R} sai: <b>${df.fixed.toLocaleString("vi-VN")}</b> px</span><span><i style="background:#e63c3c"></i>${sel.id} sai, ${R} đúng: <b>${df.broke.toLocaleString("vi-VN")}</b> px</span><span><i style="background:#fff;border:1px solid var(--border-2)"></i>Biên mặt nạ chuẩn</span></div><p class="note-s">Chỉ tô những điểm ảnh mà hai mô hình cho kết quả khác nhau. ${df.fixed > df.broke ? `${sel.id} sửa đúng nhiều hơn làm sai.` : df.fixed < df.broke ? `${sel.id} làm sai nhiều hơn sửa đúng trên ảnh này.` : "Hai mô hình gần như trùng nhau."}</p>` : `<p class="muted sm">Ảnh không có mặt nạ chuẩn nên không vẽ được bản đồ khác biệt.</p>`}</div></div>`;
    $$("#abImg .cm").forEach((c) => c.addEventListener("click", (e) => { if (e.target.tagName === "IMG") return; abSet(S.ab.rows.find((r) => r.id === c.dataset.id)); }));
  }

  // ================================================================== khởi động
  (async () => {
    await health(); setInterval(health, 15000);
    try { await loadModels(); } catch (e) { toast(e.message, "err"); }
    updateNavCases(); pollTrain();
    show(location.hash.slice(1) || "dash");
  })();
})();
