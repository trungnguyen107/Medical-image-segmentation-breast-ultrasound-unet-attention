// SonoSeg – xem ảnh siêu âm và bản đồ xác suất khối u dưới dạng bề mặt 3D (three.js, đóng gói cục bộ, chạy offline).
// Ảnh "surface" do máy chủ trả về: kênh R = ảnh siêu âm, G = xác suất khối u, B = mặt nạ dự đoán.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function loadPixels(src) {
  return new Promise((ok, bad) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
      const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0);
      ok({ w: im.width, h: im.height, d: x.getImageData(0, 0, im.width, im.height).data });
    };
    im.onerror = bad; im.src = src;
  });
}

function webgl() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; }
}

// màu: nền siêu âm xám lạnh → vùng xác suất cao chuyển dần sang xanh ngọc rồi hổ phách (giống thước đo trên máy siêu âm)
const C_LOW = new THREE.Color("#1c2b3d"), C_HIGH = new THREE.Color("#c9d6e3"), C_TEAL = new THREE.Color("#14b8a6"), C_AMBER = new THREE.Color("#f5a524");

async function mount(el, src, opt = {}) {
  if (!webgl()) { el.innerHTML = opt.fallback ? `<img src="${opt.fallback}" alt="" style="width:100%;height:100%;object-fit:contain">` : ""; return { dispose() {} }; }
  const px = await loadPixels(src);
  const W = px.w, H = px.h, aspect = W / H;
  const sx = aspect >= 1 ? 2 : 2 * aspect, sy = aspect >= 1 ? 2 / aspect : 2;
  const geo = new THREE.PlaneGeometry(sx, sy, W - 1, H - 1);
  const pos = geo.attributes.position, n = pos.count;
  const img = new Float32Array(n), pr = new Float32Array(n), mk = new Uint8Array(n);
  for (let i = 0; i < n; i++) { img[i] = px.d[i * 4] / 255; pr[i] = px.d[i * 4 + 1] / 255; mk[i] = px.d[i * 4 + 2] > 127 ? 1 : 0; }
  // làm mượt nhẹ xác suất để bề mặt không lởm chởm
  const sm = new Float32Array(n);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, c = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < W && yy < H) { s += pr[yy * W + xx]; c++; } }
    sm[y * W + x] = s / c;
  }
  const baseZ = new Float32Array(n), peakZ = opt.height || 0.5;
  const col = new Float32Array(n * 3), tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    baseZ[i] = img[i] * 0.09 + Math.pow(sm[i], 1.3) * peakZ;
    tmp.copy(C_LOW).lerp(C_HIGH, Math.pow(img[i], 0.9));
    const t = Math.min(1, sm[i] * 1.25);
    if (t > 0.05) tmp.lerp(C_TEAL, t * 0.75);
    if (mk[i]) tmp.lerp(C_AMBER, 0.35 + 0.4 * t);
    col.set([tmp.r, tmp.g, tmp.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  el.innerHTML = ""; el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(34, 1, 0.1, 50);
  cam.up.set(0, 0, 1);                 // trục độ cao là Z: xoay quanh như bàn xoay, luôn nhìn từ phía trên
  cam.position.set(1.0, -3.3, 2.0);
  scene.add(new THREE.HemisphereLight("#dbe7f5", "#0b1422", 1.15));
  const key = new THREE.DirectionalLight("#ffffff", 1.6); key.position.set(-1.6, -2.2, 3.2); scene.add(key);
  const rim = new THREE.DirectionalLight("#5eead4", 0.7); rim.position.set(2, 2.5, 1.2); scene.add(rim);

  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.62, metalness: 0.08, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat); scene.add(mesh);

  // khung đáy và lưới đo (như thước trên màn hình máy siêu âm)
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(sx, sy)), new THREE.LineBasicMaterial({ color: "#3b556f" }));
  frame.position.z = -0.002; scene.add(frame);
  const grid = new THREE.GridHelper(Math.max(sx, sy) * 1.5, 12, "#22364c", "#1a2a3d"); grid.rotation.x = Math.PI / 2; grid.position.z = -0.01; scene.add(grid);

  // đường viền khối u dự đoán, nổi trên bề mặt
  const edge = [];
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x; if (!mk[i]) continue;
    if (!mk[i - 1] || !mk[i + 1] || !mk[i - W] || !mk[i + W]) edge.push(i);
  }
  const ePos = new Float32Array(edge.length * 3);
  edge.forEach((i, k) => ePos.set([pos.getX(i), pos.getY(i), 0], k * 3));
  const eGeo = new THREE.BufferGeometry(); eGeo.setAttribute("position", new THREE.BufferAttribute(ePos, 3));
  const dots = new THREE.Points(eGeo, new THREE.PointsMaterial({ color: "#f5a524", size: 0.022, transparent: true, opacity: 0.95, depthWrite: false }));
  scene.add(dots);

  const ctl = new OrbitControls(cam, renderer.domElement);
  ctl.enableDamping = true; ctl.dampingFactor = 0.08; ctl.enablePan = false;
  ctl.minDistance = 1.8; ctl.maxDistance = 6; ctl.maxPolarAngle = Math.PI * 0.49;
  ctl.autoRotate = !reduced && opt.autoRotate !== false; ctl.autoRotateSpeed = 0.7;
  renderer.domElement.addEventListener("pointerdown", () => { ctl.autoRotate = false; });

  function resize() {
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
    cam.aspect = r.width / r.height; cam.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(el); resize();

  // một khoảnh khắc chuyển động duy nhất: bề mặt "nổi lên" khi mở
  let t0 = performance.now(), raf = 0, alive = true; const RISE = reduced ? 0 : 1400;
  function setHeight(k) {
    for (let i = 0; i < n; i++) pos.setZ(i, baseZ[i] * k);
    pos.needsUpdate = true; geo.computeVertexNormals();
    edge.forEach((i, j) => ePos[j * 3 + 2] = baseZ[i] * k + 0.012); eGeo.attributes.position.needsUpdate = true;
  }
  let rising = true;
  function loop(now) {
    if (!alive) return;
    if (rising) { const k = RISE ? Math.min(1, (now - t0) / RISE) : 1; setHeight(1 - Math.pow(1 - k, 3)); if (k >= 1) rising = false; }
    ctl.update(); renderer.render(scene, cam); raf = requestAnimationFrame(loop);
  }
  setHeight(RISE ? 0 : 1); raf = requestAnimationFrame(loop);
  // dừng vẽ khi khung không còn trên màn hình
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting && alive && !raf) raf = requestAnimationFrame(loop); else if (!e.isIntersecting) { cancelAnimationFrame(raf); raf = 0; } });
  io.observe(el);
  return {
    setHeight(h) { for (let i = 0; i < n; i++) baseZ[i] = img[i] * 0.09 + Math.pow(sm[i], 1.3) * h; setHeight(1); },
    reset() { cam.position.set(1.0, -3.3, 2.0); ctl.target.set(0, 0, 0); },
    dispose() { alive = false; cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); ctl.dispose(); renderer.dispose(); geo.dispose(); eGeo.dispose(); el.innerHTML = ""; },
  };
}

window.Scan3D = { mount };
window.dispatchEvent(new Event("scan3d-ready"));
