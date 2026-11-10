"use strict";

// The page: it polls /api/state twice a second, posts JSON for every control, and lets the two
// <img> elements carry the live views as MJPEG streams. No framework and no network fetch.

const $ = (id) => document.getElementById(id);
const edits = {};
let options = null;
let lastStatus = "";

function stageLabel(value) {
  return ({"Press the brick down": "Pressing", "Align": "Alignment",
           "Retract": "Retraction", "Verify": "Verification"})[value] || value;
}

async function get(url) {
  const r = await fetch(url, { cache: "no-store" });
  return r.json();
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// ------------------------------------------------------------------ setup ---

function fillSelect(el, items, selected) {
  el.innerHTML = "";
  for (const item of items) {
    const name = typeof item === "string" ? item : item.name;
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = typeof item === "string" ? item : (item.label || item.name);
    if (item.connected === false) {
      opt.disabled = true;
      opt.textContent = `${name} — not connected`;
    }
    if (name === selected) opt.selected = true;
    el.appendChild(opt);
  }
}

function collectSetup() {
  return {
    instruction: $("instruction").value,
    job: $("job").value,
    seed: Number($("seed").value),
    placed: $("placed").value.split("\n").filter((s) => s.trim()),
    plate_offset: $("plate-offset").value,
    plate_dx_mm: Number($("plate-dx").value),
    plate_dy_mm: Number($("plate-dy").value),
    plate_yaw_deg: Number($("plate-yaw").value),
    supply_offset: $("supply-offset").value,
    supply_dx_mm: Number($("supply-dx").value),
    supply_dy_mm: Number($("supply-dy").value),
    supply_yaw_deg: Number($("supply-yaw").value),
    params_file: $("params-file").value,
    param_edits: edits,
    source: $("source").value,
    controller: $("controller").value,
    breakable: $("breakable").checked,
    damage_monitor: $("damage").checked,
    mode: $("mode").value,
  };
}

function applySetup(s) {
  $("instruction").value = s.instruction;
  $("placed").value = (s.placed || []).join("\n");
  $("seed").value = s.seed;
  $("plate-dx").value = s.plate_dx_mm;
  $("plate-dy").value = s.plate_dy_mm;
  $("plate-yaw").value = s.plate_yaw_deg;
  $("supply-dx").value = s.supply_dx_mm;
  $("supply-dy").value = s.supply_dy_mm;
  $("supply-yaw").value = s.supply_yaw_deg;
  $("breakable").checked = s.breakable;
  $("damage").checked = s.damage_monitor;
}

// ----------------------------------------------------------------- tables ---

function kv(el, pairs) {
  el.innerHTML = "";
  for (const [k, v] of pairs) {
    const tr = document.createElement("tr");
    const a = document.createElement("td");
    const b = document.createElement("td");
    a.textContent = k;
    if (typeof v === "boolean") {
      b.textContent = v ? "yes" : "no";
      b.className = v ? "ok" : "no";
    } else {
      b.textContent = v === null || v === undefined ? "—" : String(v);
    }
    tr.append(a, b);
    el.appendChild(tr);
  }
}

function rows(el, columns, data) {
  el.innerHTML = "";
  const head = document.createElement("tr");
  for (const c of columns) {
    const th = document.createElement("th");
    th.textContent = c;
    head.appendChild(th);
  }
  el.appendChild(head);
  for (const r of data) {
    const tr = document.createElement("tr");
    for (const c of columns) {
      const td = document.createElement("td");
      const v = c === "step" ? stageLabel(r[c]) : r[c];
      td.textContent = typeof v === "boolean" ? (v ? "yes" : "no")
        : (v === null || v === undefined ? "—" : String(v));
      if (typeof v === "boolean") td.className = v ? "ok" : "no";
      tr.appendChild(td);
    }
    el.appendChild(tr);
  }
}

function drawGrid(grid) {
  const el = $("grid");
  el.innerHTML = "";
  for (const row of grid) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.style.background = cell ? cell : "transparent";
      tr.appendChild(td);
    }
    el.appendChild(tr);
  }
}

// ----------------------------------------------------------- param editing ---

let paramRows = [];

async function loadParams(file) {
  const data = await get(`/api/params?file=${encodeURIComponent(file)}`);
  paramRows = data.rows;
  renderParams();
  $("params-msg").textContent = data.problems.join("; ");
}

function renderParams() {
  const filter = $("params-filter").value.trim().toLowerCase();
  const el = $("params");
  el.innerHTML = "";
  const head = document.createElement("tr");
  for (const c of ["Parameter", "Value", "Unit"]) {
    const th = document.createElement("th");
    th.textContent = c;
    head.appendChild(th);
  }
  el.appendChild(head);
  for (const r of paramRows) {
    if (filter && !r.path.toLowerCase().includes(filter)) continue;
    const tr = document.createElement("tr");
    const a = document.createElement("td");
    a.textContent = r.path;
    const b = document.createElement("td");
    const input = document.createElement("input");
    input.value = edits[r.path] !== undefined ? edits[r.path] : String(r.value);
    input.addEventListener("change", async () => {
      const raw = input.value.trim();
      const num = Number(raw);
      edits[r.path] = raw === "" ? null : (Number.isNaN(num) ? raw : num);
      const out = await post("/api/params", { file: $("params-file").value, edits });
      $("params-msg").textContent = out.problems.join("; ");
    });
    b.appendChild(input);
    const c = document.createElement("td");
    c.textContent = r.unit || "-";
    tr.append(a, b, c);
    el.appendChild(tr);
  }
}

// -------------------------------------------------------------------- poll ---

async function poll() {
  let s;
  try {
    s = await get("/api/state");
  } catch (e) {
    return;
  }
  $("status").textContent = s.status;
  $("status").dataset.status = s.status;
  $("message").textContent = s.message;
  $("rates").textContent =
    `view ${s.fps.view} fps | ${s.tick_rate}x real time | frames ${s.frames}`;
  $("camera").textContent =
    `distance ${s.camera.distance} m, azimuth ${s.camera.azimuth} deg, elevation ${s.camera.elevation} deg`;
  $("setup-problems").textContent = (s.problems || []).join("; ");
  $("placement").textContent = s.placement || "";
  if (s.grid && s.grid.length) drawGrid(s.grid);

  const r = s.readouts || {};
  kv($("readouts"), [
    ["Stage", stageLabel(r.step)],
    ["Tracking Error [mm]", r.err_mm],
    ["Tracking tolerance [mm]", r.tol_mm],
    ["Segmentation mIoU [%]", r.miou],
    ["Prediction error [mm]", r.predict_mm],
    ["Press force [N]", r.press_n],
    ["Insertion depth [mm]", r.depth_mm],
    ["Damage state", r.damage],
    ["Correction [mm]", r.correction_mm],
    ["Reference Objects [-]", r.refs_seen],
    ["Vision tick [-]", s.k],
    ["Control tick [-]", s.tick],
  ]);
  rows($("steps"), ["step", "ticks", "err_mm"], s.steps || []);

  const res = s.result;
  kv($("result"), res ? [
    ["Seated", res.seated],
    ["On cell", res.on_cell],
    ["Lateral [mm]", res.lateral_mm],
    ["Depth error [mm]", res.depth_error_mm],
    ["Damage state", res.damage],
    ["Failure", s.failure || "none"],
  ] : [["Failure", s.failure || "none"]]);

  const d = s.displacement || {};
  kv($("displacement"), [
    ["Baseplate [mm]", d.plate_mm],
    ["Baseplate [deg]", d.plate_deg],
    ["Supply brick [mm]", d.supply_mm],
    ["Supply brick [deg]", d.supply_deg],
  ]);

  const twin = s.twin || {};
  const hasTwin = s.mode === "twin (sim)";
  $("twin-figure").hidden = !hasTwin;
  $("twin-head").hidden = !hasTwin;
  if (hasTwin && !$("twin").src.includes("/stream/twin.mjpg")) {
    $("twin").src = "/stream/twin.mjpg";
  }
  kv($("twin-table"), hasTwin ? [
    ["Joint RMSE [deg]", twin.joint_rmse_deg],
    ["Tool pose error [mm]", twin.tool_mm],
    ["Scene-state distance [-]", twin.scene_distance],
  ] : []);

  $("btn-record").textContent = s.recording ? "Stop" : "Record";
  const files = $("files");
  files.innerHTML = "";
  for (const f of s.files || []) {
    const li = document.createElement("li");
    li.textContent = f;
    files.appendChild(li);
  }

  if (s.status !== lastStatus) {
    lastStatus = s.status;
    $("btn-run").disabled = !(s.status === "ready" || s.status === "paused");
    $("btn-pause").disabled = s.status !== "running";
    $("btn-step").disabled = !(s.status === "ready" || s.status === "paused");
  }
}

async function pollBatch() {
  let b;
  try {
    b = await get("/api/batch");
  } catch (e) {
    return;
  }
  const pct = b.total ? (100 * b.done) / b.total : 0;
  $("batch-bar").style.width = `${pct}%`;
  $("batch-msg").textContent = `${b.state} ${b.done}/${b.total} ${b.message || ""}`;
  const s = b.summary;
  kv($("batch-summary"), s ? [
    ["Seeds [-]", s.n],
    ["Seated [-]", s.seated],
    ["On cell [-]", s.on_cell],
    ["Seated rate [-]", s.rate],
    ["Wilson interval [-]", `${s.wilson_low} – ${s.wilson_high}`],
    ["Lateral median [mm]", s.lateral_median_mm],
    ["Lateral max [mm]", s.lateral_max_mm],
  ] : []);
  if (b.columns && b.columns.length) rows($("batch-rows"), b.columns, b.rows);
}

// ------------------------------------------------------------------ camera ---

let drag = null;
let pending = { d_azimuth: 0, d_elevation: 0, zoom: 1 };
let sending = false;

async function flush() {
  if (sending) return;
  const p = pending;
  if (!p.d_azimuth && !p.d_elevation && p.zoom === 1) return;
  pending = { d_azimuth: 0, d_elevation: 0, zoom: 1 };
  sending = true;
  try {
    await post("/api/camera", p);
  } finally {
    sending = false;
  }
}

function bindCamera(img) {
  // A stream that ends leaves a still frame behind, so ask for it again.
  img.addEventListener("error", () => {
    const src = img.getAttribute("src");
    if (!src) return;
    setTimeout(() => { img.src = `${src.split("?")[0]}?t=${Date.now()}`; }, 1000);
  });
  img.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY };
    img.setPointerCapture(e.pointerId);
  });
  img.addEventListener("pointermove", (e) => {
    if (!drag) return;
    pending.d_azimuth += (drag.x - e.clientX) * 0.4;
    pending.d_elevation += (e.clientY - drag.y) * 0.3;
    drag = { x: e.clientX, y: e.clientY };
  });
  img.addEventListener("pointerup", () => { drag = null; });
  img.addEventListener("wheel", (e) => {
    e.preventDefault();
    pending.zoom *= e.deltaY > 0 ? 1.08 : 1 / 1.08;
  }, { passive: false });
}

// ------------------------------------------------------------------- wiring ---

async function boot() {
  options = await get("/api/options");
  const state = await get("/api/state");
  const s = state.setup;
  fillSelect($("job"), options.jobs, s.job);
  fillSelect($("plate-offset"), options.offsets, s.plate_offset);
  fillSelect($("supply-offset"), options.offsets, s.supply_offset);
  fillSelect($("source"), options.sources, s.source);
  fillSelect($("controller"), options.controllers, s.controller);
  fillSelect($("mode"), options.modes, s.mode);
  fillSelect($("speed"), options.speeds, state.speed);
  fillSelect($("params-file"), options.param_files, s.params_file);
  applySetup(s);
  await loadParams(s.params_file);

  $("btn-compile").addEventListener("click", async () => {
    const out = await post("/api/compile", { text: $("instruction").value });
    $("placement").textContent = out.placement;
    $("compile-msg").textContent = out.understood
      ? `target cell ${out.target ? out.target.join(", ") : "-"}`
      : out.problems.join("; ");
    drawGrid(out.grid);
  });

  $("btn-apply").addEventListener("click", () => post("/api/setup", collectSetup()));
  $("btn-reset").addEventListener("click", () => post("/api/command", { command: "reset" }));
  $("btn-run").addEventListener("click", () => post("/api/command", { command: "run" }));
  $("btn-pause").addEventListener("click", () => post("/api/command", { command: "pause" }));
  $("btn-step").addEventListener("click", () =>
    post("/api/command", { command: "step", ticks: 1 }));
  $("speed").addEventListener("change", () =>
    post("/api/command", { command: "speed", speed: $("speed").value }));

  $("params-file").addEventListener("change", () => {
    for (const k of Object.keys(edits)) delete edits[k];
    loadParams($("params-file").value);
  });
  $("params-filter").addEventListener("input", renderParams);
  $("params-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const out = await post("/api/upload_params", { name: file.name, text });
    $("params-msg").textContent = out.problems.join("; ") || out.file;
    if (!out.problems.length) {
      const opt = document.createElement("option");
      opt.value = out.file;
      opt.textContent = file.name;
      opt.selected = true;
      $("params-file").appendChild(opt);
      loadParams(out.file);
    }
  });

  $("btn-nudge").addEventListener("click", () => post("/api/disturb", {
    kind: "brick", body: $("dist-body").value,
    dx_mm: Number($("dist-dx").value), dy_mm: Number($("dist-dy").value),
    dyaw_deg: Number($("dist-yaw").value),
  }));
  $("btn-remove").addEventListener("click", () =>
    post("/api/disturb", { kind: "remove", body: $("dist-body").value }));
  $("btn-plate").addEventListener("click", () => post("/api/disturb", {
    kind: "plate", dx_mm: Number($("plate-ddx").value),
    dy_mm: Number($("plate-ddy").value), dyaw_deg: Number($("plate-dyaw").value),
  }));
  $("btn-hold").addEventListener("click", () =>
    post("/api/disturb", { kind: "hold", ticks: Number($("hold-ticks").value) }));

  $("btn-record").addEventListener("click", () => post("/api/record", {
    action: $("btn-record").textContent === "Record" ? "start" : "stop",
  }));
  $("btn-still").addEventListener("click", () => post("/api/record", { action: "still" }));
  $("btn-log").addEventListener("click", () => post("/api/command", { command: "log" }));

  $("btn-batch").addEventListener("click", () =>
    post("/api/batch", { action: "start", seeds: Number($("batch-seeds").value),
                         setup: collectSetup() }));
  $("btn-batch-cancel").addEventListener("click", () => post("/api/batch", { action: "cancel" }));
  $("btn-batch-csv").addEventListener("click", async () => {
    const out = await post("/api/batch", { action: "csv" });
    if (out.path) $("batch-msg").textContent = out.path;
  });

  bindCamera($("view"));
  bindCamera($("twin"));
  setInterval(flush, 60);
  setInterval(poll, 500);
  setInterval(pollBatch, 1000);
  setInterval(refreshBodies, 3000);
  poll();
}

async function refreshBodies() {
  const o = await get("/api/options");
  const sel = $("dist-body");
  const keep = sel.value;
  if (sel.options.length !== o.bricks.length) fillSelect(sel, o.bricks, keep);
}

document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  const map = {
    " ": () => post("/api/command",
      { command: lastStatus === "running" ? "pause" : "run" }),
    s: () => post("/api/command", { command: "step", ticks: 1 }),
    r: () => post("/api/command", { command: "reset" }),
    c: () => $("btn-record").click(),
    p: () => post("/api/record", { action: "still" }),
    ArrowLeft: () => { pending.d_azimuth -= 5; },
    ArrowRight: () => { pending.d_azimuth += 5; },
    ArrowUp: () => { pending.d_elevation += 4; },
    ArrowDown: () => { pending.d_elevation -= 4; },
    "+": () => { pending.zoom /= 1.15; },
    "=": () => { pending.zoom /= 1.15; },
    "-": () => { pending.zoom *= 1.15; },
    0: () => post("/api/camera", { reset: true }),
  };
  const fn = map[e.key] || map[e.key.toLowerCase()];
  if (fn) {
    e.preventDefault();
    fn();
  }
});

boot();
