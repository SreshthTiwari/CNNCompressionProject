(function () {
  const $ = (id) => document.getElementById(id);
  const PRESETS = [0, 50, 75, 90, 95, 99];
  const PER_CLASS_SHOWN = 3;
  const state = { sel: 0, prune: 0, int8: false };
  let model, images, pixels, comp, token = 0, timer = null, baseline = null;

  function drawImage(canvas, idx) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = pixels[idx * 784 + i];
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function fmtBytes(b) {
    if (b >= 1024 * 1024) return (b / 1024 / 1024).toFixed(2) + " MB";
    return (b / 1024).toFixed(b >= 10240 ? 0 : 1) + " KB";
  }

  function buildPicker() {
    const per = images.n / 10;
    const box = $("picker");
    for (let c = 0; c < 10; c++) {
      for (let j = 0; j < PER_CLASS_SHOWN; j++) {
        const idx = c * per + j;
        const b = document.createElement("button");
        b.type = "button";
        b.title = model.classes[c];
        b.setAttribute("aria-label", model.classes[c] + " " + (j + 1));
        b.dataset.idx = idx;
        const cv = document.createElement("canvas");
        cv.width = 28;
        cv.height = 28;
        drawImage(cv, idx);
        b.appendChild(cv);
        b.addEventListener("click", () => { state.sel = idx; renderSelection(); renderPrediction(); });
        box.appendChild(b);
      }
    }
  }

  function buildChips() {
    for (const p of PRESETS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = p + "%";
      b.addEventListener("click", () => { $("prune").value = p; onControl(); });
      $("chips").appendChild(b);
    }
  }

  function buildBars() {
    const box = $("bars");
    for (let c = 0; c < 10; c++) {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.id = "bar" + c;
      row.innerHTML = '<span class="name"></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
      row.querySelector(".name").textContent = model.classes[c];
      box.appendChild(row);
    }
  }

  function renderSelection() {
    drawImage($("big"), state.sel);
    $("truth").textContent = model.classes[images.labels[state.sel]];
    for (const b of $("picker").children) b.setAttribute("aria-pressed", String(+b.dataset.idx === state.sel));
  }

  function renderPrediction() {
    const p = CNN.predict(model, comp, pixels, state.sel);
    const top = CNN.argmax(p);
    const truth = images.labels[state.sel];
    for (let c = 0; c < 10; c++) {
      const row = $("bar" + c);
      row.classList.toggle("top", c === top);
      row.querySelector(".name").classList.toggle("true", c === truth);
      row.querySelector(".fill").style.width = (p[c] * 100).toFixed(1) + "%";
      row.querySelector(".pct").textContent = (p[c] * 100).toFixed(p[c] > 0.995 || p[c] < 0.0005 ? 0 : 1) + "%";
    }
    const v = $("verdict");
    if (top === truth) {
      v.textContent = "predicted " + model.classes[top] + ", correct";
      v.className = "verdict good";
    } else {
      v.textContent = "predicted " + model.classes[top] + ", actual " + model.classes[truth];
      v.className = "verdict bad";
    }
  }

  function renderStats() {
    const pct = (100 * comp.nonzero) / comp.total;
    $("nz").textContent = pct.toFixed(pct < 10 ? 1 : 0) + "%";
    $("nzn").textContent = comp.nonzero.toLocaleString() + " of " + comp.total.toLocaleString();
    $("dense").textContent = fmtBytes(comp.denseBytes);
    $("sparse").textContent = fmtBytes(comp.sparseBytes);
    const box = $("layers");
    box.innerHTML = "";
    for (const l of comp.perLayer) {
      const f = (100 * l.nonzero) / l.total;
      const row = document.createElement("div");
      row.className = "layer";
      row.innerHTML = '<span></span><div class="track"><div class="fill"></div></div><span class="n"></span>';
      row.children[0].textContent = l.name;
      row.querySelector(".fill").style.width = f + "%";
      row.children[2].textContent = f.toFixed(f < 10 ? 1 : 0) + "% of " + l.total.toLocaleString();
      box.appendChild(row);
    }
  }

  async function runAccuracy(my, c) {
    const n = images.n;
    let ok = 0;
    for (let i = 0; i < n; i++) {
      if (CNN.argmax(CNN.forward(c, CNN.prepare(model, pixels, i))) === images.labels[i]) ok++;
      if (i % 25 === 24) {
        await new Promise((r) => setTimeout(r, 0));
        if (my !== token) return null;
      }
    }
    return ok / n;
  }

  function scheduleAccuracy() {
    token++;
    const my = token;
    const c = comp;
    $("statcard").classList.add("busy");
    $("status").textContent = "measuring accuracy";
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const a = await runAccuracy(my, c);
      if (a === null || my !== token) return;
      if (baseline === null && state.prune === 0 && !state.int8) baseline = a;
      $("acc").textContent = (a * 100).toFixed(1) + "%";
      $("accfill").style.width = a * 100 + "%";
      $("accbase").textContent = baseline === null ? " " : "unpruned: " + (baseline * 100).toFixed(1) + "%";
      $("statcard").classList.remove("busy");
      $("status").textContent = "";
    }, 180);
  }

  function onControl() {
    state.prune = +$("prune").value;
    state.int8 = $("int8").checked;
    $("pv").textContent = state.prune + "%";
    comp = CNN.compress(model, state.prune / 100, state.int8);
    renderPrediction();
    renderStats();
    scheduleAccuracy();
  }

  async function init() {
    try {
      const [w, im] = await Promise.all([
        fetch("data/weights.json").then((r) => r.json()),
        fetch("data/images.json").then((r) => r.json()),
      ]);
      model = CNN.loadModel(w);
      images = im;
      pixels = CNN.decodeU8(im.pixels);
    } catch (e) {
      $("verdict").textContent = "could not load model data. serve this folder over http, file:// blocks the fetch.";
      $("verdict").className = "verdict bad";
      return;
    }
    buildPicker();
    buildChips();
    buildBars();
    renderSelection();
    $("prune").addEventListener("input", onControl);
    $("int8").addEventListener("change", onControl);
    onControl();
  }

  init();
})();
