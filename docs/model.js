(function (root) {
  function decodeF32(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  function decodeU8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }

  function loadModel(json) {
    const layers = json.layers.map((l) => ({
      name: l.name,
      shape: l.shape,
      w: decodeF32(l.w),
      b: decodeF32(l.b),
    }));
    let total = 0;
    for (const l of layers) total += l.w.length;
    const all = new Float32Array(total);
    let o = 0;
    for (const l of layers) {
      for (let i = 0; i < l.w.length; i++) all[o++] = Math.abs(l.w[i]);
    }
    all.sort();
    return { layers, total, sortedAbs: all, classes: json.classes, mean: json.mean, std: json.std };
  }

  function roundHalfEven(x) {
    const r = Math.round(x);
    if (Math.abs(x % 1) === 0.5 && r % 2 !== 0) return r - 1;
    return r;
  }

  function compress(model, amount, int8) {
    const k = Math.floor(amount * model.total);
    const thresh = k >= 1 ? model.sortedAbs[k - 1] : -1;
    const layers = [];
    let nz = 0;
    const perLayer = [];
    for (const l of model.layers) {
      const w = new Float32Array(l.w.length);
      let maxabs = 0;
      let count = 0;
      for (let i = 0; i < w.length; i++) {
        const v = l.w[i];
        if (k >= 1 && !(Math.abs(v) > thresh)) continue;
        w[i] = v;
        if (v !== 0) {
          count++;
          const a = Math.abs(v);
          if (a > maxabs) maxabs = a;
        }
      }
      if (int8) {
        const scale = Math.fround(Math.max(maxabs, 1e-12) / 127);
        for (let i = 0; i < w.length; i++) {
          if (w[i] === 0) continue;
          let q = roundHalfEven(Math.fround(w[i] / scale));
          q = Math.max(-127, Math.min(127, q));
          w[i] = Math.fround(q * scale);
        }
      }
      nz += count;
      perLayer.push({ name: l.name, nonzero: count, total: w.length });
      layers.push({ shape: l.shape, w, b: l.b });
    }
    let biasBytes = 0;
    for (const l of model.layers) biasBytes += l.b.length * 4;
    const scales = int8 ? 4 * model.layers.length : 0;
    const valBytes = int8 ? 1 : 4;
    return {
      layers,
      perLayer,
      nonzero: nz,
      total: model.total,
      denseBytes: model.total * valBytes + scales + biasBytes,
      sparseBytes: nz * (valBytes + 4) + scales + biasBytes,
    };
  }

  function conv3x3(x, cin, h, w, layer) {
    const cout = layer.shape[0];
    const out = new Float32Array(cout * h * w);
    const wt = layer.w;
    for (let co = 0; co < cout; co++) {
      const base = co * h * w;
      out.fill(layer.b[co], base, base + h * w);
      for (let ci = 0; ci < cin; ci++) {
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const wv = wt[((co * cin + ci) * 3 + ky) * 3 + kx];
            if (wv === 0) continue;
            const dy = ky - 1;
            const dx = kx - 1;
            const y0 = Math.max(0, -dy);
            const y1 = Math.min(h, h - dy);
            const x0 = Math.max(0, -dx);
            const x1 = Math.min(w, w - dx);
            for (let y = y0; y < y1; y++) {
              const ib = ci * h * w + (y + dy) * w + dx;
              const ob = base + y * w;
              for (let xx = x0; xx < x1; xx++) out[ob + xx] += wv * x[ib + xx];
            }
          }
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
    return out;
  }

  function maxpool2(x, c, h, w) {
    const oh = h >> 1;
    const ow = w >> 1;
    const out = new Float32Array(c * oh * ow);
    for (let ch = 0; ch < c; ch++) {
      for (let y = 0; y < oh; y++) {
        for (let xx = 0; xx < ow; xx++) {
          const i = ch * h * w + 2 * y * w + 2 * xx;
          out[ch * oh * ow + y * ow + xx] = Math.max(x[i], x[i + 1], x[i + w], x[i + w + 1]);
        }
      }
    }
    return out;
  }

  function linear(x, layer, relu) {
    const nout = layer.shape[0];
    const nin = layer.shape[1];
    const out = new Float32Array(nout);
    for (let o = 0; o < nout; o++) {
      let s = layer.b[o];
      const base = o * nin;
      for (let i = 0; i < nin; i++) {
        const wv = layer.w[base + i];
        if (wv !== 0) s += wv * x[i];
      }
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  function forward(c, x) {
    let a = conv3x3(x, 1, 28, 28, c.layers[0]);
    a = maxpool2(a, 16, 28, 28);
    a = conv3x3(a, 16, 14, 14, c.layers[1]);
    a = maxpool2(a, 32, 14, 14);
    a = linear(a, c.layers[2], true);
    return linear(a, c.layers[3], false);
  }

  function softmax(z) {
    let m = -Infinity;
    for (const v of z) if (v > m) m = v;
    const e = Array.from(z, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function prepare(model, pixels, idx) {
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = (pixels[idx * 784 + i] / 255 - model.mean) / model.std;
    return x;
  }

  function predict(model, c, pixels, idx) {
    return softmax(forward(c, prepare(model, pixels, idx)));
  }

  function argmax(a) {
    let b = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
    return b;
  }

  const api = { decodeU8, decodeF32, loadModel, compress, forward, softmax, prepare, predict, argmax };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CNN = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
