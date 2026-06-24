/*
 * umeh.js — the Umeh formula, client-side (educational)
 * =====================================================
 * A faithful JavaScript port of umeh_model.py. The long-term valuation models
 * (Power Law, Stock-to-Flow, Top Cap) are exact closed-form math; the
 * short-term Nwachukwu core is a KMeans-free kernel-Bayesian estimator + CFA
 * fusion. tools/parity_check.py verifies this file produces the same numbers
 * as the Python reference on a fixed snapshot.
 *
 * EDUCATIONAL USE ONLY. Not financial advice.
 *
 * Pure compute functions take injected data + a fixed `nowMs`, so they run
 * identically in the browser and under Node (for the parity test).
 */
(function (root) {
  'use strict';

  // ---- Shared constants (mirror umeh_model.py) ----
  const GENESIS_MS = Date.UTC(2009, 0, 3);
  const SECONDS_PER_DAY = 86400.0;
  const BLOCKS_PER_DAY = 144.0;
  const BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365.0;
  const HALVING_INTERVAL = 210000;
  const INITIAL_REWARD = 50.0;

  const PL_A = -17.0, PL_B = 5.8;
  const PL_SUPPORT_FACTOR = 0.42, PL_RESISTANCE_FACTOR = 2.10;
  const S2F_K = 0.40, S2F_B = 3.30;
  const TOP_CAP_MULT = 35.0;
  const UMEH_WEIGHTS = { market: 0.45, power_law: 0.38, top_cap: 0.12, stock_to_flow: 0.05 };
  const EPS = 1e-12;

  // ---- small numeric helpers (population std/mean, matching numpy) ----
  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function std(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return Math.sqrt(s / a.length); }
  function diff(a) { const o = new Array(a.length - 1); for (let i = 1; i < a.length; i++) o[i - 1] = a[i] - a[i - 1]; return o; }
  function slice(a, lo, hi) { return a.slice(lo, hi); }
  function last(a) { return a[a.length - 1]; }

  function zscoreWindow(w) {
    const m = mean(w), s = std(w);
    const out = new Array(w.length);
    if (s > EPS) for (let i = 0; i < w.length; i++) out[i] = (w[i] - m) / s;
    else out.fill(0);
    return out;
  }

  // ---- Long-term valuation (identical to Python) ----
  function daysSinceGenesis(nowMs) { return (nowMs - GENESIS_MS) / 1000 / SECONDS_PER_DAY; }

  function powerLaw(nowMs) {
    const d = daysSinceGenesis(nowMs);
    const center = Math.pow(10, PL_A + PL_B * Math.log10(d));
    return { days: d, center, support: center * PL_SUPPORT_FACTOR, resistance: center * PL_RESISTANCE_FACTOR };
  }

  function blockHeight(nowMs) { return Math.floor(daysSinceGenesis(nowMs) * BLOCKS_PER_DAY); }

  function supplyAndFlow(nowMs) {
    const height = blockHeight(nowMs);
    const eras = Math.floor(height / HALVING_INTERVAL);
    let supply = 0.0, reward = INITIAL_REWARD;
    for (let e = 0; e < eras; e++) { supply += HALVING_INTERVAL * reward; reward /= 2.0; }
    const blocksInCurrent = height - eras * HALVING_INTERVAL;
    supply += blocksInCurrent * reward;
    const currentReward = INITIAL_REWARD / Math.pow(2.0, eras);
    const flow = currentReward * BLOCKS_PER_YEAR;
    return { height, supply, current_reward: currentReward, flow, s2f_ratio: flow ? supply / flow : NaN };
  }

  function stockToFlow(nowMs) {
    const sf = supplyAndFlow(nowMs);
    return Object.assign({}, sf, { model_price: S2F_K * Math.pow(sf.s2f_ratio, S2F_B) });
  }

  function topCap(nowMs) {
    const pl = powerLaw(nowMs);
    const averagePrice = pl.center / (PL_B + 1.0);
    return { average_price: averagePrice, top_price: TOP_CAP_MULT * averagePrice };
  }

  // ---- Short-term Nwachukwu core (kernel-Bayesian + CFA) ----
  function kernelBayesianDelta(closes, length, nRef, c) {
    nRef = nRef || 1500; c = c || 0.25;
    const n = closes.length;
    if (n < length + 2) return 0.0;
    const maxStart = n - length - 1;
    let starts = [];
    for (let s = 0; s < maxStart; s++) starts.push(s);
    if (starts.length > nRef) {
      const sub = new Array(nRef);
      for (let i = 0; i < nRef; i++) {
        // mirror np.linspace(0, len-1, nRef).astype(int) (truncation toward 0)
        sub[i] = starts[Math.trunc(i * (starts.length - 1) / (nRef - 1))];
      }
      starts = sub;
    }
    const cur = zscoreWindow(slice(closes, n - length, n));
    let num = 0.0, den = 0.0;
    for (let k = 0; k < starts.length; k++) {
      const s = starts[k];
      const w = zscoreWindow(slice(closes, s, s + length));
      let d2 = 0.0;
      for (let i = 0; i < length; i++) { const dd = w[i] - cur[i]; d2 += dd * dd; }
      const weight = Math.exp(-c * d2);
      const delta = closes[s + length] - closes[s + length - 1];
      num += weight * delta; den += weight;
    }
    return den > EPS ? num / den : 0.0;
  }

  function kernelBayesianPrice(closes, lengths, c) {
    lengths = lengths || [30, 60, 120]; c = c || 0.25;
    const lp = last(closes);
    const deltas = [];
    for (const L of lengths) if (closes.length > L + 2) deltas.push(kernelBayesianDelta(closes, L, 1500, c));
    if (!deltas.length) return lp;
    return lp + mean(deltas);
  }

  function ema(series, span) {
    const alpha = 2.0 / (span + 1.0);
    const out = new Array(series.length);
    out[0] = series[0];
    for (let i = 1; i < series.length; i++) out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1];
    return out;
  }

  function baseModelPredictions(closes, volumes, orderFlowR) {
    const lp = last(closes);
    const out = {};
    out.Bayesian = kernelBayesianPrice(closes);

    out.Momentum = closes.length > 4 ? lp + mean(diff(slice(closes, closes.length - 4, closes.length))) : lp;

    let thrust = lp;
    if (closes.length > 4) {
      const drift = mean(diff(slice(closes, closes.length - 4, closes.length)));
      let surge = 1.0;
      if (volumes && volumes.length >= 30) {
        const avgV = mean(slice(volumes, volumes.length - 30, volumes.length)) || EPS;
        surge = Math.min(3.0, Math.max(0.3, last(volumes) / avgV));
      }
      thrust = lp + drift * 1.5 * surge;
    }
    out.VolThrust = thrust;

    if (closes.length > 6) {
      const ref = closes[closes.length - 6];
      const roc = (lp - ref) / Math.max(Math.abs(ref), EPS);
      out.ROC = lp * (1.0 + 0.5 * roc / 5.0);
    } else out.ROC = lp;

    if (closes.length > 10) {
      const d = diff(slice(closes, closes.length - 10, closes.length));
      let g = 0, l = 0;
      for (const x of d) { g += x > 0 ? x : 0; l += x < 0 ? -x : 0; }
      g /= d.length; l /= d.length;
      const rs = g / (l + EPS);
      const rsi = 100.0 - 100.0 / (1.0 + rs);
      const vol = std(d) || 1.0;
      let tilt = (rsi - 50.0) / 50.0;
      if (rsi > 70.0 || rsi < 30.0) tilt = -tilt * 0.5;
      out.RSI = lp + 0.4 * tilt * vol;
    } else out.RSI = lp;

    if (closes.length > 20) {
      const macd = ema(closes, 6).map((v, i) => v - ema(closes, 13)[i]);
      const signal = ema(macd, 5);
      out.EMA_MACD = lp + (last(macd) - last(signal));
    } else out.EMA_MACD = lp;

    const vol = closes.length > 20 ? (std(diff(slice(closes, closes.length - 20, closes.length))) || 1.0) : 1.0;
    out.OrderFlow = lp + 2.0 * orderFlowR * vol;

    return out;
  }

  function linspace(lo, hi, n) {
    const out = new Array(n);
    if (n === 1) { out[0] = lo; return out; }
    const step = (hi - lo) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = lo + step * i;
    return out;
  }

  function cfaFuse(preds, closes, gridSize) {
    gridSize = gridSize || 401;
    const names = Object.keys(preds);
    const means = names.map(n => preds[n]);
    const vol = closes.length > 60 ? std(diff(slice(closes, closes.length - 60, closes.length)))
                                   : (std(diff(closes)) || 1.0);
    const s = Math.max(vol, EPS);

    const lo = Math.min.apply(null, means) - 2.0 * s;
    let hi = Math.max.apply(null, means) + 2.0 * s;
    if (hi - lo < EPS) hi = lo + 1.0;
    const grid = linspace(lo, hi, gridSize);

    const scores = means.map(m => {
      const sc = new Array(gridSize);
      let peak = 0;
      for (let i = 0; i < gridSize; i++) {
        const z = (grid[i] - m) / s;
        let v = Math.exp(-0.5 * z * z);
        if (Math.abs(z) > 2.0) v = 0.0;
        sc[i] = v; if (v > peak) peak = v;
      }
      if (peak > EPS) for (let i = 0; i < gridSize; i++) sc[i] /= peak;
      else for (let i = 0; i < gridSize; i++) sc[i] = 0;
      return sc;
    });

    // RSC = scores sorted descending.
    const rscs = scores.map(sc => sc.slice().sort((a, b) => b - a));
    const t = names.length;
    const ds = new Array(t).fill(0);
    for (let j = 0; j < t; j++) {
      if (t > 1) {
        let acc = 0;
        for (let k = 0; k < t; k++) if (k !== j) {
          let sq = 0; for (let i = 0; i < gridSize; i++) { const d = rscs[j][i] - rscs[k][i]; sq += d * d; }
          acc += Math.sqrt(sq / gridSize);
        }
        ds[j] = acc / (t - 1);
      } else ds[j] = 1.0;
    }
    const dsSum = ds.reduce((a, b) => a + b, 0);
    const w = dsSum > EPS ? ds : new Array(t).fill(1.0);
    const wSum = w.reduce((a, b) => a + b, 0);

    let best = 0, bestVal = -Infinity;
    for (let i = 0; i < gridSize; i++) {
      let acc = 0; for (let m = 0; m < t; m++) acc += w[m] * scores[m][i];
      acc /= Math.max(wSum, EPS);
      if (acc > bestVal) { bestVal = acc; best = i; }
    }
    const diversity = {}, weights = {}, meansOut = {};
    names.forEach((nm, i) => { diversity[nm] = ds[i]; weights[nm] = w[i]; meansOut[nm] = means[i]; });
    return { predicted_price: grid[best], diversity, weights, means: meansOut };
  }

  function direction(d) { return d > 0 ? 'UP' : d < 0 ? 'DOWN' : 'FLAT'; }

  // ---- The Umeh formula ----
  function computeUmeh(closes, volumes, spot, orderFlowR, nowMs) {
    orderFlowR = orderFlowR || 0.0;
    nowMs = nowMs || Date.now();
    closes = closes.map(Number);
    if (volumes) volumes = volumes.map(Number);

    let series = closes;
    let volSeries = volumes;
    if (spot && isFinite(spot)) {
      series = closes.concat([Number(spot)]);
      if (volumes) volSeries = volumes.concat([last(volumes)]);
    }

    const preds = baseModelPredictions(series, volSeries, orderFlowR);
    const fused = cfaFuse(preds, series);
    const stPrice = fused.predicted_price;

    const pl = powerLaw(nowMs);
    const s2f = stockToFlow(nowMs);
    const tc = topCap(nowMs);

    const anchors = { market: spot, power_law: pl.center, top_cap: tc.top_price, stock_to_flow: s2f.model_price };
    let wsum = 0, lnFair = 0;
    for (const k in UMEH_WEIGHTS) { wsum += UMEH_WEIGHTS[k]; lnFair += UMEH_WEIGHTS[k] * Math.log(Math.max(anchors[k], EPS)); }
    const umehFair = Math.exp(lnFair / wsum);

    let band = (spot - pl.support) / Math.max(pl.resistance - pl.support, EPS);
    band = Math.max(0.0, Math.min(1.0, band));
    const pctOfFair = spot / Math.max(pl.center, EPS);
    const pctToTop = (tc.top_price - spot) / Math.max(tc.top_price, EPS);

    const valuation = 1.0 - band;
    const momentum = 0.5 + 0.5 * Math.tanh((stPrice - spot) / Math.max(0.001 * spot, EPS));
    const flow = 0.5 + 0.5 * Math.max(-1.0, Math.min(1.0, orderFlowR));
    const ceiling = Math.max(0.0, Math.min(1.0, pctToTop));
    let score = 100.0 * (0.40 * valuation + 0.25 * momentum + 0.20 * flow + 0.15 * ceiling);
    score = Math.max(0.0, Math.min(100.0, score));

    return {
      spot, as_of_ms: nowMs,
      short_term_price: stPrice,
      short_term_delta: stPrice - spot,
      short_term_dir: direction(stPrice - spot),
      base_means: fused.means, base_weights: fused.weights, base_diversity: fused.diversity,
      order_flow_r: orderFlowR,
      power_law: pl, stock_to_flow: s2f, top_cap: tc,
      umeh_fair_value: umehFair,
      pl_band_position: band, pct_of_fair: pctOfFair, pct_to_top_cap: pctToTop,
      umeh_score: score, n_ticks: closes.length,
    };
  }

  const api = {
    computeUmeh, powerLaw, supplyAndFlow, stockToFlow, topCap,
    kernelBayesianPrice, baseModelPredictions, cfaFuse, daysSinceGenesis,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Umeh = api;
})(typeof window !== 'undefined' ? window : globalThis);
