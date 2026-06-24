"""
umeh_model — the Umeh formula (educational)
============================================

A multi-factor Bitcoin valuation + short-term predictor that fuses four models
into one composite, the **Umeh formula**:

  1. Nwachukwu model   — the live short-term predictor (CFA fusion of diverse
                          base models + a kernel-Bayesian pattern estimator, in
                          the spirit of Shah & Zhang 2014). Drives the next-few-
                          minutes forecast from live 1-minute ticks.
  2. Power Law         — Santostasi's long-term fair value: price ~ A * days^n
                          (n ~ 5.8). A deterministic function of time.
  3. Stock-to-Flow     — PlanB's scarcity model: price ~ k * (stock/flow)^b.
                          Stock and flow come from the deterministic halving
                          schedule. (Known to run hot; weighted low.)
  4. Top Cap           — Willy Woo's cycle-ceiling: ~35 * Average Cap. Average
                          Cap is approximated from the power-law price integral.

The Umeh formula factors all of these simultaneously: it reports a near-term
predicted price (Nwachukwu, live), a long-term "Umeh fair value" (a weighted
geometric blend of the valuation anchors), where the current price sits inside
the power-law band and relative to the top-cap ceiling, and a single 0-100
"Umeh score" that leans bullish when price is cheap vs. the long-term models and
backed by live buy pressure, and bearish near the ceiling under sell pressure.

This file is the **Python reference implementation**. The GitHub Pages site runs
a JavaScript port (docs/umeh.js); tools/parity_check.py verifies the two produce
the same numbers on a fixed snapshot, so the page is a faithful mirror of this
model.

EDUCATIONAL USE ONLY. The constants are rough public fits; none of this is
financial advice, and these long-horizon models are contested and frequently
wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Shared constants (mirror these exactly in docs/umeh.js)
# --------------------------------------------------------------------------

GENESIS = datetime(2009, 1, 3, tzinfo=timezone.utc)   # Bitcoin genesis block
SECONDS_PER_DAY = 86400.0
BLOCKS_PER_DAY = 144.0                                  # ~10 min blocks
BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365.0
HALVING_INTERVAL = 210_000                              # blocks per reward era
INITIAL_REWARD = 50.0                                   # BTC per block, era 0

# Power Law (educational fit): log10(price) = PL_A + PL_B * log10(days).
# PL_A is calibrated so the center line is ~$60k around mid-2024 with PL_B=5.8.
PL_A = -17.0
PL_B = 5.8
PL_SUPPORT_FACTOR = 0.42      # lower band ~ support
PL_RESISTANCE_FACTOR = 2.10   # upper band ~ resistance

# Stock-to-Flow (PlanB-style): price = S2F_K * S2F_ratio ** S2F_B.
S2F_K = 0.40
S2F_B = 3.30

# Top Cap: top_price ~ TOP_CAP_MULT * average_price, average_price approximated
# by the power-law integral average = center / (PL_B + 1).
TOP_CAP_MULT = 35.0

# Umeh blend weights (geometric mean of ln(price) anchors). Power law is the
# most empirically robust, so it dominates; S2F is down-weighted (runs hot).
UMEH_WEIGHTS = {
    "market": 0.45,     # current price (near anchor)
    "power_law": 0.38,  # long-term fair value
    "top_cap": 0.12,    # ceiling pull
    "stock_to_flow": 0.05,  # scarcity (low weight: overshoots)
}


# --------------------------------------------------------------------------
# Long-term valuation models (closed form, identical in Py and JS)
# --------------------------------------------------------------------------

def days_since_genesis(now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (now - GENESIS).total_seconds() / SECONDS_PER_DAY


def power_law(now: Optional[datetime] = None) -> Dict[str, float]:
    """Power-law fair-value band for the current date."""
    d = days_since_genesis(now)
    center = 10.0 ** (PL_A + PL_B * math.log10(d))
    return {
        "days": d,
        "center": center,
        "support": center * PL_SUPPORT_FACTOR,
        "resistance": center * PL_RESISTANCE_FACTOR,
    }


def block_height(now: Optional[datetime] = None) -> int:
    return int(days_since_genesis(now) * BLOCKS_PER_DAY)


def supply_and_flow(now: Optional[datetime] = None) -> Dict[str, float]:
    """Deterministic circulating supply and annual flow from the halving schedule.

    Written in a plain loop so the JavaScript port is line-for-line identical.
    """
    height = block_height(now)
    eras = height // HALVING_INTERVAL          # completed halving eras
    supply = 0.0
    reward = INITIAL_REWARD
    for _ in range(eras):
        supply += HALVING_INTERVAL * reward
        reward /= 2.0
    # Partial blocks mined in the current era (reward == 50 / 2**eras).
    blocks_in_current = height - eras * HALVING_INTERVAL
    supply += blocks_in_current * reward
    current_reward = INITIAL_REWARD / (2.0 ** eras)
    flow = current_reward * BLOCKS_PER_YEAR
    return {
        "height": float(height),
        "supply": supply,
        "current_reward": current_reward,
        "flow": flow,
        "s2f_ratio": supply / flow if flow else float("nan"),
    }


def stock_to_flow(now: Optional[datetime] = None) -> Dict[str, float]:
    sf = supply_and_flow(now)
    price = S2F_K * (sf["s2f_ratio"] ** S2F_B)
    return {**sf, "model_price": price}


def top_cap(now: Optional[datetime] = None) -> Dict[str, float]:
    """Approximate Willy Woo Top Cap ceiling from the power-law integral."""
    pl = power_law(now)
    average_price = pl["center"] / (PL_B + 1.0)   # time-average of A*t^b over [0,D]
    return {
        "average_price": average_price,
        "top_price": TOP_CAP_MULT * average_price,
    }


# --------------------------------------------------------------------------
# Short-term Nwachukwu core (KMeans-free, fully portable to JS)
# --------------------------------------------------------------------------

_EPS = 1e-12


def _zscore_window(w: np.ndarray) -> np.ndarray:
    m = w.mean()
    s = w.std()
    return (w - m) / s if s > _EPS else np.zeros_like(w)


def kernel_bayesian_delta(closes: np.ndarray, length: int, n_ref: int = 1500,
                          c: float = 0.25) -> float:
    """Shah & Zhang kernel-Bayesian next-step delta for one window length.

    For the most recent normalized window of ``length`` points, RBF-weight every
    historical window of the same length and return the kernel-weighted average
    of their realized next-step price changes. No clustering — deterministic and
    identical in Python and JavaScript.
    """
    n = len(closes)
    if n < length + 2:
        return 0.0
    # Build reference windows + their next-step deltas.
    max_start = n - length - 1            # last index whose "next" delta exists
    starts = np.arange(0, max_start)
    if len(starts) > n_ref:
        starts = starts[np.linspace(0, len(starts) - 1, n_ref).astype(int)]
    cur = _zscore_window(closes[n - length:].astype(float))
    num = 0.0
    den = 0.0
    for s in starts:
        w = _zscore_window(closes[s:s + length].astype(float))
        d2 = float(np.dot(w - cur, w - cur))
        weight = math.exp(-c * d2)
        delta = float(closes[s + length] - closes[s + length - 1])
        num += weight * delta
        den += weight
    return num / den if den > _EPS else 0.0


def kernel_bayesian_price(closes: np.ndarray,
                          lengths=(30, 60, 120), c: float = 0.25) -> float:
    """Average kernel-Bayesian next price across several window lengths."""
    last = float(closes[-1])
    deltas = [kernel_bayesian_delta(closes, L, c=c) for L in lengths if len(closes) > L + 2]
    if not deltas:
        return last
    return last + float(np.mean(deltas))


# -- Simple base models (mirror nwachukwu_model, volume/flow aware) ----------

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def base_model_predictions(closes: np.ndarray, volumes: Optional[np.ndarray],
                           order_flow_r: float) -> Dict[str, float]:
    """Each diverse base model's next-price estimate (1-minute horizon)."""
    last = float(closes[-1])
    out: Dict[str, float] = {}

    # Kernel-Bayesian pattern model (the Nwachukwu core).
    out["Bayesian"] = kernel_bayesian_price(closes)

    # Short momentum (drift over last 3).
    if len(closes) > 4:
        out["Momentum"] = last + float(np.mean(np.diff(closes[-4:])))
    else:
        out["Momentum"] = last

    # Volume-confirmed thrust.
    thrust = last
    if len(closes) > 4:
        drift = float(np.mean(np.diff(closes[-4:])))
        surge = 1.0
        if volumes is not None and len(volumes) >= 30:
            avg_v = float(np.mean(volumes[-30:])) or _EPS
            surge = min(3.0, max(0.3, float(volumes[-1]) / avg_v))
        thrust = last + drift * 1.5 * surge
    out["VolThrust"] = thrust

    # Rate of change.
    if len(closes) > 6:
        roc = (last - float(closes[-6])) / max(abs(float(closes[-6])), _EPS)
        out["ROC"] = last * (1.0 + 0.5 * roc / 5.0)
    else:
        out["ROC"] = last

    # RSI tilt (lean with momentum, fade extremes).
    if len(closes) > 10:
        diffs = np.diff(closes[-10:])
        gains = float(np.mean(np.where(diffs > 0, diffs, 0.0)))
        losses = float(np.mean(np.where(diffs < 0, -diffs, 0.0)))
        rs = gains / (losses + _EPS)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        vol = float(np.std(diffs)) or 1.0
        tilt = (rsi - 50.0) / 50.0
        if rsi > 70.0 or rsi < 30.0:
            tilt = -tilt * 0.5
        out["RSI"] = last + 0.4 * tilt * vol
    else:
        out["RSI"] = last

    # EMA/MACD momentum.
    if len(closes) > 20:
        macd = _ema(closes.astype(float), 6) - _ema(closes.astype(float), 13)
        signal = _ema(macd, 5)
        out["EMA_MACD"] = last + float(macd[-1] - signal[-1])
    else:
        out["EMA_MACD"] = last

    # Live order-flow tilt (buy/sell pressure).
    vol = float(np.std(np.diff(closes[-20:]))) if len(closes) > 20 else 1.0
    out["OrderFlow"] = last + 2.0 * order_flow_r * (vol or 1.0)

    return out


def cfa_fuse(preds: Dict[str, float], closes: np.ndarray,
             grid_size: int = 401) -> Dict[str, object]:
    """Combinatorial Fusion Analysis over base-model predictions (portable)."""
    names = list(preds.keys())
    means = np.array([preds[n] for n in names], dtype=float)
    # Per-model std from recent volatility (shared scale keeps JS parity simple).
    vol = float(np.std(np.diff(closes[-60:]))) if len(closes) > 60 else float(np.std(np.diff(closes)) or 1.0)
    std = max(vol, _EPS)

    lo = float(means.min() - 2.0 * std)
    hi = float(means.max() + 2.0 * std)
    if hi - lo < _EPS:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, grid_size)

    # Truncated-normal scores, normalized to peak 1.
    scores = []
    for m in means:
        z = (grid - m) / std
        sc = np.exp(-0.5 * z * z)
        sc = np.where(np.abs(z) <= 2.0, sc, 0.0)
        peak = sc.max()
        scores.append(sc / peak if peak > _EPS else np.zeros_like(grid))
    scores = np.vstack(scores)

    # Rank-Score Characteristic functions + cognitive diversity.
    rscs = np.vstack([np.sort(scores[i])[::-1] for i in range(len(names))])
    t = len(names)
    ds = np.zeros(t)
    for j in range(t):
        ds[j] = np.mean([
            math.sqrt(float(np.mean((rscs[j] - rscs[k]) ** 2)))
            for k in range(t) if k != j
        ]) if t > 1 else 1.0
    w = ds if ds.sum() > _EPS else np.ones(t)

    combined = (w[:, None] * scores).sum(axis=0) / max(w.sum(), _EPS)
    best = int(np.argmax(combined))
    return {
        "predicted_price": float(grid[best]),
        "diversity": {names[i]: float(ds[i]) for i in range(t)},
        "weights": {names[i]: float(w[i]) for i in range(t)},
        "means": {names[i]: float(means[i]) for i in range(t)},
    }


# --------------------------------------------------------------------------
# The Umeh formula — factor everything simultaneously
# --------------------------------------------------------------------------

@dataclass
class UmehResult:
    spot: float
    as_of_utc: str
    # short term
    short_term_price: float
    short_term_delta: float
    short_term_dir: str
    base_means: Dict[str, float] = field(default_factory=dict)
    base_weights: Dict[str, float] = field(default_factory=dict)
    base_diversity: Dict[str, float] = field(default_factory=dict)
    order_flow_r: float = 0.0
    # long term anchors
    power_law: Dict[str, float] = field(default_factory=dict)
    stock_to_flow: Dict[str, float] = field(default_factory=dict)
    top_cap: Dict[str, float] = field(default_factory=dict)
    # composite
    umeh_fair_value: float = 0.0
    pl_band_position: float = 0.0   # 0=support .. 1=resistance
    pct_of_fair: float = 0.0        # price / power-law center
    pct_to_top_cap: float = 0.0     # headroom to ceiling
    umeh_score: float = 50.0        # 0 (distribute) .. 100 (accumulate)
    n_ticks: int = 0


def _direction(delta: float) -> str:
    return "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"


# --------------------------------------------------------------------------
# Slow-horizon Nwachukwu forecast (15-minute / hourly): price-pattern roster.
# Backtests showed the fast thrust/volume models add noise at slower horizons,
# so these use the original price-only roster (no order flow).
# --------------------------------------------------------------------------

def base_model_predictions_slow(closes: np.ndarray) -> Dict[str, float]:
    last = float(closes[-1])
    out: Dict[str, float] = {}
    out["Bayesian"] = kernel_bayesian_price(closes)
    out["Momentum"] = last + float(np.mean(np.diff(closes[-11:]))) if len(closes) > 11 else last
    n = min(20, len(closes))
    out["MeanRev"] = last + 0.25 * (float(np.mean(closes[-n:])) - last)
    if len(closes) > 35:
        macd = _ema(closes.astype(float), 12) - _ema(closes.astype(float), 26)
        signal = _ema(macd, 9)
        out["EMA_MACD"] = last + float(macd[-1] - signal[-1])
    else:
        out["EMA_MACD"] = last
    out["RandomWalk"] = last
    return out


def nwachukwu_slow(closes: np.ndarray, spot: float) -> Dict[str, object]:
    """Fuse the price-pattern roster for one slow-horizon next-candle forecast."""
    closes = np.asarray(closes, dtype=float).ravel()
    series = np.append(closes, float(spot)) if spot and math.isfinite(spot) else closes
    preds = base_model_predictions_slow(series)
    fused = cfa_fuse(preds, series)
    delta = fused["predicted_price"] - float(spot)
    return {
        "predicted_price": float(fused["predicted_price"]),
        "delta": float(delta),
        "direction": _direction(delta),
        "base_means": fused["means"],
        "base_weights": fused["weights"],
    }



def compute_umeh(closes: np.ndarray, volumes: Optional[np.ndarray], spot: float,
                 order_flow_r: float = 0.0,
                 now: Optional[datetime] = None) -> UmehResult:
    """Run the full Umeh formula on a 1-minute close/volume series + live spot."""
    now = now or datetime.now(timezone.utc)
    closes = np.asarray(closes, dtype=float).ravel()
    if volumes is not None:
        volumes = np.asarray(volumes, dtype=float).ravel()

    # Feed the live spot as the most-recent tick so the formula is live.
    series = np.append(closes, float(spot)) if spot and math.isfinite(spot) else closes
    vol_series = volumes
    if volumes is not None and len(series) > len(volumes):
        vol_series = np.append(volumes, volumes[-1])

    # 1) Short-term Nwachukwu (CFA fusion of base models).
    preds = base_model_predictions(series, vol_series, order_flow_r)
    fused = cfa_fuse(preds, series)
    st_price = fused["predicted_price"]

    # 2) Long-term anchors.
    pl = power_law(now)
    s2f = stock_to_flow(now)
    tc = top_cap(now)

    # 3) Umeh composite fair value: weighted geometric mean of ln(anchors).
    anchors = {
        "market": spot,
        "power_law": pl["center"],
        "top_cap": tc["top_price"],
        "stock_to_flow": s2f["model_price"],
    }
    wsum = sum(UMEH_WEIGHTS.values())
    ln_fair = sum(UMEH_WEIGHTS[k] * math.log(max(v, _EPS)) for k, v in anchors.items()) / wsum
    umeh_fair = math.exp(ln_fair)

    # 4) Where price sits in the power-law band and vs. the ceiling.
    band = (spot - pl["support"]) / max(pl["resistance"] - pl["support"], _EPS)
    band = max(0.0, min(1.0, band))
    pct_of_fair = spot / max(pl["center"], _EPS)
    pct_to_top = (tc["top_price"] - spot) / max(tc["top_price"], _EPS)

    # 5) Umeh score (0-100): cheap vs fair + buy pressure + positive short-term
    #    => accumulate; expensive/near ceiling + sell pressure => distribute.
    valuation = (1.0 - band)               # 1 at support, 0 at resistance
    momentum = 0.5 + 0.5 * math.tanh((st_price - spot) / max(0.001 * spot, _EPS))
    flow = 0.5 + 0.5 * max(-1.0, min(1.0, order_flow_r))
    ceiling = max(0.0, min(1.0, pct_to_top))
    score = 100.0 * (0.40 * valuation + 0.25 * momentum + 0.20 * flow + 0.15 * ceiling)

    return UmehResult(
        spot=float(spot),
        as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        short_term_price=float(st_price),
        short_term_delta=float(st_price - spot),
        short_term_dir=_direction(st_price - spot),
        base_means=fused["means"],
        base_weights=fused["weights"],
        base_diversity=fused["diversity"],
        order_flow_r=float(order_flow_r),
        power_law=pl,
        stock_to_flow=s2f,
        top_cap=tc,
        umeh_fair_value=float(umeh_fair),
        pl_band_position=float(band),
        pct_of_fair=float(pct_of_fair),
        pct_to_top_cap=float(pct_to_top),
        umeh_score=float(max(0.0, min(100.0, score))),
        n_ticks=int(len(closes)),
    )
