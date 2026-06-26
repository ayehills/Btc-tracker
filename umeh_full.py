#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 umeh_full.py  —  The Umeh Formula (single-file, high-depth reference build)
================================================================================

A complete, self-contained Bitcoin price-analysis engine that fuses every
technique developed in this project into ONE composite, the **Umeh formula**.
It runs on live market data and prints a deep, multi-section report; it can also
backtest its own short-term predictor on held-out candles.

    EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE.
    The long-horizon valuation models (power law, stock-to-flow, top cap) are
    contested public fits and are frequently wrong. Backtested or modeled
    performance does not imply live results. Trade at your own risk.

--------------------------------------------------------------------------------
 WHAT IS COMBINED (the four pillars of the Umeh formula)
--------------------------------------------------------------------------------

  1. NWACHUKWU MODEL  — the live short-term predictor.
       * A KMeans-free kernel-Bayesian estimator in the spirit of
         Shah & Zhang (2014), "Bayesian Regression and Bitcoin": for the most
         recent normalized price window it RBF-weights every historical window
         of the same length and returns the kernel-weighted average of their
         realized next-step moves, across several window lengths.
       * A roster of diverse base forecasters (momentum, mean-reversion,
         EMA/MACD, RSI, rate-of-change, volume-confirmed thrust, and a live
         order-book-imbalance model) each produce a next-price estimate.
       * These are fused with Combinatorial Fusion Analysis (CFA) — Wu, Ye, Xu
         & Hsu (IEEE CAI): each model becomes a scoring system over a grid of
         candidate prices; rank-score characteristic (RSC) functions yield a
         cognitive-diversity weighting; a weighted score combination picks the
         fused prediction.
       * Two rosters: a FAST roster (1-minute horizon, includes thrust + order
         flow) and a SLOW roster (15-minute / hourly horizons, price-pattern
         only — backtests showed thrust models add noise at slower horizons).

  2. POWER LAW  — Santostasi's long-term fair value: price ~ A * days^n
       (n ~ 5.8). A deterministic function of time since the genesis block,
       giving support / center / resistance bands.

  3. STOCK-TO-FLOW  — PlanB's scarcity model: price ~ k * (stock/flow)^b.
       Stock and flow are derived deterministically from the halving schedule.
       (Known to run hot post-2024-halving; weighted low in the blend.)

  4. TOP CAP  — Willy Woo's cycle ceiling: ~35 * Average Cap. Average Cap is
       approximated here from the power-law price integral (a transparent
       proxy, since on-chain realized cap is not in the candle feed).

The Umeh formula factors all of these SIMULTANEOUSLY: it reports a near-term
forecast (Nwachukwu, live), 15-minute and hourly forecasts, a multi-scenario
projection fan (bull / base / bear) that bends to last-minute jumps, an Umeh
fair value (a weighted geometric blend of the valuation anchors), where price
sits inside the power-law band and relative to the top-cap ceiling, and a single
0-100 "Umeh score" that leans bullish when price is cheap vs. the long-term
models and backed by live buy pressure, bearish near the ceiling under selling.

--------------------------------------------------------------------------------
 DATA SOURCES (no API key required; public market-data endpoints)
--------------------------------------------------------------------------------
  * Binance.US  /api/v3/klines    — 1m / 15m / 1h OHLCV (paginated for history)
  * Binance.US  /api/v3/ticker    — live spot
  * Coinbase    /products/.../book — live order book for buy/sell imbalance

--------------------------------------------------------------------------------
 REQUIREMENTS
--------------------------------------------------------------------------------
    python >= 3.9
    numpy

    (Networking uses only the standard library: urllib. No 'requests' needed.)

--------------------------------------------------------------------------------
 QUICK START
--------------------------------------------------------------------------------
    pip install numpy

    python umeh_full.py                 # full live report (default)
    python umeh_full.py predict         # same as default
    python umeh_full.py backtest        # walk-forward accuracy on 1m & 15m
    python umeh_full.py monitor --every 10   # live loop, refresh every 10s
    python umeh_full.py anchors         # just the long-term valuation models
    python umeh_full.py --json          # machine-readable JSON output

--------------------------------------------------------------------------------
 AUTHORSHIP / PROVENANCE
--------------------------------------------------------------------------------
This single file consolidates and expands the project's modules
(nwachukwu_model.py, umeh_model.py, predictor_service.py) plus the client-side
projection logic, into one downloadable reference. The numeric methods mirror
the repository's parity-checked Python/JS implementations.

================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: numpy is required. Install it with:  pip install numpy\n"
        f"(import error: {exc})\n"
    )
    raise

__version__ = "1.0.0"
_EPS = 1e-12


# ============================================================================
# SECTION 1 — CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """All tunable constants for the Umeh engine, in one place."""

    # ---- Data / networking -------------------------------------------------
    binance_base: str = "https://api.binance.us/api/v3"
    coinbase_base: str = "https://api.exchange.coinbase.com"
    # Kalshi public read-only market data (no auth required for market browsing).
    kalshi_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_series: str = "KXBTC15M"  # Kalshi "BTC price up in next 15 mins?" market
    symbol_binance: str = "BTCUSDT"
    product_coinbase: str = "BTC-USD"
    http_timeout: float = 25.0
    user_agent: str = "umeh-full/1.0 (+educational)"

    # How much history to analyze, per horizon (number of candles).
    ticks_1m: int = 10080      # ~1 week of 1-minute candles (paginated)
    ticks_15m: int = 1000      # ~10 days of 15-minute candles
    ticks_1h: int = 1000       # ~41 days of hourly candles
    max_analyze: int = 700     # cap the window actually fed to the slow models

    # ---- Kernel-Bayesian (Shah & Zhang) -----------------------------------
    kb_window_lengths: Tuple[int, ...] = (30, 60, 120)
    kb_n_ref: int = 1500       # subsample of historical windows (speed)
    kb_smoothing_c: float = 0.25

    # ---- CFA fusion --------------------------------------------------------
    cfa_grid_size: int = 401
    cfa_trunc_std: float = 2.0

    # ---- Power Law (educational fit) --------------------------------------
    pl_a: float = -17.0
    pl_b: float = 5.8
    pl_support_factor: float = 0.42
    pl_resistance_factor: float = 2.10

    # ---- Stock-to-Flow (PlanB-style) --------------------------------------
    s2f_k: float = 0.40
    s2f_b: float = 3.30

    # ---- Top Cap (Willy Woo) ----------------------------------------------
    top_cap_mult: float = 35.0

    # ---- Bitcoin emission schedule ----------------------------------------
    genesis: datetime = datetime(2009, 1, 3, tzinfo=timezone.utc)
    blocks_per_day: float = 144.0
    halving_interval: int = 210_000
    initial_reward: float = 50.0

    # ---- Umeh composite blend weights (geometric mean of ln-anchors) ------
    umeh_w_market: float = 0.45
    umeh_w_power_law: float = 0.38
    umeh_w_top_cap: float = 0.12
    umeh_w_s2f: float = 0.05

    # ---- Umeh score weights (0..1) ----------------------------------------
    score_w_valuation: float = 0.40
    score_w_momentum: float = 0.25
    score_w_flow: float = 0.20
    score_w_ceiling: float = 0.15

    # ---- Projection fan ----------------------------------------------------
    proj_horizon_min: int = 60      # project this many minutes ahead
    proj_band_z: float = 1.5        # band width = z * vol * sqrt(minutes)
    proj_jump_minutes: int = 3      # how long a last-minute jump biases the path

    @property
    def blocks_per_year(self) -> float:
        return self.blocks_per_day * 365.0


# ============================================================================
# SECTION 2 — LOW-LEVEL NUMERIC UTILITIES
# ============================================================================

def zscore(window: np.ndarray) -> np.ndarray:
    """Standardize a 1-D window; return zeros if the window is flat."""
    w = np.asarray(window, dtype=np.float64)
    m = w.mean()
    s = w.std()
    return (w - m) / s if s > _EPS else np.zeros_like(w)


def ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (recursive form)."""
    series = np.asarray(series, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def population_std(arr: Sequence[float]) -> float:
    a = np.asarray(arr, dtype=np.float64)
    return float(a.std()) if a.size else 0.0


def rsi(closes: np.ndarray, n: int = 14) -> float:
    """Relative Strength Index over the last ``n`` deltas (0..100)."""
    closes = np.asarray(closes, dtype=np.float64)
    if len(closes) < n + 1:
        return 50.0
    d = np.diff(closes[-(n + 1):])
    gains = float(np.mean(np.where(d > 0, d, 0.0)))
    losses = float(np.mean(np.where(d < 0, -d, 0.0)))
    rs = gains / (losses + _EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def per_minute_volatility(closes: np.ndarray, lookback: int = 30) -> float:
    """Std of recent one-step price changes (a per-candle volatility)."""
    closes = np.asarray(closes, dtype=np.float64)
    n = min(lookback + 1, len(closes))
    if n < 3:
        return 1.0
    return float(np.std(np.diff(closes[-n:]))) or 1.0


def fmt_usd(x: float, dp: int = 2) -> str:
    try:
        return "$" + format(float(x), ",." + str(dp) + "f")
    except Exception:
        return "$nan"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================
# SECTION 3 — DATA LAYER (Binance.US klines/ticker, Coinbase order book)
# ============================================================================

class DataError(RuntimeError):
    pass


def _http_get_json(url: str, cfg: Config):
    """GET a URL and parse JSON, using only the standard library."""
    req = Request(url, headers={"User-Agent": cfg.user_agent})
    try:
        with urlopen(req, timeout=cfg.http_timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DataError(f"request failed: {url} ({exc})") from exc


@dataclass
class Candles:
    """Ascending OHLCV-derived series for one interval."""
    times: np.ndarray      # open time, seconds (int64)
    closes: np.ndarray     # close price (float64)
    volumes: np.ndarray    # base-asset volume (float64)

    def __len__(self) -> int:
        return len(self.closes)

    def tail(self, n: int) -> "Candles":
        return Candles(self.times[-n:], self.closes[-n:], self.volumes[-n:])


def fetch_klines(interval: str, limit: int, cfg: Config,
                 end_time_ms: Optional[int] = None) -> List[list]:
    """Raw Binance.US klines (list of rows). limit <= 1000."""
    url = (f"{cfg.binance_base}/klines?symbol={cfg.symbol_binance}"
           f"&interval={interval}&limit={min(limit, 1000)}")
    if end_time_ms is not None:
        url += f"&endTime={end_time_ms}"
    data = _http_get_json(url, cfg)
    if not isinstance(data, list):
        raise DataError(f"unexpected klines response: {str(data)[:160]}")
    return data


def fetch_candles(interval: str, total: int, cfg: Config) -> Candles:
    """Fetch up to ``total`` candles for an interval, paginating backward.

    Binance.US returns at most 1000 rows per call, so for deep 1-minute history
    we page backward via ``endTime`` until we have enough.
    """
    rows: List[list] = []
    end_ms: Optional[int] = None
    guard = 0
    while len(rows) < total and guard < 30:
        guard += 1
        batch = fetch_klines(interval, 1000, cfg, end_ms)
        if not batch:
            break
        rows = batch + rows
        end_ms = int(batch[0][0]) - 1
        if len(batch) < 1000:
            break
    # de-duplicate by open time and sort ascending
    seen = set()
    clean: List[list] = []
    for r in rows:
        t = int(r[0])
        if t not in seen:
            seen.add(t)
            clean.append(r)
    clean.sort(key=lambda r: r[0])
    clean = clean[-total:]
    times = np.array([int(r[0]) // 1000 for r in clean], dtype=np.int64)
    closes = np.array([float(r[4]) for r in clean], dtype=np.float64)
    volumes = np.array([float(r[5]) for r in clean], dtype=np.float64)
    return Candles(times, closes, volumes)


def fetch_spot(cfg: Config) -> float:
    """Live BTC spot from Binance.US ticker."""
    url = f"{cfg.binance_base}/ticker/price?symbol={cfg.symbol_binance}"
    data = _http_get_json(url, cfg)
    return float(data["price"])


@dataclass
class KalshiImplied:
    """Kalshi's "BTC price up in next 15 mins?" market (KXBTC15M).

    A single binary contract per 15-minute window: Yes settles if BTC is ABOVE
    the target ("To Beat") price at expiration, per CF Benchmarks' Real-Time
    Index. So the market gives us two comparable numbers vs. Binance:
      * target_price  — the reference level the contract is measured against;
      * prob_up       — the market-implied probability BTC finishes above it.
    """
    ok: bool
    series: str = ""
    ticker: str = ""
    settle_utc: str = ""
    minutes_to_settle: int = 0
    target_price: float = float("nan")    # the "To Beat" / target level
    prob_up: float = float("nan")         # market P(BTC above target at settle)
    last_price: float = float("nan")      # last traded Yes price (a probability)
    note: str = ""


def fetch_kalshi_btc(cfg: Config, now: Optional[datetime] = None) -> KalshiImplied:
    """Read Kalshi's active 15-minute "BTC up?" market (KXBTC15M).

    Returns ok=False (rather than raising) so the engine degrades gracefully if
    Kalshi is unreachable. Picks the active window (nearest future settlement).
    """
    now = now or datetime.now(timezone.utc)
    try:
        url = (f"{cfg.kalshi_base}/markets?limit=50"
               f"&series_ticker={cfg.kalshi_series}&status=open")
        data = _http_get_json(url, cfg)
        markets = [m for m in data.get("markets", []) if m.get("floor_strike") is not None]
        if not markets:
            return KalshiImplied(ok=False, note="no open Kalshi 15-min BTC market")

        # Choose the nearest future settlement (the live 15-minute window).
        def close_dt(m):
            return datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
        future = [m for m in markets if close_dt(m) > now]
        chosen = min(future, key=close_dt) if future else min(markets, key=close_dt)
        dt = close_dt(chosen)

        target = float(chosen["floor_strike"])
        yb = float(chosen.get("yes_bid_dollars") or 0.0)
        ya = float(chosen.get("yes_ask_dollars") or 0.0)
        prob_up = (yb + ya) / 2.0 if (yb or ya) else float("nan")
        last = float(chosen.get("last_price_dollars") or 0.0) or float("nan")
        minutes = max(0, int(round((dt - now).total_seconds() / 60)))
        return KalshiImplied(
            ok=True, series=cfg.kalshi_series, ticker=chosen.get("ticker", ""),
            settle_utc=dt.strftime("%H:%M UTC"), minutes_to_settle=minutes,
            target_price=target, prob_up=prob_up, last_price=last,
        )
    except Exception as exc:
        return KalshiImplied(ok=False, note=f"Kalshi fetch failed: {exc}")


def fetch_order_flow(cfg: Config) -> float:
    """Order-book imbalance r = (bid_vol - ask_vol)/(bid_vol + ask_vol).

    Positive => more resting bids than asks (buy pressure). Returns 0.0 on
    failure so the engine degrades gracefully to price-only.
    """
    try:
        url = f"{cfg.coinbase_base}/products/{cfg.product_coinbase}/book?level=2"
        data = _http_get_json(url, cfg)
        bid = sum(float(x[1]) for x in data.get("bids", []))
        ask = sum(float(x[1]) for x in data.get("asks", []))
        denom = bid + ask
        return float((bid - ask) / denom) if denom else 0.0
    except Exception:
        return 0.0


# ============================================================================
# SECTION 4 — LONG-TERM VALUATION MODELS (closed form, deterministic)
# ============================================================================

def days_since_genesis(cfg: Config, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (now - cfg.genesis).total_seconds() / 86400.0


def power_law(cfg: Config, now: Optional[datetime] = None) -> Dict[str, float]:
    """Power-law fair-value band for the current date.

    log10(price) = pl_a + pl_b * log10(days_since_genesis)
    """
    d = days_since_genesis(cfg, now)
    center = 10.0 ** (cfg.pl_a + cfg.pl_b * math.log10(d))
    return {
        "days": d,
        "center": center,
        "support": center * cfg.pl_support_factor,
        "resistance": center * cfg.pl_resistance_factor,
    }


def block_height(cfg: Config, now: Optional[datetime] = None) -> int:
    return int(days_since_genesis(cfg, now) * cfg.blocks_per_day)


def supply_and_flow(cfg: Config, now: Optional[datetime] = None) -> Dict[str, float]:
    """Deterministic circulating supply and annual flow from the halving schedule."""
    height = block_height(cfg, now)
    eras = height // cfg.halving_interval
    supply = 0.0
    reward = cfg.initial_reward
    for _ in range(eras):
        supply += cfg.halving_interval * reward
        reward /= 2.0
    blocks_in_current = height - eras * cfg.halving_interval
    supply += blocks_in_current * reward
    current_reward = cfg.initial_reward / (2.0 ** eras)
    flow = current_reward * cfg.blocks_per_year
    return {
        "height": float(height),
        "supply": supply,
        "current_reward": current_reward,
        "flow": flow,
        "s2f_ratio": supply / flow if flow else float("nan"),
        "eras": float(eras),
    }


def stock_to_flow(cfg: Config, now: Optional[datetime] = None) -> Dict[str, float]:
    sf = supply_and_flow(cfg, now)
    price = cfg.s2f_k * (sf["s2f_ratio"] ** cfg.s2f_b)
    return {**sf, "model_price": price}


def top_cap(cfg: Config, now: Optional[datetime] = None) -> Dict[str, float]:
    """Approximate Willy Woo Top Cap ceiling from the power-law integral.

    Average price over [0, D] of A*t^b is A*D^b/(b+1) = center/(b+1).
    Top price ~ mult * average price.
    """
    pl = power_law(cfg, now)
    average_price = pl["center"] / (cfg.pl_b + 1.0)
    return {
        "average_price": average_price,
        "top_price": cfg.top_cap_mult * average_price,
    }


# ============================================================================
# SECTION 5 — KERNEL-BAYESIAN ESTIMATOR (Shah & Zhang, KMeans-free)
# ============================================================================

def kernel_bayesian_delta(closes: np.ndarray, length: int, cfg: Config) -> float:
    """Kernel-weighted expected next-step change for one window length.

    For the most recent normalized window of ``length`` points, RBF-weight every
    historical normalized window of the same length and return the weighted
    average of their realized next-step price changes.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    if n < length + 2:
        return 0.0
    max_start = n - length - 1
    starts = np.arange(0, max_start)
    if len(starts) > cfg.kb_n_ref:
        idx = np.linspace(0, len(starts) - 1, cfg.kb_n_ref).astype(int)
        starts = starts[idx]
    cur = zscore(closes[n - length:])
    num = 0.0
    den = 0.0
    c = cfg.kb_smoothing_c
    for s in starts:
        w = zscore(closes[s:s + length])
        diff = w - cur
        d2 = float(np.dot(diff, diff))
        weight = math.exp(-c * d2)
        delta = float(closes[s + length] - closes[s + length - 1])
        num += weight * delta
        den += weight
    return num / den if den > _EPS else 0.0


def kernel_bayesian_price(closes: np.ndarray, cfg: Config) -> float:
    """Average kernel-Bayesian next price across the configured window lengths."""
    closes = np.asarray(closes, dtype=np.float64)
    last = float(closes[-1])
    deltas = [
        kernel_bayesian_delta(closes, L, cfg)
        for L in cfg.kb_window_lengths if len(closes) > L + 2
    ]
    if not deltas:
        return last
    return last + float(np.mean(deltas))


# ============================================================================
# SECTION 6 — BASE FORECASTERS (each yields a next-price estimate)
# ============================================================================

class BaseForecaster:
    """Interface: predict the next price from closes (+ optional volume/flow)."""

    name: str = "base"

    def predict(self, closes: np.ndarray, cfg: Config,
                volumes: Optional[np.ndarray] = None,
                order_flow_r: float = 0.0) -> float:
        raise NotImplementedError


class BayesianForecaster(BaseForecaster):
    name = "Bayesian"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        return kernel_bayesian_price(closes, cfg)


class MomentumForecaster(BaseForecaster):
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"Momentum{k}"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        if len(closes) < self.k + 1:
            return last
        return last + float(np.mean(np.diff(closes[-(self.k + 1):])))


class MeanReversionForecaster(BaseForecaster):
    def __init__(self, n: int = 20, alpha: float = 0.25):
        self.n = n
        self.alpha = alpha
        self.name = f"MeanRev{n}"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        n = min(self.n, len(closes))
        sma = float(np.mean(closes[-n:]))
        return last + self.alpha * (sma - last)


class EmaMacdForecaster(BaseForecaster):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, name: str = "EMA_MACD"):
        self.fast, self.slow, self.signal = fast, slow, signal
        self.name = name

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        if len(closes) < self.slow + self.signal:
            return last
        macd = ema(closes, self.fast) - ema(closes, self.slow)
        signal_line = ema(macd, self.signal)
        return last + float(macd[-1] - signal_line[-1])


class RsiForecaster(BaseForecaster):
    def __init__(self, n: int = 14, gain: float = 0.4):
        self.n = n
        self.gain = gain
        self.name = "RSI"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        if len(closes) < self.n + 1:
            return last
        r = rsi(closes, self.n)
        vol = float(np.std(np.diff(closes[-(self.n + 1):]))) or 1.0
        tilt = (r - 50.0) / 50.0
        if r > 70.0 or r < 30.0:
            tilt = -tilt * 0.5   # fade extremes
        return last + self.gain * tilt * vol


class RateOfChangeForecaster(BaseForecaster):
    def __init__(self, n: int = 5, damp: float = 0.5):
        self.n = n
        self.damp = damp
        self.name = f"ROC{n}"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        if len(closes) < self.n + 1:
            return last
        ref = float(closes[-self.n - 1])
        roc = (last - ref) / max(abs(ref), _EPS)
        return last * (1.0 + self.damp * roc / self.n)


class VolumeThrustForecaster(BaseForecaster):
    """Volume-confirmed momentum — leans into a buy/sell push backed by volume."""

    def __init__(self, k: int = 3, vlookback: int = 30, gain: float = 1.5):
        self.k, self.vlookback, self.gain = k, vlookback, gain
        self.name = "VolThrust"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        if len(closes) < self.k + 1:
            return last
        drift = float(np.mean(np.diff(closes[-(self.k + 1):])))
        surge = 1.0
        if volumes is not None and len(volumes) >= self.vlookback:
            avg_v = float(np.mean(volumes[-self.vlookback:])) or _EPS
            surge = clamp(float(volumes[-1]) / avg_v, 0.3, 3.0)
        return last + drift * self.gain * surge


class OrderFlowForecaster(BaseForecaster):
    """Live order-book imbalance tilt (buy/sell pressure). Live-only."""

    def __init__(self, gain: float = 2.0):
        self.gain = gain
        self.name = "OrderFlow"

    def predict(self, closes, cfg, volumes=None, order_flow_r=0.0):
        closes = np.asarray(closes, dtype=np.float64)
        last = float(closes[-1])
        vol = float(np.std(np.diff(closes[-20:]))) if len(closes) > 2 else 1.0
        return last + self.gain * order_flow_r * (vol or 1.0)


def fast_roster() -> List[BaseForecaster]:
    """1-minute roster: includes thrust + (live) order flow."""
    return [
        BayesianForecaster(),
        MomentumForecaster(k=3),
        VolumeThrustForecaster(k=3),
        RateOfChangeForecaster(n=5),
        RsiForecaster(n=9),
        EmaMacdForecaster(6, 13, 5),
        # OrderFlow is appended live only when an imbalance is supplied.
    ]


def slow_roster() -> List[BaseForecaster]:
    """15-minute / hourly roster: price-pattern only (no thrust/flow)."""
    return [
        BayesianForecaster(),
        MomentumForecaster(k=10),
        MeanReversionForecaster(n=20, alpha=0.25),
        EmaMacdForecaster(12, 26, 9),
    ]


# ============================================================================
# SECTION 7 — COMBINATORIAL FUSION ANALYSIS (CFA)
# ============================================================================

@dataclass
class FusionResult:
    predicted_price: float
    means: Dict[str, float]
    weights: Dict[str, float]
    diversity: Dict[str, float]
    grid_low: float
    grid_high: float


def _normal_scores(grid: np.ndarray, mean: float, std: float,
                   trunc: float) -> np.ndarray:
    std = max(std, _EPS)
    z = (grid - mean) / std
    sc = np.exp(-0.5 * z * z)
    sc = np.where(np.abs(z) <= trunc, sc, 0.0)
    peak = sc.max()
    return sc / peak if peak > _EPS else np.zeros_like(grid)


def cfa_fuse(preds: Dict[str, float], closes: np.ndarray,
             cfg: Config) -> FusionResult:
    """Fuse base-model predictions via CFA (RSC + cognitive-diversity weights)."""
    names = list(preds.keys())
    means = np.array([preds[n] for n in names], dtype=np.float64)
    vol = per_minute_volatility(closes, 60)
    std = max(vol, _EPS)

    lo = float(means.min() - cfg.cfa_trunc_std * std)
    hi = float(means.max() + cfg.cfa_trunc_std * std)
    lo = max(0.0, lo)
    if hi - lo < _EPS:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, cfg.cfa_grid_size)

    scores = np.vstack([_normal_scores(grid, m, std, cfg.cfa_trunc_std) for m in means])
    # Rank-Score Characteristic functions = scores sorted high->low.
    rscs = np.vstack([np.sort(scores[i])[::-1] for i in range(len(names))])

    t = len(names)
    ds = np.zeros(t)
    if t > 1:
        for j in range(t):
            ds[j] = np.mean([
                math.sqrt(float(np.mean((rscs[j] - rscs[k]) ** 2)))
                for k in range(t) if k != j
            ])
    w = ds.copy() if ds.sum() > _EPS else np.ones(t)

    combined = (w[:, None] * scores).sum(axis=0) / max(w.sum(), _EPS)
    best = int(np.argmax(combined))
    return FusionResult(
        predicted_price=float(grid[best]),
        means={names[i]: float(means[i]) for i in range(t)},
        weights={names[i]: float(w[i]) for i in range(t)},
        diversity={names[i]: float(ds[i]) for i in range(t)},
        grid_low=lo, grid_high=hi,
    )


# ============================================================================
# SECTION 8 — NWACHUKWU FORECASTS (fast / slow rosters)
# ============================================================================

def _direction(delta: float) -> str:
    return "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"


@dataclass
class Forecast:
    label: str
    predicted_price: float
    delta: float
    direction: str
    fusion: FusionResult
    target_utc: Optional[str] = None
    minutes_ahead: Optional[int] = None


def nwachukwu_forecast(closes: np.ndarray, cfg: Config, spot: float,
                       roster: List[BaseForecaster],
                       volumes: Optional[np.ndarray] = None,
                       order_flow_r: Optional[float] = None,
                       label: str = "forecast") -> Forecast:
    """Run a roster + CFA fusion, feeding the live spot as the newest tick."""
    closes = np.asarray(closes, dtype=np.float64)
    series = np.append(closes, float(spot)) if spot and math.isfinite(spot) else closes
    vol_series = volumes
    if volumes is not None and len(series) > len(volumes):
        vol_series = np.append(volumes, volumes[-1])

    models = list(roster)
    if order_flow_r is not None and math.isfinite(order_flow_r):
        models = models + [OrderFlowForecaster()]

    preds = {
        m.name: m.predict(series, cfg, vol_series, order_flow_r or 0.0)
        for m in models
    }
    fusion = cfa_fuse(preds, series, cfg)
    delta = fusion.predicted_price - float(spot)
    return Forecast(label, fusion.predicted_price, delta, _direction(delta), fusion)


# Clock-alignment helpers -----------------------------------------------------

def next_quarter_hour(now: datetime) -> datetime:
    q = (now.minute // 15 + 1) * 15
    return now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=q)


def next_top_of_hour(now: datetime) -> datetime:
    return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


# ============================================================================
# SECTION 9 — PROJECTION FAN (bull / base / bear) + last-minute jump
# ============================================================================

@dataclass
class Projection:
    minutes: List[int]
    base: List[float]
    bull: List[float]
    bear: List[float]
    velocity_per_min: float
    jump_detected: bool
    anchors: List[Tuple[int, float]]


def build_projection(cfg: Config, spot: float, vol_1m: float,
                     anchors: List[Tuple[int, float]],
                     velocity_per_min: float) -> Projection:
    """Build a forward fan over proj_horizon_min minutes.

    base(m)  = piecewise-linear interpolation through model anchors, bent toward
               a detected last-minute jump for the first few minutes.
    band(m)  = proj_band_z * vol_1m * sqrt(m).
    bull/bear = base +/- band.
    """
    anchors = sorted(anchors, key=lambda a: a[0])

    def interp(m: float) -> float:
        for i in range(1, len(anchors)):
            if m <= anchors[i][0]:
                m0, p0 = anchors[i - 1]
                m1, p1 = anchors[i]
                return p0 + (p1 - p0) * (m - m0) / max(m1 - m0, 1e-9)
        return anchors[-1][1]

    jump = abs(velocity_per_min) > 2.0 * vol_1m and abs(velocity_per_min) > 1.0
    mins, base, bull, bear = [], [], [], []
    for m in range(1, cfg.proj_horizon_min + 1):
        center = interp(m) + velocity_per_min * min(m, cfg.proj_jump_minutes) * 0.5
        band = cfg.proj_band_z * vol_1m * math.sqrt(m)
        mins.append(m)
        base.append(center)
        bull.append(center + band)
        bear.append(center - band)
    return Projection(mins, base, bull, bear, velocity_per_min, jump, anchors)


def scenario(cfg: Config, base_price: float, minutes: int,
             vol_1m: float, velocity_per_min: float) -> Dict[str, float]:
    """Bull/base/bear prices at a given horizon."""
    band = (cfg.proj_band_z * vol_1m * math.sqrt(minutes)
            + abs(velocity_per_min) * min(minutes, cfg.proj_jump_minutes) * 0.5)
    return {"bull": base_price + band, "base": base_price, "bear": base_price - band}


# ============================================================================
# SECTION 10 — THE UMEH COMPOSITE (fair value + 0-100 score)
# ============================================================================

@dataclass
class UmehResult:
    spot: float
    as_of_utc: str
    order_flow_r: float
    # short term
    short_term: Forecast
    f15: Optional[Forecast]
    f60: Optional[Forecast]
    velocity_per_min: float
    vol_1m: float
    projection: Projection
    # long term
    power_law: Dict[str, float]
    stock_to_flow: Dict[str, float]
    top_cap: Dict[str, float]
    # composite
    umeh_fair_value: float
    pl_band_position: float
    pct_of_fair: float
    pct_to_top_cap: float
    umeh_score: float
    n_ticks_1m: int
    kalshi: Optional["KalshiImplied"] = None


def umeh_score(cfg: Config, spot: float, st_price: float, order_flow_r: float,
               pl: Dict[str, float], tc: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """0..100: cheap+buying => accumulate; expensive/near-ceiling+selling => distribute."""
    band = clamp((spot - pl["support"]) / max(pl["resistance"] - pl["support"], _EPS), 0.0, 1.0)
    valuation = 1.0 - band
    momentum = 0.5 + 0.5 * math.tanh((st_price - spot) / max(0.001 * spot, _EPS))
    flow = 0.5 + 0.5 * clamp(order_flow_r, -1.0, 1.0)
    pct_to_top = (tc["top_price"] - spot) / max(tc["top_price"], _EPS)
    ceiling = clamp(pct_to_top, 0.0, 1.0)
    score = 100.0 * (
        cfg.score_w_valuation * valuation
        + cfg.score_w_momentum * momentum
        + cfg.score_w_flow * flow
        + cfg.score_w_ceiling * ceiling
    )
    parts = {"valuation": valuation, "momentum": momentum, "flow": flow, "ceiling": ceiling}
    return clamp(score, 0.0, 100.0), parts


def compute_umeh(cfg: Config,
                 c1: Candles, c15: Candles, c60: Candles,
                 spot: float, order_flow_r: float,
                 velocity_per_min: float = 0.0,
                 now: Optional[datetime] = None,
                 kalshi: Optional["KalshiImplied"] = None) -> UmehResult:
    """Run the entire Umeh formula and assemble the result object."""
    now = now or datetime.now(timezone.utc)

    closes1 = c1.closes[-cfg.max_analyze * 3:]      # 1m can use more depth
    vols1 = c1.volumes[-len(closes1):]
    closes15 = c15.closes[-cfg.max_analyze:]
    closes60 = c60.closes[-cfg.max_analyze:]

    # --- short term (1-2 min): fast roster + order flow ---
    st = nwachukwu_forecast(closes1, cfg, spot, fast_roster(),
                            volumes=vols1, order_flow_r=order_flow_r,
                            label="Next 1-2 min (thrust)")

    # --- 15-minute mark ---
    f15 = None
    if len(closes15) > 40:
        f15 = nwachukwu_forecast(closes15, cfg, spot, slow_roster(),
                                 label="Next 15-min mark")
        f15.target_utc = next_quarter_hour(now).strftime("%H:%M UTC")

    # --- top of the hour (scale by fraction of hour remaining) ---
    f60 = None
    if len(closes60) > 40:
        raw = nwachukwu_forecast(closes60, cfg, spot, slow_roster(),
                                 label="Top of the hour")
        toh = next_top_of_hour(now)
        mins_left = max(1, int(round((toh - now).total_seconds() / 60)))
        scale = clamp(mins_left / 60.0, 0.0, 1.0)
        scaled_delta = raw.delta * scale
        f60 = Forecast("Top of the hour", spot + scaled_delta, scaled_delta,
                       _direction(scaled_delta), raw.fusion,
                       target_utc=toh.strftime("%H:%M UTC"), minutes_ahead=mins_left)

    # --- long-term anchors ---
    pl = power_law(cfg, now)
    s2f = stock_to_flow(cfg, now)
    tc = top_cap(cfg, now)

    # --- Umeh composite fair value (weighted geometric blend) ---
    anchors = {
        "market": (spot, cfg.umeh_w_market),
        "power_law": (pl["center"], cfg.umeh_w_power_law),
        "top_cap": (tc["top_price"], cfg.umeh_w_top_cap),
        "stock_to_flow": (s2f["model_price"], cfg.umeh_w_s2f),
    }
    wsum = sum(w for _, w in anchors.values())
    ln_fair = sum(w * math.log(max(v, _EPS)) for v, w in anchors.values()) / wsum
    umeh_fair = math.exp(ln_fair)

    band = clamp((spot - pl["support"]) / max(pl["resistance"] - pl["support"], _EPS), 0.0, 1.0)
    pct_of_fair = spot / max(pl["center"], _EPS)
    pct_to_top = (tc["top_price"] - spot) / max(tc["top_price"], _EPS)
    score, _parts = umeh_score(cfg, spot, st.predicted_price, order_flow_r, pl, tc)

    # --- projection fan ---
    vol_1m = per_minute_volatility(closes1, 30)
    proj_anchors: List[Tuple[int, float]] = [(0, spot), (2, st.predicted_price)]
    if f15 is not None:
        proj_anchors.append((15, f15.predicted_price))
    if f60 is not None:
        proj_anchors.append((60, f60.predicted_price))
    projection = build_projection(cfg, spot, vol_1m, proj_anchors, velocity_per_min)

    return UmehResult(
        spot=float(spot), as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        order_flow_r=float(order_flow_r),
        short_term=st, f15=f15, f60=f60,
        velocity_per_min=velocity_per_min, vol_1m=vol_1m, projection=projection,
        power_law=pl, stock_to_flow=s2f, top_cap=tc,
        umeh_fair_value=umeh_fair, pl_band_position=band,
        pct_of_fair=pct_of_fair, pct_to_top_cap=pct_to_top,
        umeh_score=score, n_ticks_1m=int(len(closes1)),
        kalshi=kalshi,
    )


# ============================================================================
# SECTION 11 — WALK-FORWARD BACKTESTER
# ============================================================================

@dataclass
class BacktestResult:
    interval: str
    n: int
    hit_rate: float
    up_recall: float
    up_total: int
    mean_abs_err: float
    mape: float
    baseline_hit: float


def backtest_roster(closes: np.ndarray, volumes: Optional[np.ndarray],
                    cfg: Config, roster_factory: Callable[[], List[BaseForecaster]],
                    interval: str, test_n: int = 60) -> BacktestResult:
    """Walk-forward: predict each next candle on unseen data, score direction."""
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    test_n = min(test_n, max(5, n // 5))
    start = n - test_n
    hits = tot = up_hits = up_tot = 0
    ae = 0.0
    roster = roster_factory()
    for i in range(start, n - 1):
        series = closes[:i + 1]
        vser = volumes[:i + 1] if volumes is not None else None
        spot = float(series[-1])
        preds = {m.name: m.predict(series, cfg, vser, 0.0) for m in roster}
        fused = cfa_fuse(preds, series, cfg)
        pred_delta = fused.predicted_price - spot
        actual = closes[i + 1] - closes[i]
        if abs(actual) < 1e-9:
            continue
        tot += 1
        if (pred_delta > 0) == (actual > 0):
            hits += 1
        if actual > 0:
            up_tot += 1
            if pred_delta > 0:
                up_hits += 1
        ae += abs(fused.predicted_price - closes[i + 1])
    mean_px = float(np.mean(closes[start:])) or 1.0
    return BacktestResult(
        interval=interval, n=tot,
        hit_rate=100.0 * hits / tot if tot else 0.0,
        up_recall=100.0 * up_hits / up_tot if up_tot else 0.0,
        up_total=up_tot,
        mean_abs_err=ae / tot if tot else 0.0,
        mape=100.0 * (ae / tot) / mean_px if tot else 0.0,
        baseline_hit=50.0,
    )


# ============================================================================
# SECTION 12 — REPORTING (human-readable + JSON)
# ============================================================================

def _bar(value01: float, width: int = 24) -> str:
    fill = int(round(clamp(value01, 0.0, 1.0) * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def render_report(cfg: Config, r: UmehResult) -> str:
    L: List[str] = []
    add = L.append
    add("=" * 70)
    add("  THE UMEH FORMULA  —  live report")
    add("  " + r.as_of_utc + "   (educational only — not financial advice)")
    add("=" * 70)
    add("")
    add(f"  SPOT BTC/USD : {fmt_usd(r.spot)}")
    add(f"  Order flow r : {r.order_flow_r:+.3f}  "
        f"({'buy pressure' if r.order_flow_r > 0.05 else 'sell pressure' if r.order_flow_r < -0.05 else 'balanced'})")
    add(f"  Last-60s vel : {r.velocity_per_min:+.2f}/min"
        + ("   ⚡ JUMP DETECTED" if r.projection.jump_detected else ""))
    add(f"  1m vol       : {fmt_usd(r.vol_1m)}/candle    ticks: {r.n_ticks_1m:,}")
    add("")
    add("-" * 70)
    add("  UMEH SCORE")
    add("-" * 70)
    lean = ("ACCUMULATE" if r.umeh_score >= 60 else
            "DISTRIBUTE" if r.umeh_score <= 40 else "NEUTRAL")
    add(f"  {r.umeh_score:5.1f} / 100   {_bar(r.umeh_score / 100.0)}   {lean}")
    add("  (cheap-vs-fair + short-term momentum + buy pressure + headroom)")
    add("")
    add("-" * 70)
    add("  MULTI-HORIZON FORECASTS (Nwachukwu)")
    add("-" * 70)
    for f in [r.short_term, r.f15, r.f60]:
        if f is None:
            continue
        tgt = f"  -> {f.target_utc}" if f.target_utc else ""
        add(f"  {f.label:<22s} {fmt_usd(f.predicted_price):>14s}  "
            f"{f.direction:<4s} ({f.delta:+.2f}){tgt}")
        ranked = sorted(f.fusion.weights, key=f.fusion.weights.get, reverse=True)
        models = ", ".join(f"{k} {fmt_usd(f.fusion.means[k],0)}" for k in ranked[:4])
        add(f"       models: {models}")
    add("")
    add("-" * 70)
    add("  PROJECTION FAN (bull / base / bear)")
    add("-" * 70)
    for m in (5, 15, 30, 60):
        sc = scenario(cfg, r.projection.base[m - 1], m, r.vol_1m, r.velocity_per_min)
        add(f"  +{m:>2d} min   bull {fmt_usd(sc['bull']):>13s}   "
            f"base {fmt_usd(sc['base']):>13s}   bear {fmt_usd(sc['bear']):>13s}")
    add("")
    add("-" * 70)
    add("  KALSHI 15-MIN  vs  BINANCE   (BTC up in next 15 min?)")
    add("-" * 70)
    k = r.kalshi
    if k is not None and k.ok:
        spot_vs_target = r.spot - k.target_price
        add(f"  Binance spot (now)        : {fmt_usd(r.spot)}")
        add(f"  Kalshi 15m target (beat)  : {fmt_usd(k.target_price)}  "
            f"(settles {k.settle_utc}, {k.minutes_to_settle} min out)")
        add(f"  Spot vs Kalshi target     : {spot_vs_target:+.2f}  "
            f"(BTC currently {'ABOVE' if spot_vs_target >= 0 else 'BELOW'} target)")
        if not math.isnan(k.prob_up):
            lean = "UP" if k.prob_up >= 0.5 else "DOWN"
            add(f"  Market-implied P(up)      : {100*k.prob_up:4.1f}%   (Kalshi leans {lean})")
        # Compare the Umeh 15-minute forecast against the same target.
        if r.f15 is not None:
            umeh15 = r.f15.predicted_price
            umeh_up = umeh15 > k.target_price
            kalshi_up = (not math.isnan(k.prob_up)) and k.prob_up >= 0.5
            agree = "AGREE" if (umeh_up == kalshi_up) else "DISAGREE"
            add(f"  Umeh 15m forecast         : {fmt_usd(umeh15)}  "
                f"({umeh15 - k.target_price:+.2f} vs target -> Umeh says {'UP' if umeh_up else 'DOWN'})")
            if not math.isnan(k.prob_up):
                add(f"  Umeh vs Kalshi direction  : {agree}")
    else:
        note = (k.note if k is not None else "not fetched")
        add(f"  Kalshi 15-min data unavailable ({note}).")
    add("")
    add("-" * 70)
    add("  LONG-TERM ANCHORS")
    add("-" * 70)
    pl, s2f, tc = r.power_law, r.stock_to_flow, r.top_cap
    add(f"  Power Law      center {fmt_usd(pl['center'],0):>12s}   "
        f"supp {fmt_usd(pl['support'],0)} / res {fmt_usd(pl['resistance'],0)}")
    add(f"                 day {pl['days']:.0f} since genesis")
    add(f"  Stock-to-Flow  model  {fmt_usd(s2f['model_price'],0):>12s}   "
        f"S2F {s2f['s2f_ratio']:.1f}  supply {s2f['supply']/1e6:.2f}M  reward {s2f['current_reward']}")
    add(f"  Top Cap        ceiling{fmt_usd(tc['top_price'],0):>12s}   "
        f"(avg price {fmt_usd(tc['average_price'],0)} x{cfg.top_cap_mult:.0f})")
    add(f"  UMEH FAIR      value  {fmt_usd(r.umeh_fair_value,0):>12s}   (weighted blend)")
    add("")
    add("-" * 70)
    add("  WHERE PRICE SITS")
    add("-" * 70)
    add(f"  Power-law band position : {100*r.pl_band_position:5.1f}%  {_bar(r.pl_band_position)} "
        f"(0=support, 100=resistance)")
    add(f"  Price vs power-law fair : {100*r.pct_of_fair:5.1f}% of center "
        f"({'undervalued' if r.pct_of_fair < 1 else 'rich'})")
    add(f"  Headroom to Top Cap     : {100*r.pct_to_top_cap:5.1f}%")
    add("")
    add("=" * 70)
    add("  EDUCATIONAL ONLY — NOT FINANCIAL ADVICE.")
    add("  Long-horizon models are contested and frequently wrong.")
    add("=" * 70)
    return "\n".join(L)


def umeh_to_dict(r: UmehResult) -> dict:
    def fc(f: Optional[Forecast]):
        if f is None:
            return None
        return {
            "label": f.label, "predicted_price": f.predicted_price,
            "delta": f.delta, "direction": f.direction,
            "target_utc": f.target_utc,
            "base_means": f.fusion.means, "weights": f.fusion.weights,
        }
    return {
        "spot": r.spot, "as_of_utc": r.as_of_utc, "order_flow_r": r.order_flow_r,
        "velocity_per_min": r.velocity_per_min, "vol_1m": r.vol_1m,
        "umeh_score": r.umeh_score, "umeh_fair_value": r.umeh_fair_value,
        "pl_band_position": r.pl_band_position, "pct_of_fair": r.pct_of_fair,
        "pct_to_top_cap": r.pct_to_top_cap,
        "short_term": fc(r.short_term), "f15": fc(r.f15), "f60": fc(r.f60),
        "power_law": r.power_law, "stock_to_flow": r.stock_to_flow, "top_cap": r.top_cap,
        "kalshi": (None if (r.kalshi is None or not r.kalshi.ok) else {
            "series": r.kalshi.series, "ticker": r.kalshi.ticker,
            "settle_utc": r.kalshi.settle_utc,
            "minutes_to_settle": r.kalshi.minutes_to_settle,
            "target_price": r.kalshi.target_price, "prob_up": r.kalshi.prob_up,
            "last_price": r.kalshi.last_price,
            "binance_spot": r.spot, "spot_vs_target": r.spot - r.kalshi.target_price,
            "umeh_15m": (r.f15.predicted_price if r.f15 is not None else None),
        }),
        "projection": {
            "minutes": r.projection.minutes, "base": r.projection.base,
            "bull": r.projection.bull, "bear": r.projection.bear,
            "jump_detected": r.projection.jump_detected,
        },
    }


# ============================================================================
# SECTION 13 — HIGH-LEVEL ORCHESTRATION
# ============================================================================

def load_all_data(cfg: Config):
    """Fetch 1m/15m/1h candles + live spot + order flow + Kalshi implied price."""
    c1 = fetch_candles("1m", cfg.ticks_1m, cfg)
    c15 = fetch_candles("15m", cfg.ticks_15m, cfg)
    c60 = fetch_candles("1h", cfg.ticks_1h, cfg)
    spot = fetch_spot(cfg)
    flow = fetch_order_flow(cfg)
    kalshi = fetch_kalshi_btc(cfg)
    return c1, c15, c60, spot, flow, kalshi


def run_predict(cfg: Config, as_json: bool = False) -> int:
    try:
        c1, c15, c60, spot, flow, kalshi = load_all_data(cfg)
    except DataError as exc:
        sys.stderr.write(f"Data error: {exc}\n")
        return 2
    result = compute_umeh(cfg, c1, c15, c60, spot, flow, kalshi=kalshi)
    if as_json:
        print(json.dumps(umeh_to_dict(result), indent=2))
    else:
        print(render_report(cfg, result))
    return 0


def run_monitor(cfg: Config, every: float) -> int:
    """Live loop: recompute every `every` seconds, tracking last-minute velocity."""
    spot_buf: List[Tuple[float, float]] = []
    print(f"Monitoring every {every:.0f}s. Ctrl-C to stop.\n")
    # Load slower candles once; refresh 1m tail each loop for speed.
    c1, c15, c60, spot, flow, kalshi = load_all_data(cfg)
    try:
        while True:
            now_ms = time.time()
            try:
                spot = fetch_spot(cfg)
                flow = fetch_order_flow(cfg)
                kalshi = fetch_kalshi_btc(cfg)
                tail = fetch_candles("1m", 3, cfg)
                # splice the fresh tail onto the cached 1m series
                for t, cl, vol in zip(tail.times, tail.closes, tail.volumes):
                    idx = np.where(c1.times == t)[0]
                    if idx.size:
                        c1.closes[idx[-1]] = cl
                        c1.volumes[idx[-1]] = vol
                    else:
                        c1.times = np.append(c1.times, t)
                        c1.closes = np.append(c1.closes, cl)
                        c1.volumes = np.append(c1.volumes, vol)
            except DataError as exc:
                sys.stderr.write(f"  (transient data error: {exc})\n")
                time.sleep(every)
                continue

            spot_buf.append((now_ms, spot))
            spot_buf = [x for x in spot_buf if now_ms - x[0] <= 120.0]
            vel = 0.0
            if len(spot_buf) >= 2:
                a, b = spot_buf[0], spot_buf[-1]
                dt_min = max((b[0] - a[0]) / 60.0, 1 / 60.0)
                vel = (b[1] - a[1]) / dt_min

            result = compute_umeh(cfg, c1, c15, c60, spot, flow,
                                  velocity_per_min=vel, kalshi=kalshi)
            print("\033[2J\033[H", end="")  # clear screen
            print(render_report(cfg, result))
            time.sleep(every)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def run_backtest(cfg: Config) -> int:
    try:
        c1 = fetch_candles("1m", 1000, cfg)
        c15 = fetch_candles("15m", 1000, cfg)
    except DataError as exc:
        sys.stderr.write(f"Data error: {exc}\n")
        return 2

    print("=" * 70)
    print("  WALK-FORWARD BACKTEST (directional accuracy on unseen candles)")
    print("=" * 70)

    rows = [
        backtest_roster(c1.closes, c1.volumes, cfg, slow_roster, "1m  price-only"),
        backtest_roster(c1.closes, c1.volumes, cfg, fast_roster, "1m  fast/thrust"),
        backtest_roster(c15.closes, c15.volumes, cfg, slow_roster, "15m price-only"),
        backtest_roster(c15.closes, c15.volumes, cfg, fast_roster, "15m fast/thrust"),
    ]
    print(f"\n  {'roster':<18s} {'N':>4s} {'dir%':>7s} {'UPrec%':>8s} "
          f"{'MAPE%':>7s} {'absErr':>10s}")
    print("  " + "-" * 60)
    for b in rows:
        print(f"  {b.interval:<18s} {b.n:>4d} {b.hit_rate:>7.1f} "
              f"{b.up_recall:>8.1f} {b.mape:>7.3f} {fmt_usd(b.mean_abs_err):>10s}")
    print("\n  (50% direction = coin flip. UPrec% = fraction of up-moves caught.)")
    print("  EDUCATIONAL ONLY — not financial advice.\n")
    return 0


def run_anchors(cfg: Config, as_json: bool = False) -> int:
    now = datetime.now(timezone.utc)
    pl = power_law(cfg, now)
    s2f = stock_to_flow(cfg, now)
    tc = top_cap(cfg, now)
    if as_json:
        print(json.dumps({"power_law": pl, "stock_to_flow": s2f, "top_cap": tc}, indent=2))
        return 0
    print("=" * 70)
    print("  LONG-TERM VALUATION ANCHORS  (deterministic, no live price)")
    print("=" * 70)
    print(f"  Power Law   : center {fmt_usd(pl['center'],0)}  "
          f"[supp {fmt_usd(pl['support'],0)} .. res {fmt_usd(pl['resistance'],0)}]")
    print(f"                day {pl['days']:.0f} since genesis, exponent n={cfg.pl_b}")
    print(f"  Supply      : {s2f['supply']/1e6:.3f}M BTC  "
          f"(block reward {s2f['current_reward']}, era {int(s2f['eras'])})")
    print(f"  Stock/Flow  : ratio {s2f['s2f_ratio']:.1f}  -> model {fmt_usd(s2f['model_price'],0)}")
    print(f"  Top Cap     : ceiling {fmt_usd(tc['top_price'],0)}  "
          f"(avg {fmt_usd(tc['average_price'],0)} x{cfg.top_cap_mult:.0f})")
    print("\n  EDUCATIONAL ONLY — these long-horizon fits are contested.\n")
    return 0


# ============================================================================
# SECTION 14 — COMMAND-LINE INTERFACE
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umeh_full.py",
        description="The Umeh formula — a deep, single-file BTC analysis engine "
                    "(educational only, not financial advice).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", nargs="?", default="predict",
                   choices=["predict", "monitor", "backtest", "anchors"],
                   help="what to run (default: predict)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    p.add_argument("--every", type=float, default=10.0,
                   help="monitor refresh interval in seconds (default 10)")
    p.add_argument("--ticks", type=int, default=None,
                   help="override how many 1-minute candles to analyze")
    p.add_argument("--version", action="version", version=f"umeh_full {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.ticks:
        cfg.ticks_1m = max(200, args.ticks)
    if args.command == "predict":
        return run_predict(cfg, as_json=args.json)
    if args.command == "monitor":
        return run_monitor(cfg, every=max(2.0, args.every))
    if args.command == "backtest":
        return run_backtest(cfg)
    if args.command == "anchors":
        return run_anchors(cfg, as_json=args.json)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
