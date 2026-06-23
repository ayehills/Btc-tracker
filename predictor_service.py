"""
predictor_service
=================

Live glue between the price feed and the Bayesian-regression model in
``btc_bayesian_predictor.py`` (a faithful port of Shah & Zhang, 2014,
"Bayesian Regression and Bitcoin").

On demand it:
  1. pulls the current BTC/USD spot price (live, every call),
  2. pulls the last 700 intraday candles (15-minute and 1-hour),
  3. fits the Nwachukwu (Bayesian + CFA) model on that history, and
  4. forecasts the price for the next 15-minute clock mark (:00/:15/:30/:45)
     and for the upcoming top of the hour.

Two-speed design so every tick is genuinely fed through the formula: the
expensive fit (k-means pattern library + weights) is cached and only rebuilt
when a NEW candle closes, while the cheap prediction step (RBF kernel weighting
+ CFA fusion) is re-run on every request with the live spot appended as the
most-recent tick. So each 10s poll re-computes the forecast on fresh data
rather than reusing a frozen delta.

Research and education only. Not financial advice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from btc_bayesian_predictor import Config, WindowPreset
from nwachukwu_model import NwachukwuModel, NwachukwuResult

KRAKEN_BASE = "https://api.kraken.com/0/public"
COINBASE_SPOT = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
_HEADERS = {"User-Agent": "btc-tracker/1.0 (+bayesian-forecast)"}

REQUEST_TIMEOUT = 20.0


# --------------------------------------------------------------------------
# Data feed
# --------------------------------------------------------------------------

def fetch_spot() -> float:
    """Current BTC/USD spot price. Tries Kraken, falls back to Coinbase."""
    try:
        r = requests.get(
            f"{KRAKEN_BASE}/Ticker",
            params={"pair": "XBTUSD"},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        result = r.json()["result"]
        # result is keyed by the canonical pair name (e.g. "XXBTZUSD").
        ticker = next(iter(result.values()))
        return float(ticker["c"][0])  # "c" = last trade close [price, lot]
    except Exception:
        r = requests.get(COINBASE_SPOT, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])


def fetch_candle_series(interval_min: int):
    """Ascending (timestamps, closes, volumes) for the given interval.

    Uses Kraken's public OHLC endpoint, which returns up to ~720 candles.
    """
    r = requests.get(
        f"{KRAKEN_BASE}/OHLC",
        params={"pair": "XBTUSD", "interval": interval_min},
        headers=_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC error: {payload['error']}")
    result = payload["result"]
    # The series lives under the pair key; "last" is also present, skip it.
    series = next(v for k, v in result.items() if k != "last")
    # Each row: [time, open, high, low, close, vwap, volume, count]
    times = np.array([int(row[0]) for row in series], dtype=np.int64)
    closes = np.array([float(row[4]) for row in series], dtype=np.float64)
    volumes = np.array([float(row[6]) for row in series], dtype=np.float64)
    return times, closes, volumes


def fetch_closes(interval_min: int) -> np.ndarray:
    """Ascending array of candle close prices for the given interval (minutes)."""
    return fetch_candle_series(interval_min)[1]


def fetch_order_flow() -> float:
    """Live order-book imbalance r = (bid_vol - ask_vol) / (bid_vol + ask_vol).

    Positive => more resting bid size than ask (buy pressure). Computed over the
    top 25 levels of Kraken's public order book. Returns 0.0 on any failure so
    the forecast degrades gracefully to price-only.
    """
    try:
        r = requests.get(
            f"{KRAKEN_BASE}/Depth",
            params={"pair": "XBTUSD", "count": 25},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        book = next(iter(r.json()["result"].values()))
        bid_vol = sum(float(x[1]) for x in book["bids"])
        ask_vol = sum(float(x[1]) for x in book["asks"])
        denom = bid_vol + ask_vol
        return float((bid_vol - ask_vol) / denom) if denom else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# Model fitting (per-candle) and prediction (per-tick)
# --------------------------------------------------------------------------
#
# Two-speed design so every tick actually flows through the formula:
#   * Fitting the pattern library + linear weights is the expensive step
#     (k-means clustering). It only needs to change when a *new candle closes*,
#     so it is cached and keyed on the newest candle timestamp.
#   * Prediction (RBF kernel weighting + the diverse base models + CFA fusion)
#     is cheap and is re-run on EVERY request, with the live spot appended as
#     the most-recent tick. So each 10s poll feeds the live price through the
#     full Nwachukwu formula rather than reusing a frozen delta.

MAX_TICKS = 700        # analyze the last 700 ticks, as requested
CANDLE_TTL = 12.0      # seconds; avoid hammering the candle API between polls


def _build_config() -> Config:
    """A config tuned for the shorter intraday series we feed the model."""
    cfg = Config(
        data_source="csv",                 # we supply prices directly
        window_preset=WindowPreset.DAILY_SHORT,  # windows of 7 / 14 / 30 candles
        n_clusters=60,
        n_selected=15,
        smoothing_c=0.25,
        weight_method="ols",
        random_state=0,
    )
    return cfg


# Short-lived raw-candle cache shared by fitting and the chart history.
_candle_cache: dict[int, tuple[float, tuple]] = {}
_candle_lock = threading.Lock()


def _get_candles(interval_min: int):
    """(times, closes, volumes) for an interval, cached briefly."""
    with _candle_lock:
        cached = _candle_cache.get(interval_min)
        if cached and (time.time() - cached[0]) < CANDLE_TTL:
            return cached[1]
        data = fetch_candle_series(interval_min)
        _candle_cache[interval_min] = (time.time(), data)
        return data


@dataclass
class _FitEntry:
    model: NwachukwuModel       # fitted pattern library + CFA base models
    closes: np.ndarray          # the (<=700) closed-candle closes it was fit on
    volumes: np.ndarray         # aligned candle volumes
    fit_candle_ts: int          # newest candle timestamp at fit time
    n_points: int


_fit_cache: dict[int, _FitEntry] = {}
_locks: dict[int, threading.Lock] = {}


def _lock_for(interval_min: int) -> threading.Lock:
    return _locks.setdefault(interval_min, threading.Lock())


def _get_fitted(interval_min: int) -> _FitEntry:
    """Return a model fit on the last 700 ticks, refitting when a candle closes."""
    with _lock_for(interval_min):
        times, closes, volumes = _get_candles(interval_min)
        times = times[-MAX_TICKS:]
        closes = closes[-MAX_TICKS:].astype(np.float64)
        volumes = volumes[-MAX_TICKS:].astype(np.float64)
        newest_ts = int(times[-1])

        entry = _fit_cache.get(interval_min)
        if entry is not None and entry.fit_candle_ts == newest_ts:
            entry.closes = closes  # same candle; keep the fitted model
            entry.volumes = volumes
            return entry

        cfg = _build_config()
        max_len = max(cfg.window_lengths)
        if len(closes) < max_len + cfg.n_clusters + 10:
            raise RuntimeError(
                f"not enough {interval_min}m candles ({len(closes)}) to fit"
            )

        # The 1-minute horizon uses the fast thrust roster (validated to catch
        # quick buy-side moves); slower horizons use the price-pattern roster.
        builder = NwachukwuModel.fast if interval_min <= 1 else NwachukwuModel.default
        model = builder(config=cfg).fit(closes, volumes)
        entry = _FitEntry(
            model=model,
            closes=closes,
            volumes=volumes,
            fit_candle_ts=newest_ts,
            n_points=len(closes),
        )
        _fit_cache[interval_min] = entry
        return entry


@dataclass
class _LivePrediction:
    result: NwachukwuResult
    n_points: int
    fit_candle_ts: int


def _predict_live(interval_min: int, spot: float,
                  order_flow_r: Optional[float] = None) -> _LivePrediction:
    """Run the Nwachukwu formula NOW, feeding the live spot as the newest tick.

    ``order_flow_r`` (live order-book imbalance) adds a buy/sell-pressure model
    to the fusion for this prediction only.
    """
    entry = _get_fitted(interval_min)
    series = entry.closes
    volumes = entry.volumes
    if spot and np.isfinite(spot):
        # Append the live price as the most-recent tick so the prediction
        # responds to live movement within the forming candle.
        series = np.append(entry.closes, float(spot))
        volumes = np.append(entry.volumes, entry.volumes[-1])
    result = entry.model.predict(series, volumes, order_flow_r=order_flow_r)
    return _LivePrediction(
        result=result,
        n_points=entry.n_points,
        fit_candle_ts=entry.fit_candle_ts,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

@dataclass
class Forecast:
    label: str
    target_time_utc: str
    minutes_ahead: int
    predicted_price: float
    delta: float
    direction: str
    history_points: int
    combination: str = ""             # CFA strategy used, e.g. "score/diversity"
    base_models: list = None          # [{name, price, diversity, weight}, ...]
    fit_candle_utc: str = ""          # candle the model was last fit on (provable)


@dataclass
class Analysis:
    ok: bool
    spot: float
    as_of_utc: str
    forecasts: list[Forecast]
    error: Optional[str] = None


def _direction(delta: float) -> str:
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    return "FLAT"


def _next_top_of_hour(now: datetime) -> datetime:
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return nxt


def _next_quarter_hour(now: datetime) -> datetime:
    """Next :00 / :15 / :30 / :45 clock boundary after ``now``."""
    q = (now.minute // 15 + 1) * 15
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=q)
    return nxt


def _base_breakdown(r: NwachukwuResult) -> list:
    rows = [
        {
            "name": name,
            "price": price,
            "diversity": r.diversity_strength.get(name, 0.0),
            "weight": r.weights.get(name, 0.0),
        }
        for name, price in r.base_means.items()
    ]
    # Show the most influential (highest-weight) systems first.
    rows.sort(key=lambda d: d["weight"], reverse=True)
    return rows


def get_analysis() -> Analysis:
    """Run the full live analysis: spot price + 15-minute and hourly forecasts."""
    now = datetime.now(timezone.utc)
    try:
        spot = fetch_spot()
        # Live order-book imbalance (buy/sell pressure) drives the fast models.
        flow_r = fetch_order_flow()
        flow_label = ("buy pressure" if flow_r > 0.05 else
                      "sell pressure" if flow_r < -0.05 else "balanced")

        def _fit_iso(ts: int) -> str:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")

        # --- Next 1-2 minutes: quick buy/sell thrust (1m model + order flow) ---
        m1 = _predict_live(1, spot, order_flow_r=flow_r)
        f1 = Forecast(
            label="Next 1–2 min (thrust)",
            target_time_utc=(now + timedelta(minutes=2)).strftime("%H:%M UTC"),
            minutes_ahead=2,
            predicted_price=m1.result.predicted_price,
            delta=m1.result.delta,
            direction=_direction(m1.result.delta),
            history_points=m1.n_points,
            combination=f"{m1.result.combination} · flow r={flow_r:+.2f} ({flow_label})",
            base_models=_base_breakdown(m1.result),
            fit_candle_utc=_fit_iso(m1.fit_candle_ts),
        )

        # --- Next 15-minute mark (:00/:15/:30/:45), 15m candle model ---
        # Re-run the formula NOW with the live spot fed in as the newest tick.
        # Order flow is reserved for the fast model (validated there, not here).
        m15 = _predict_live(15, spot)
        nxt_q = _next_quarter_hour(now)
        mins_to_q = max(1, int(round((nxt_q - now).total_seconds() / 60)))
        f15 = Forecast(
            label=f"Next 15-min mark ({nxt_q.strftime('%H:%M')})",
            target_time_utc=nxt_q.strftime("%H:%M UTC"),
            minutes_ahead=mins_to_q,
            predicted_price=m15.result.predicted_price,
            delta=m15.result.delta,
            direction=_direction(m15.result.delta),
            history_points=m15.n_points,
            combination=m15.result.combination,
            base_models=_base_breakdown(m15.result),
            fit_candle_utc=_fit_iso(m15.fit_candle_ts),
        )

        # --- Top-of-the-hour forecast (from 1h candle model) ---
        h1 = _predict_live(60, spot)
        toh = _next_top_of_hour(now)
        mins_to_hour = int(round((toh - now).total_seconds() / 60))
        # The hourly model predicts the next hourly close; scale the predicted
        # change by the fraction of the hour remaining so a forecast made at
        # :55 isn't treated the same as one made at :05.
        scale = max(0.0, min(1.0, mins_to_hour / 60.0))
        delta_hour = h1.result.delta * scale
        price_hour = spot + delta_hour
        fhour = Forecast(
            label="Top of the hour",
            target_time_utc=toh.strftime("%H:%M UTC"),
            minutes_ahead=mins_to_hour,
            predicted_price=price_hour,
            delta=delta_hour,
            direction=_direction(delta_hour),
            history_points=h1.n_points,
            combination=h1.result.combination,
            base_models=_base_breakdown(h1.result),
            fit_candle_utc=_fit_iso(h1.fit_candle_ts),
        )

        return Analysis(
            ok=True,
            spot=spot,
            as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            forecasts=[f1, f15, fhour],
        )
    except Exception as exc:  # surfaced on the page rather than crashing
        return Analysis(
            ok=False,
            spot=float("nan"),
            as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            forecasts=[],
            error=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------
# Live payload (price history for charting + spot + forecasts)
# --------------------------------------------------------------------------

def get_history(interval_min: int = 15, limit: int = 96) -> list:
    """Recent candle history as ``[{"t": iso, "ts": secs, "price": close}, ...]``."""
    times, closes, _ = _get_candles(interval_min)
    rows = [
        {
            "ts": int(t),
            "t": datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat(),
            "price": float(c),
        }
        for t, c in zip(times, closes)
    ]
    return rows[-limit:]


def get_live(history_interval: int = 15, history_limit: int = 96) -> dict:
    """One JSON-serializable bundle: spot, forecasts, and recent price history.

    This is what the live (auto-updating) front-end polls.
    """
    a = get_analysis()
    payload = {
        "ok": a.ok,
        "spot": a.spot,
        "as_of_utc": a.as_of_utc,
        "error": a.error,
        "forecasts": [
            {
                "label": f.label,
                "target_time_utc": f.target_time_utc,
                "minutes_ahead": f.minutes_ahead,
                "predicted_price": f.predicted_price,
                "delta": f.delta,
                "direction": f.direction,
                "history_points": f.history_points,
                "combination": f.combination,
                "base_models": f.base_models or [],
                "fit_candle_utc": f.fit_candle_utc,
            }
            for f in a.forecasts
        ],
        "history": [],
    }
    if a.ok:
        try:
            payload["history"] = get_history(history_interval, history_limit)
        except Exception as exc:
            payload["history_error"] = f"{type(exc).__name__}: {exc}"
    return payload
