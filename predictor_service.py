"""
predictor_service
=================

Live glue between the price feed and the Bayesian-regression model in
``btc_bayesian_predictor.py`` (a faithful port of Shah & Zhang, 2014,
"Bayesian Regression and Bitcoin").

On demand it:
  1. pulls the current BTC/USD spot price (live, every call),
  2. pulls recent intraday candles (15-minute and 1-hour),
  3. fits the Bayesian pattern model on that history, and
  4. forecasts the next-step price change, giving a price estimate for
     "+15 minutes from now" and for the upcoming "top of the hour".

The expensive step (fitting the pattern library + weights) is cached per
timeframe so that rapid page refreshes stay instant; only the live spot price
is re-fetched on every call. The cache is refreshed when a new candle closes
(or after ``MODEL_TTL`` seconds, whichever comes first).

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

# How long a fitted model is reused before being rebuilt (seconds). The model
# only changes meaningfully when a new candle closes, so this keeps refreshes
# snappy without going stale.
MODEL_TTL = 90.0
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


def fetch_closes(interval_min: int) -> np.ndarray:
    """Ascending array of candle close prices for the given interval (minutes).

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
    closes = np.array([float(row[4]) for row in series], dtype=np.float64)
    return closes


# --------------------------------------------------------------------------
# Model fitting (cached per timeframe)
# --------------------------------------------------------------------------

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


@dataclass
class _ModelCacheEntry:
    result: NwachukwuResult       # full CFA fusion result for this timeframe
    last_close: float
    delta: float                  # predicted next-candle price change
    n_points: int
    built_at: float


_model_cache: dict[int, _ModelCacheEntry] = {}
_locks: dict[int, threading.Lock] = {}


def _lock_for(interval_min: int) -> threading.Lock:
    return _locks.setdefault(interval_min, threading.Lock())


def _fit_model(interval_min: int) -> _ModelCacheEntry:
    """Fetch history, fit the Nwachukwu (Bayesian + CFA) model, predict delta."""
    closes = fetch_closes(interval_min)
    cfg = _build_config()
    max_len = max(cfg.window_lengths)

    if len(closes) < max_len + cfg.n_clusters + 10:
        raise RuntimeError(
            f"not enough {interval_min}m candles ({len(closes)}) to fit the model"
        )

    model = NwachukwuModel.default(config=cfg).fit(closes)
    result = model.predict(closes)

    return _ModelCacheEntry(
        result=result,
        last_close=float(closes[-1]),
        delta=float(result.delta),
        n_points=len(closes),
        built_at=time.time(),
    )


def _get_model(interval_min: int) -> _ModelCacheEntry:
    """Return a cached fitted model, rebuilding it when stale."""
    with _lock_for(interval_min):
        entry = _model_cache.get(interval_min)
        fresh = (
            entry is not None
            and (time.time() - entry.built_at) < MODEL_TTL
        )
        if not fresh:
            entry = _fit_model(interval_min)
            _model_cache[interval_min] = entry
        return entry


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


def _base_breakdown(entry: _ModelCacheEntry) -> list:
    r = entry.result
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

        # --- 15-minute-ahead forecast (from 15m candle model) ---
        m15 = _get_model(15)
        # Anchor the predicted change to the live spot so the number reflects
        # the price you actually see right now.
        price_15 = spot + m15.delta
        f15 = Forecast(
            label="In 15 minutes",
            target_time_utc=(now + timedelta(minutes=15)).strftime("%H:%M UTC"),
            minutes_ahead=15,
            predicted_price=price_15,
            delta=m15.delta,
            direction=_direction(m15.delta),
            history_points=m15.n_points,
            combination=m15.result.combination,
            base_models=_base_breakdown(m15),
        )

        # --- Top-of-the-hour forecast (from 1h candle model) ---
        h1 = _get_model(60)
        toh = _next_top_of_hour(now)
        mins_to_hour = int(round((toh - now).total_seconds() / 60))
        # The hourly model predicts the next hourly close; scale the predicted
        # change by the fraction of the hour remaining so a forecast made at
        # :55 isn't treated the same as one made at :05.
        scale = max(0.0, min(1.0, mins_to_hour / 60.0))
        delta_hour = h1.delta * scale
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
            base_models=_base_breakdown(h1),
        )

        return Analysis(
            ok=True,
            spot=spot,
            as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            forecasts=[f15, fhour],
        )
    except Exception as exc:  # surfaced on the page rather than crashing
        return Analysis(
            ok=False,
            spot=float("nan"),
            as_of_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            forecasts=[],
            error=f"{type(exc).__name__}: {exc}",
        )
