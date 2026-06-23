#!/usr/bin/env python3
"""
btc_bayesian (single-file build)
================================

A faithful Python port of the Bayesian-regression Bitcoin price-prediction
algorithm by Anvita Pandit (https://github.com/panditanvita/BTCpredictor),
implementing the method from Shah & Zhang (2014), "Bayesian Regression and
Bitcoin" (https://arxiv.org/abs/1410.1231).

This is the entire package compiled into one module: normalization, sample
entropy, the RBF Bayesian estimator, pattern mining, weight fitting, the model,
the backtester, the Bitcoin.com data layer, and the command-line interface.

Quickstart
----------
    pip install numpy scipy scikit-learn requests python-dotenv
    python btc_bayesian_predictor.py backtest --window-preset daily
    python btc_bayesian_predictor.py predict  --window-preset daily

Configuration is read from a .env file (or the process environment); every
value has a sensible default. See Config.from_env for the recognized keys.

Research and education only. Not financial advice. Backtested performance does
not imply live results.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at runtime
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

__version__ = "1.0.0"


# ==========================================================================
# normalize
# ==========================================================================

_EPS = 1e-12


def zscore(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Z-score ``a`` along ``axis``, returning zeros where std is ~0.

    Parameters
    ----------
    a:
        Input array.
    axis:
        Axis along which to standardize. ``None`` standardizes the flattened
        array; an integer standardizes each slice along that axis (e.g.
        ``axis=1`` standardizes each row independently).
    """
    a = np.asarray(a, dtype=np.float64)
    mean = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, keepdims=True)
    out = np.zeros_like(a)
    safe = std > _EPS
    # Broadcast-safe division only where std is non-zero.
    np.divide(a - mean, np.where(safe, std, 1.0), out=out)
    out = np.where(np.broadcast_to(safe, a.shape), out, 0.0)
    return out


# ==========================================================================
# sample_entropy
# ==========================================================================

DEFAULT_M: int = 2
DEFAULT_R_FACTOR: float = 0.2


def _phi(x: np.ndarray, m: int, r: float) -> float:
    """Average fraction of within-tolerance matches for embedding length ``m``.

    Mirrors the per-row normalization in the MATLAB source: each row count is
    divided by ``(n - m)`` and the row means are averaged over ``(n - m + 1)``
    template vectors.
    """
    n = x.shape[0]
    rows = n - m + 1
    if rows <= 1:
        return 0.0

    # Build the (rows, m) matrix of length-m templates.
    templates = np.lib.stride_tricks.sliding_window_view(x, m)  # (rows, m)

    # Pairwise Chebyshev distance, excluding the diagonal (self matches).
    # |templates[i] - templates[j]| max over the embedding axis.
    diff = np.abs(templates[:, None, :] - templates[None, :, :])  # (rows, rows, m)
    cheb = diff.max(axis=2)                                       # (rows, rows)
    np.fill_diagonal(cheb, np.inf)                               # drop self-pairs

    counts = (cheb < r).sum(axis=1).astype(np.float64)          # (rows,)
    counts /= (n - m)                                           # original norm
    return float(counts.sum() / rows)


def sample_entropy(
    x: np.ndarray,
    m: int = DEFAULT_M,
    r_factor: float = DEFAULT_R_FACTOR,
) -> float:
    """Compute Sample Entropy of a 1-D series.

    Parameters
    ----------
    x:
        1-D signal (a normalized cluster centroid in this pipeline).
    m:
        Embedding dimension (default 2, as in the original).
    r_factor:
        Tolerance as a multiple of the signal standard deviation
        (default 0.2, as in the original).

    Returns
    -------
    float
        Sample entropy. Returns ``0.0`` for degenerate inputs (constant
        signals, or counts that vanish) so downstream sorting stays finite -
        this is a robustness improvement over the raw MATLAB, which could
        emit ``NaN``/``Inf`` in those cases.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    if n < m + 2:
        return 0.0

    std = float(np.std(x))  # population std, matching np default
    if std == 0.0:
        return 0.0
    r = r_factor * std

    b = _phi(x, m, r)          # length-m matches
    a = _phi(x, m + 1, r)      # length-(m+1) matches

    if b <= 0.0 or a <= 0.0:
        return 0.0
    # log(B) - log(A) == -log(A / B) == SampEn(m, r)
    return float(np.log(b) - np.log(a))


# ==========================================================================
# bayesian
# ==========================================================================

# The original MATLAB writes exp(c * norm^2) with c = -1/4. We carry the sign
# explicitly here so SMOOTHING_C stays a positive, intuitive bandwidth term.
DEFAULT_SMOOTHING_C: float = 0.25


def bayesian_delta(
    x: np.ndarray,
    pattern_prefixes: np.ndarray,
    pattern_deltas: np.ndarray,
    c: float = DEFAULT_SMOOTHING_C,
) -> float:
    """Estimate the expected next-step price change for window ``x``.

    Parameters
    ----------
    x:
        1-D array, the current (normalized) price window of length ``L``.
    pattern_prefixes:
        2-D array of shape ``(K, L)`` - the normalized price prefixes of the
        ``K`` reference patterns. ``L`` must match ``len(x)``.
    pattern_deltas:
        1-D array of shape ``(K,)`` - the price change associated with each
        reference pattern (the un-normalized last column of each cluster
        centroid in the original algorithm).
    c:
        Positive RBF bandwidth constant. Larger ``c`` makes the estimator more
        local (sharper kernel).

    Returns
    -------
    float
        The kernel-weighted expected price change. Returns ``0.0`` if every
        kernel weight underflows to zero (degenerate case guarded in the
        original code).
    """
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(pattern_prefixes, dtype=np.float64)
    y = np.asarray(pattern_deltas, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError("pattern_prefixes must be 2-D (K, L)")
    if x.shape[0] != P.shape[1]:
        raise ValueError(
            f"window length {x.shape[0]} != pattern prefix length {P.shape[1]}"
        )
    if y.shape[0] != P.shape[0]:
        raise ValueError("pattern_deltas length must equal number of patterns")

    diff = P - x                                   # (K, L)
    sq_dist = np.einsum("ij,ij->i", diff, diff)    # (K,) squared L2 distances
    weights = np.exp(-c * sq_dist)                 # (K,) RBF similarities

    denom = float(weights.sum())
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(weights, y) / denom)


def bayesian_delta_batch(
    windows: np.ndarray,
    pattern_prefixes: np.ndarray,
    pattern_deltas: np.ndarray,
    c: float = DEFAULT_SMOOTHING_C,
) -> np.ndarray:
    """Vectorized estimate over many windows at once.

    Parameters
    ----------
    windows:
        2-D array of shape ``(N, L)`` of normalized price windows.
    pattern_prefixes, pattern_deltas, c:
        As in :func:`bayesian_delta`.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(N,)`` of expected price changes.
    """
    W = np.asarray(windows, dtype=np.float64)
    P = np.asarray(pattern_prefixes, dtype=np.float64)
    y = np.asarray(pattern_deltas, dtype=np.float64)

    if W.ndim != 2 or P.ndim != 2:
        raise ValueError("windows and pattern_prefixes must be 2-D")
    if W.shape[1] != P.shape[1]:
        raise ValueError("window length must match pattern prefix length")

    # ||w - p||^2 = ||w||^2 + ||p||^2 - 2 w.p   -> (N, K)
    w_sq = np.einsum("ij,ij->i", W, W)[:, None]    # (N, 1)
    p_sq = np.einsum("ij,ij->i", P, P)[None, :]    # (1, K)
    cross = W @ P.T                                # (N, K)
    sq_dist = w_sq + p_sq - 2.0 * cross
    np.clip(sq_dist, 0.0, None, out=sq_dist)       # guard tiny negatives

    weights = np.exp(-c * sq_dist)                 # (N, K)
    denom = weights.sum(axis=1)                    # (N,)
    num = weights @ y                              # (N,)

    out = np.zeros_like(denom)
    good = denom > 0.0
    out[good] = num[good] / denom[good]
    return out


# ==========================================================================
# patterns
# ==========================================================================

@dataclass
class PatternSet:
    """Selected patterns for a single timescale."""

    length: int
    prefixes: np.ndarray  # (n_selected, length) z-scored
    deltas: np.ndarray    # (n_selected,) price change per pattern

    def __post_init__(self) -> None:
        if self.prefixes.shape[0] != self.deltas.shape[0]:
            raise ValueError("prefixes and deltas count mismatch")
        if self.prefixes.shape[1] != self.length:
            raise ValueError("prefix width must equal length")


@dataclass
class PatternLibrary:
    """Container of :class:`PatternSet` keyed by window length."""

    sets: Dict[int, PatternSet] = field(default_factory=dict)

    @property
    def lengths(self) -> Sequence[int]:
        return sorted(self.sets.keys())

    def __getitem__(self, length: int) -> PatternSet:
        return self.sets[length]

    @classmethod
    def build(
        cls,
        prices: np.ndarray,
        window_lengths: Sequence[int],
        n_clusters: int = 100,
        n_selected: int = 20,
        interval_jump: int = 1,
        random_state: int = 0,
        entropy_slice: int | None = None,
    ) -> "PatternLibrary":
        """Construct a pattern library from a 1-D training price series.

        Parameters
        ----------
        prices:
            1-D training prices (the first partition of the dataset).
        window_lengths:
            The timescales to model, e.g. ``(180, 360, 720)`` for the original
            high-frequency setup or ``(30, 60, 120)`` for daily data.
        n_clusters:
            k-means cluster count per timescale (original: 100).
        n_selected:
            Number of top-entropy patterns to keep per timescale (original: 20).
        interval_jump:
            Stride between consecutive training windows (original: 1).
        random_state:
            Seed for reproducible k-means.
        entropy_slice:
            If given, score entropy on only the first ``entropy_slice`` samples
            of each centroid. The original MATLAB scored all three timescales on
            their first 180 points (a documented quirk that, per the author,
            happened to perform well). Pass ``180`` to reproduce that exactly;
            leave ``None`` to score each centroid over its full length.

        Returns
        -------
        PatternLibrary
        """
        prices = np.asarray(prices, dtype=np.float64).ravel()
        max_len = max(window_lengths)
        price_diff = np.diff(prices)  # length len(prices) - 1

        # Need room for the largest window plus its next-step delta.
        n_windows = len(prices) - max_len - 1
        if n_windows < n_clusters:
            raise ValueError(
                f"Not enough training prices: {len(prices)} points yield only "
                f"{max(n_windows, 0)} windows for max window {max_len}, but "
                f"{n_clusters} clusters were requested. Use a longer series, "
                f"shorter windows, or fewer clusters."
            )

        sets: Dict[int, PatternSet] = {}
        for L in window_lengths:
            # Build (n_windows, L + 1): each row is a price window plus the
            # price change immediately following that window.
            windowed = np.lib.stride_tricks.sliding_window_view(prices, L)
            idx = np.arange(0, n_windows, interval_jump)
            prefix = windowed[idx]                       # (m, L)
            following = price_diff[idx + (L - 1)]        # next-step change
            samples = np.column_stack([prefix, following])  # (m, L + 1)

            km = KMeans(
                n_clusters=n_clusters,
                n_init=4,           # original used 4 replicates
                max_iter=10000,
                random_state=random_state,
            )
            km.fit(samples)
            centroids = km.cluster_centers_              # (n_clusters, L + 1)

            # Normalize the price prefix only; keep the delta column raw.
            norm_prefix = zscore(centroids[:, :L], axis=1)  # (n_clusters, L)
            deltas = centroids[:, L]                         # (n_clusters,)

            # Rank by sample entropy and keep the most informative patterns.
            if entropy_slice is not None:
                score_input = norm_prefix[:, :entropy_slice]
            else:
                score_input = norm_prefix
            scores = np.array([sample_entropy(row) for row in score_input])
            keep = np.argsort(scores)[::-1][:n_selected]

            sets[L] = PatternSet(
                length=L,
                prefixes=norm_prefix[keep],
                deltas=deltas[keep],
            )

        return cls(sets=sets)


# ==========================================================================
# weights
# ==========================================================================

@dataclass
class WeightModel:
    """Learned linear weights for combining Bayesian features."""

    theta: np.ndarray  # (n_features,)
    theta0: float      # intercept

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict price change(s) from a feature matrix ``(N, n_features)``."""
        features = np.asarray(features, dtype=np.float64)
        return features @ self.theta + self.theta0


def _fit_ols(X: np.ndarray, y: np.ndarray) -> WeightModel:
    n = X.shape[0]
    design = np.column_stack([X, np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return WeightModel(theta=coef[:-1], theta0=float(coef[-1]))


def _fit_de(
    X: np.ndarray,
    y: np.ndarray,
    bound: float,
    seed: int,
    maxiter: int,
) -> WeightModel:
    from scipy.optimize import differential_evolution

    n_features = X.shape[1]

    def cost(params: np.ndarray) -> float:
        theta = params[:n_features]
        theta0 = params[n_features]
        return float(np.linalg.norm(y - (X @ theta + theta0)))

    bounds = [(-bound, bound)] * (n_features + 1)
    result = differential_evolution(
        cost,
        bounds,
        seed=seed,
        maxiter=maxiter,
        polish=True,
        tol=1e-8,
    )
    params = result.x
    return WeightModel(theta=params[:n_features], theta0=float(params[n_features]))


def fit_weights(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "ols",
    de_bound: float = 5.0,
    de_seed: int = 0,
    de_maxiter: int = 200,
) -> WeightModel:
    """Fit linear combination weights.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(N, n_features)`` (the per-timescale Bayesian
        estimates, plus ``r`` if used).
    y:
        Target vector of realized next-step price changes, shape ``(N,)``.
    method:
        ``"ols"`` (default, closed form) or ``"de"`` (Differential Evolution,
        matching the original repository).
    de_bound, de_seed, de_maxiter:
        Differential Evolution settings (ignored for OLS).

    Returns
    -------
    WeightModel
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D (N, n_features)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[0] == 0:
        raise ValueError("no training rows provided to fit_weights")

    method = method.lower()
    if method == "ols":
        return _fit_ols(X, y)
    if method == "de":
        return _fit_de(X, y, de_bound, de_seed, de_maxiter)
    raise ValueError(f"unknown method {method!r}; use 'ols' or 'de'")


# ==========================================================================
# config
# ==========================================================================

# Two ready-made timescale presets. The high-frequency preset reproduces the
# original paper/repository (windows in 10-second steps). The daily preset is
# scaled for the coarser data returned by the Bitcoin.com price index.
WINDOW_PRESETS: dict[str, Tuple[int, ...]] = {
    "hf": (180, 360, 720),     # original: 180/360/720 * 10s
    "daily": (30, 60, 120),    # 30/60/120 daily candles
    "daily_short": (7, 14, 30),
}


class WindowPreset:
    HF = "hf"
    DAILY = "daily"
    DAILY_SHORT = "daily_short"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


@dataclass
class Config:
    """Runtime configuration for the predictor and backtester."""

    # ---- Data source -------------------------------------------------------
    # "bitcoin_com" pulls the BCX price index; "csv" loads a local file in the
    # original okcoin/coinbase column layout (col2=price, col3=ask, col4=bid).
    data_source: str = "bitcoin_com"
    csv_path: str = "data/prices.csv"
    downsample: int = 1  # keep every Nth row (original used 2 to go 5s -> 10s)

    # Bitcoin.com price data (consumed by the data layer; no API key needed).
    # Primary: the current charts API. Fallback: the legacy index API.
    charts_base_url: str = "https://charts.bitcoin.com/api/v1"
    charts_id: str = "rainbow"  # carries the full daily price history
    index_base_url: str = "https://index-api.bitcoin.com/api"
    index_version: str = "v0"
    track_currency: str = "USD"
    history_span: str = "all"  # "all" / "max" / e.g. "1y"

    # ---- Model -------------------------------------------------------------
    window_preset: str = WindowPreset.DAILY
    window_lengths: Tuple[int, ...] = field(default_factory=tuple)
    n_clusters: int = 100
    n_selected: int = 20
    smoothing_c: float = 0.25
    use_order_book_r: bool = False  # only possible with bid/ask volume (CSV)
    entropy_slice: int | None = None  # set 180 to reproduce the original quirk
    random_state: int = 0

    # ---- Weight fitting ----------------------------------------------------
    weight_method: str = "ols"  # "ols" or "de"
    de_bound: float = 5.0
    de_maxiter: int = 200

    # ---- Data split (fractions of the series) ------------------------------
    split_patterns: float = 0.34  # part 1 -> pattern library
    split_train: float = 0.33     # part 2 -> weight fitting
    # remainder -> part 3 (backtest)

    # ---- Trading -----------------------------------------------------------
    fee_buy: float = 0.001
    fee_sell: float = 0.003
    # If True, fees are read as fractions of price * the configured percent,
    # matching the original "fee != 0" branch. If False, the small absolute
    # thresholds above are used directly.
    fee_as_percent: bool = False
    fee_percent: float = 1.0

    def __post_init__(self) -> None:
        if not self.window_lengths:
            preset = WINDOW_PRESETS.get(self.window_preset)
            if preset is None:
                raise ValueError(
                    f"unknown window_preset {self.window_preset!r}; "
                    f"choose from {sorted(WINDOW_PRESETS)} or set window_lengths"
                )
            self.window_lengths = preset

    @property
    def n_features(self) -> int:
        base = len(self.window_lengths)
        return base + (1 if self.use_order_book_r else 0)

    @classmethod
    def from_env(cls, dotenv_path: str | None = ".env") -> "Config":
        """Build a Config from environment variables (loading ``.env`` first)."""
        if dotenv_path:
            load_dotenv(dotenv_path)

        preset = _get_str("WINDOW_PRESET", WindowPreset.DAILY)
        # Allow an explicit override like WINDOW_LENGTHS="30,60,120".
        raw_lengths = os.getenv("WINDOW_LENGTHS")
        window_lengths: Tuple[int, ...] = ()
        if raw_lengths:
            window_lengths = tuple(
                int(p) for p in raw_lengths.replace(" ", "").split(",") if p
            )

        slice_raw = os.getenv("ENTROPY_SLICE")
        entropy_slice = int(slice_raw) if slice_raw not in (None, "") else None

        return cls(
            data_source=_get_str("DATA_SOURCE", "bitcoin_com"),
            csv_path=_get_str("CSV_PATH", "data/prices.csv"),
            downsample=_get_int("DOWNSAMPLE", 1),
            charts_base_url=_get_str(
                "BITCOIN_COM_CHARTS_BASE_URL", "https://charts.bitcoin.com/api/v1"
            ),
            charts_id=_get_str("BITCOIN_COM_CHARTS_ID", "rainbow"),
            index_base_url=_get_str(
                "BITCOIN_COM_INDEX_BASE_URL", "https://index-api.bitcoin.com/api"
            ),
            index_version=_get_str("BITCOIN_COM_INDEX_VERSION", "v0"),
            track_currency=_get_str("TRACK_CURRENCY", "USD"),
            history_span=_get_str("HISTORY_SPAN", "all"),
            window_preset=preset,
            window_lengths=window_lengths,
            n_clusters=_get_int("N_CLUSTERS", 100),
            n_selected=_get_int("N_SELECTED", 20),
            smoothing_c=_get_float("SMOOTHING_C", 0.25),
            use_order_book_r=_get_bool("USE_ORDER_BOOK_R", False),
            entropy_slice=entropy_slice,
            random_state=_get_int("RANDOM_STATE", 0),
            weight_method=_get_str("WEIGHT_METHOD", "ols"),
            de_bound=_get_float("DE_BOUND", 5.0),
            de_maxiter=_get_int("DE_MAXITER", 200),
            split_patterns=_get_float("SPLIT_PATTERNS", 0.34),
            split_train=_get_float("SPLIT_TRAIN", 0.33),
            fee_buy=_get_float("FEE_BUY", 0.001),
            fee_sell=_get_float("FEE_SELL", 0.003),
            fee_as_percent=_get_bool("FEE_AS_PERCENT", False),
            fee_percent=_get_float("FEE_PERCENT", 1.0),
        )


# ==========================================================================
# model
# ==========================================================================

@dataclass
class BayesianRegressionModel:
    config: Config
    library: Optional[PatternLibrary] = None
    weights: Optional[WeightModel] = None

    # -- Feature construction -------------------------------------------------
    def _features_for_index(
        self,
        prices: np.ndarray,
        t: int,
        r_value: float | None = None,
    ) -> np.ndarray:
        """Bayesian features for the window ending at index ``t`` (inclusive)."""
        feats = []
        for L in self.config.window_lengths:
            window = zscore(prices[t - L + 1 : t + 1])
            ps = self.library[L]
            feats.append(bayesian_delta(window, ps.prefixes, ps.deltas, self.config.smoothing_c))
        if self.config.use_order_book_r:
            feats.append(0.0 if r_value is None else float(r_value))
        return np.asarray(feats, dtype=np.float64)

    def _feature_matrix(
        self,
        prices: np.ndarray,
        start: int,
        end: int,
        r_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized features for every index in ``[start, end)``."""
        idx = np.arange(start, end)
        cols = []
        for L in self.config.window_lengths:
            # Build (len(idx), L) matrix of raw windows, then z-score each row.
            starts = idx - L + 1
            offsets = np.arange(L)
            raw = prices[starts[:, None] + offsets[None, :]]  # (N, L)
            norm = zscore(raw, axis=1)
            ps = self.library[L]
            cols.append(
                bayesian_delta_batch(norm, ps.prefixes, ps.deltas, self.config.smoothing_c)
            )
        feat = np.column_stack(cols)
        if self.config.use_order_book_r:
            r_col = (
                np.zeros(len(idx)) if r_values is None else np.asarray(r_values)[idx]
            )
            feat = np.column_stack([feat, r_col])
        return feat

    # -- Fitting --------------------------------------------------------------
    def fit(
        self,
        prices_part1: np.ndarray,
        prices_part2: np.ndarray,
        r_values_part2: np.ndarray | None = None,
    ) -> "BayesianRegressionModel":
        """Build the pattern library and learn the combination weights.

        Parameters
        ----------
        prices_part1:
            Training prices used to mine and cluster patterns.
        prices_part2:
            Prices used to compute Bayesian features and fit weights.
        r_values_part2:
            Optional order-book imbalance aligned with ``prices_part2`` (only
            used when ``config.use_order_book_r`` is True).
        """
        cfg = self.config
        prices_part1 = np.asarray(prices_part1, dtype=np.float64).ravel()
        prices_part2 = np.asarray(prices_part2, dtype=np.float64).ravel()

        self.library = PatternLibrary.build(
            prices_part1,
            window_lengths=cfg.window_lengths,
            n_clusters=cfg.n_clusters,
            n_selected=cfg.n_selected,
            random_state=cfg.random_state,
            entropy_slice=cfg.entropy_slice,
        )

        max_len = max(cfg.window_lengths)
        start = max_len
        end = len(prices_part2) - 1  # need t+1 for the target
        if end <= start:
            raise ValueError(
                "prices_part2 is too short for the chosen window lengths "
                f"(need > {max_len + 1} points, got {len(prices_part2)})"
            )

        X = self._feature_matrix(prices_part2, start, end, r_values_part2)
        # Target: realized next-step change for each window end.
        idx = np.arange(start, end)
        y = prices_part2[idx + 1] - prices_part2[idx]

        self.weights = fit_weights(
            X,
            y,
            method=cfg.weight_method,
            de_bound=cfg.de_bound,
            de_seed=cfg.random_state,
            de_maxiter=cfg.de_maxiter,
        )
        return self

    # -- Prediction -----------------------------------------------------------
    def predict_delta(
        self,
        prices: np.ndarray,
        t: int | None = None,
        r_value: float | None = None,
    ) -> float:
        """Predict the next-step price change for the window ending at ``t``.

        If ``t`` is None, uses the last index of ``prices``.
        """
        if self.library is None or self.weights is None:
            raise RuntimeError("model must be fit() before predict_delta()")
        prices = np.asarray(prices, dtype=np.float64).ravel()
        if t is None:
            t = len(prices) - 1
        feats = self._features_for_index(prices, t, r_value)
        return float(self.weights.predict(feats[None, :])[0])

    def is_fitted(self) -> bool:
        return self.library is not None and self.weights is not None


# ==========================================================================
# backtest
# ==========================================================================

@dataclass
class BacktestResult:
    """Summary of a single backtest run."""

    pnl: float                      # realized profit in price units
    return_pct: float               # PnL relative to first traded price
    n_trades: int                   # completed round-trips
    win_rate: float                 # % of round-trips that were profitable
    mean_abs_error: float           # mean |actual - predicted| per step
    buys: List[int] = field(default_factory=list)   # indices of buys
    sells: List[int] = field(default_factory=list)  # indices of sells
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            f"  PnL (price units) : {self.pnl:,.2f}",
            f"  Return            : {self.return_pct:.2f}%",
            f"  Round-trip trades : {self.n_trades}",
            f"  Win rate          : {self.win_rate:.1f}%",
            f"  Mean abs error    : {self.mean_abs_error:.4f}",
        ]
        return "\n".join(lines)


def backtest(
    model: BayesianRegressionModel,
    prices: np.ndarray,
    config: Config,
    r_values: Optional[np.ndarray] = None,
) -> BacktestResult:
    """Run the threshold trading strategy over a held-out price series.

    Parameters
    ----------
    model:
        A fitted :class:`BayesianRegressionModel`.
    prices:
        1-D held-out prices (the final partition of the dataset).
    config:
        Run configuration (thresholds, window lengths, etc.).
    r_values:
        Optional order-book imbalance aligned with ``prices``.

    Returns
    -------
    BacktestResult
    """
    if not model.is_fitted():
        raise RuntimeError("model must be fit() before backtesting")

    prices = np.asarray(prices, dtype=np.float64).ravel()
    max_len = max(config.window_lengths)
    n = len(prices)
    if n <= max_len + 1:
        raise ValueError(
            f"held-out series too short ({n} points) for window {max_len}"
        )

    # Resolve buy/sell thresholds (mirrors the fee branches in brtrade.m).
    if config.fee_as_percent:
        thr = config.fee_percent * prices[min(1, n - 1)] / 100.0
        buy_threshold = sell_threshold = thr
    else:
        buy_threshold = config.fee_buy
        sell_threshold = config.fee_sell

    position = 0          # 0 = flat, 1 = holding one unit
    entry_price = 0.0
    bank = 0.0
    abs_error_sum = 0.0
    steps = 0
    wins = 0
    completed = 0
    buys: List[int] = []
    sells: List[int] = []
    equity = np.zeros(n)
    first_traded_price = None

    # Vectorize the (expensive) feature/prediction step across the whole range,
    # then run the stateful trading loop over the precomputed predictions.
    start = max_len
    end = n - 1
    preds = np.empty(end - start)
    feat_matrix = model._feature_matrix(prices, start, end, r_values)
    preds = model.weights.predict(feat_matrix)  # type: ignore[union-attr]

    for offset, t in enumerate(range(start, end)):
        dp = float(preds[offset])
        actual = prices[t + 1] - prices[t]
        abs_error_sum += abs(actual - dp)
        steps += 1

        if dp > buy_threshold and position == 0:
            position = 1
            entry_price = prices[t]
            if first_traded_price is None:
                first_traded_price = entry_price
            buys.append(t)
        elif dp < -sell_threshold and position == 1:
            position = 0
            bank += prices[t] - entry_price
            completed += 1
            if prices[t] - entry_price > 0:
                wins += 1
            sells.append(t)

        equity[t] = bank

    # Force-close any open position at the final price.
    if position == 1:
        last = end
        bank += prices[last] - entry_price
        completed += 1
        if prices[last] - entry_price > 0:
            wins += 1
        sells.append(last)
        equity[last] = bank

    equity[end:] = bank
    win_rate = (wins / completed * 100.0) if completed else 0.0
    mean_abs_error = (abs_error_sum / steps) if steps else 0.0
    base = first_traded_price if first_traded_price else prices[start]
    return_pct = (bank / base * 100.0) if base else 0.0

    return BacktestResult(
        pnl=bank,
        return_pct=return_pct,
        n_trades=completed,
        win_rate=win_rate,
        mean_abs_error=mean_abs_error,
        buys=buys,
        sells=sells,
        equity_curve=equity,
    )


# ==========================================================================
# data
# ==========================================================================

@dataclass
class PriceData:
    """A loaded price series and any auxiliary order-book volumes."""

    prices: np.ndarray
    ask_volume: Optional[np.ndarray] = None
    bid_volume: Optional[np.ndarray] = None

    @property
    def order_book_r(self) -> Optional[np.ndarray]:
        """Order-book imbalance r = (bid - ask) / (bid + ask), if available."""
        if self.bid_volume is None or self.ask_volume is None:
            return None
        denom = self.bid_volume + self.ask_volume
        r = np.zeros_like(denom, dtype=np.float64)
        good = denom != 0
        r[good] = (self.bid_volume[good] - self.ask_volume[good]) / denom[good]
        return r

    def downsampled(self, step: int) -> "PriceData":
        if step <= 1:
            return self
        return PriceData(
            prices=self.prices[::step],
            ask_volume=None if self.ask_volume is None else self.ask_volume[::step],
            bid_volume=None if self.bid_volume is None else self.bid_volume[::step],
        )


def load_csv(path: str) -> PriceData:
    """Load prices (and optional volumes) from a CSV in the original layout."""
    raw = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        # Single column -> treat as prices directly.
        return PriceData(prices=raw.astype(np.float64))
    prices = raw[:, 1].astype(np.float64)
    ask = raw[:, 2].astype(np.float64) if raw.shape[1] > 2 else None
    bid = raw[:, 3].astype(np.float64) if raw.shape[1] > 3 else None
    return PriceData(prices=prices, ask_volume=ask, bid_volume=bid)


def fetch_bitcoin_com_charts_price(
    base_url: str = "https://charts.bitcoin.com/api/v1",
    chart_id: str = "rainbow",
    timespan: str = "all",
    interval: str = "daily",
    timeout: float = 30.0,
) -> PriceData:
    """Fetch the daily BTC price series from the Bitcoin.com charts API.

    This is the current, public, key-less Bitcoin.com price endpoint. Several
    of its charts embed the same underlying daily close series under
    ``data.price`` (``rainbow`` carries the full history - ~5,800 points back
    to 2010 with ``timespan="all"``). Returns prices in ascending time order.
    """
    import requests

    url = f"{base_url.rstrip('/')}/charts/{chart_id}"
    params = {"interval": interval, "timespan": timespan}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", {})
    series = data.get("price")
    if not series:
        # Fall back to chart variants that nest price elsewhere (e.g. mayer).
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict) \
                    and "price" in val[0]:
                series = val
                break
    if not series:
        raise RuntimeError(
            f"no price series found in charts endpoint for chart '{chart_id}'"
        )

    series_sorted = sorted(series, key=lambda p: p["timestamp"])
    prices = np.array([float(p["price"]) for p in series_sorted], dtype=np.float64)
    return PriceData(prices=prices)


def fetch_bitcoin_com_history(
    base_url: str = "https://index-api.bitcoin.com/api",
    version: str = "v0",
    span: str = "all",
    timeout: float = 30.0,
) -> PriceData:
    """Fetch the BCX daily price history from the legacy Bitcoin.com index API.

    NOTE: as of this writing the ``index-api.bitcoin.com`` host returns 503;
    :func:`fetch_bitcoin_com_charts_price` is the working primary source and
    this is retained as a documented fallback. The endpoint returns an array
    of ``[iso_timestamp, price]`` pairs in reverse-chronological order; we sort
    ascending. Pass ``span="all"`` for the full history or e.g. ``"6m"``.
    """
    import requests

    url = f"{base_url.rstrip('/')}/{version}/history"
    params = {"span": span} if span else None
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError("Bitcoin.com history endpoint returned no data")

    rows_sorted = sorted(rows, key=lambda r: r[0])
    prices = np.array([float(r[1]) for r in rows_sorted], dtype=np.float64)
    return PriceData(prices=prices)


def fetch_bitcoin_com_spot(
    currency: str = "USD",
    base_url: str = "https://index-api.bitcoin.com/api",
    version: str = "v0",
    timeout: float = 15.0,
) -> float:
    """Fetch the current BCX spot price for ``currency`` (live use)."""
    import requests

    url = f"{base_url.rstrip('/')}/{version}/price/{currency}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    # The endpoint reports the spot price; accept a couple of shapes defensively.
    if isinstance(payload, dict):
        for key in ("price", "spot", "value"):
            if key in payload:
                return float(payload[key])
    return float(payload)


def load_price_data(config: Config) -> PriceData:
    """Load a :class:`PriceData` according to ``config.data_source``."""
    if config.data_source == "csv":
        data = load_csv(config.csv_path)
    elif config.data_source == "bitcoin_com":
        # Primary: current charts API. Fallback: legacy index API.
        try:
            data = fetch_bitcoin_com_charts_price(
                base_url=config.charts_base_url,
                chart_id=config.charts_id,
                timespan=config.history_span,
            )
        except Exception as exc:  # pragma: no cover - network-dependent
            try:
                data = fetch_bitcoin_com_history(
                    base_url=config.index_base_url,
                    version=config.index_version,
                    span=config.history_span,
                )
            except Exception as exc2:
                raise RuntimeError(
                    "Both Bitcoin.com price endpoints failed. "
                    f"charts API: {exc}; legacy index API: {exc2}"
                ) from exc2
    else:
        raise ValueError(
            f"unknown data_source {config.data_source!r}; "
            f"use 'csv' or 'bitcoin_com'"
        )
    return data.downsampled(config.downsample)


def split_three(
    data: PriceData,
    frac1: float,
    frac2: float,
) -> Tuple[PriceData, PriceData, PriceData]:
    """Split a price series into the three partitions the algorithm needs."""
    n = len(data.prices)
    b1 = int(n * frac1)
    b2 = b1 + int(n * frac2)
    if b1 <= 0 or b2 >= n or b2 <= b1:
        raise ValueError(
            f"invalid split: n={n}, frac1={frac1}, frac2={frac2} -> "
            f"boundaries ({b1}, {b2})"
        )

    def _slice(a: Optional[np.ndarray], lo: int, hi: int) -> Optional[np.ndarray]:
        return None if a is None else a[lo:hi]

    part1 = PriceData(
        data.prices[:b1],
        _slice(data.ask_volume, 0, b1),
        _slice(data.bid_volume, 0, b1),
    )
    part2 = PriceData(
        data.prices[b1:b2],
        _slice(data.ask_volume, b1, b2),
        _slice(data.bid_volume, b1, b2),
    )
    part3 = PriceData(
        data.prices[b2:],
        _slice(data.ask_volume, b2, n),
        _slice(data.bid_volume, b2, n),
    )
    return part1, part2, part3


# ==========================================================================
# cli (from run.py)
# ==========================================================================

def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.data_source:
        cfg.data_source = args.data_source
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.window_preset:
        cfg.window_preset = args.window_preset
        cfg.window_lengths = ()          # force preset re-resolution
        cfg.__post_init__()
    if args.downsample is not None:
        cfg.downsample = args.downsample
    if args.weight_method:
        cfg.weight_method = args.weight_method
    if args.span:
        cfg.history_span = args.span
    return cfg


def _build_model(cfg: Config):
    print(f"Loading data from: {cfg.data_source}")
    data = load_price_data(cfg)
    print(f"Loaded {len(data.prices):,} price points "
          f"(downsample={cfg.downsample}).")
    print(f"Window lengths: {cfg.window_lengths} | "
          f"clusters={cfg.n_clusters} selected={cfg.n_selected} | "
          f"weights={cfg.weight_method}")

    part1, part2, part3 = split_three(data, cfg.split_patterns, cfg.split_train)
    print(f"Partitions -> patterns={len(part1.prices):,} "
          f"train={len(part2.prices):,} test={len(part3.prices):,}")

    r2 = part2.order_book_r if cfg.use_order_book_r else None
    model = BayesianRegressionModel(config=cfg)
    print("Building pattern library and fitting weights ...")
    model.fit(part1.prices, part2.prices, r2)
    w = model.weights
    print(f"Learned weights: theta={np.round(w.theta, 5).tolist()} "
          f"theta0={w.theta0:.5f}")
    return model, part3


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(Config.from_env(args.env), args)
    model, part3 = _build_model(cfg)
    r3 = part3.order_book_r if cfg.use_order_book_r else None
    print("\nRunning backtest on held-out partition ...")
    result = backtest(model, part3.prices, cfg, r3)
    print("\n=== Backtest result ===")
    print(result.summary())
    print("\nReminder: research/education only, not financial advice. "
          "Backtested performance does not imply live results.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(Config.from_env(args.env), args)
    model, _ = _build_model(cfg)
    # Predict the next step from the most recent full window of all data.
    data = load_price_data(cfg)
    dp = model.predict_delta(data.prices)
    last = data.prices[-1]
    direction = "UP" if dp > 0 else "DOWN" if dp < 0 else "FLAT"
    print("\n=== Next-step prediction ===")
    print(f"  Last price        : {last:,.2f}")
    print(f"  Predicted change  : {dp:+.4f}")
    print(f"  Implied next price : {last + dp:,.2f}  ({direction})")
    print("\nReminder: research/education only, not financial advice.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they may be supplied either
    # before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env", default=".env", help="path to .env file")
    common.add_argument("--data-source", choices=["bitcoin_com", "csv"])
    common.add_argument("--csv-path")
    common.add_argument("--window-preset", choices=["hf", "daily", "daily_short"])
    common.add_argument("--downsample", type=int)
    common.add_argument("--weight-method", choices=["ols", "de"])
    common.add_argument("--span", help="Bitcoin.com history span, e.g. 'all' or '1y'")

    p = argparse.ArgumentParser(
        description=__doc__,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("backtest", parents=[common],
                   help="fit and backtest on held-out data")
    sub.add_parser("predict", parents=[common],
                   help="print one live next-step prediction")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "predict":
        return cmd_predict(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
