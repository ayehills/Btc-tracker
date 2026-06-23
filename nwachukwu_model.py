"""
nwachukwu_model
===============

The **Nwachukwu Model** — a blended Bitcoin price forecaster that fuses the
Bayesian-regression estimator of Shah & Zhang (2014) with the *Combinatorial
Fusion Analysis* (CFA) framework of Wu, Ye, Xu & Hsu (2025/26, IEEE CAI),
"Bitcoin Price Prediction using Machine Learning and Combinatorial Fusion
Analysis".

Idea
----
No single estimator is robust on its own. CFA combines a moderate set of
*diverse* and *individually decent* scoring systems and consistently beats any
one of them. The Nwachukwu Model does exactly that for live next-step BTC
prediction:

  1. Several diverse, fast base forecasters each predict the next price and
     carry an uncertainty (a per-model residual std). Following the paper, each
     becomes a **scoring system** by spreading its prediction into a (truncated)
     normal distribution over a grid of candidate prices — the density is the
     score s_A(d_i), normalized to [0, 1].
  2. From the scores we derive each model's **rank function** r_A and its
     **rank-score characteristic (RSC) function** f_A(i) = s_A(r_A^{-1}(i)).
  3. **Cognitive diversity** between two systems is the RMS area between their
     RSC functions:  CD(A,B) = sqrt( mean_i (f_A(i) - f_B(i))^2 ).
     A model's **diversity strength** ds(A_j) is its mean CD to the others.
  4. The systems are fused by **score combination** (optionally **weighted by
     diversity strength**, WCDS). The fused score's arg-max candidate price is
     the Nwachukwu prediction.

Base models include the Bayesian regressor (the "expert" port) plus momentum,
mean-reversion, an EMA/MACD technical model, and a random-walk baseline — a
deliberately diverse roster, mirroring the paper's use of structurally
different learners (SVM/RF/XGBoost/CNN/LSTM) and technical indicators (EMA,
MACD).

Research and education only. Not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from btc_bayesian_predictor import (
    BayesianRegressionModel,
    Config,
    WindowPreset,
)

_EPS = 1e-12


# --------------------------------------------------------------------------
# Base forecasters — each predicts the next-step price and a residual std.
# --------------------------------------------------------------------------

@dataclass
class BasePrediction:
    name: str
    mean: float   # predicted next-step price
    std: float    # uncertainty (residual standard deviation)


class BaseForecaster:
    """Interface: fit on a 1-D close series, then predict the next price."""

    name: str = "base"

    def predict_from(self, closes: np.ndarray) -> float:
        """One-step-ahead point prediction using ``closes`` as the history."""
        raise NotImplementedError

    def fit(self, closes: np.ndarray) -> "BaseForecaster":
        return self

    # Residual std via a light walk-forward over the most recent window. This
    # mirrors the paper's "std derived from the test set" but stays cheap.
    def residual_std(self, closes: np.ndarray, window: int = 60) -> float:
        n = len(closes)
        span = min(window, n - 5)
        if span <= 2:
            return float(np.std(np.diff(closes)) or 1.0)
        errs = []
        for t in range(n - span, n - 1):
            pred = self.predict_from(closes[: t + 1])
            errs.append(closes[t + 1] - pred)
        s = float(np.std(errs)) if errs else 0.0
        # Never let std collapse to zero (degenerate scoring system).
        return s if s > _EPS else float(np.std(np.diff(closes)) or 1.0)

    def predict(self, closes: np.ndarray) -> BasePrediction:
        return BasePrediction(self.name, self.predict_from(closes),
                              self.residual_std(closes))


class RandomWalkForecaster(BaseForecaster):
    name = "RandomWalk"

    def predict_from(self, closes: np.ndarray) -> float:
        return float(closes[-1])


class MomentumForecaster(BaseForecaster):
    """Persist the recent average drift (trend-following)."""

    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"Momentum{k}"

    def predict_from(self, closes: np.ndarray) -> float:
        if len(closes) < self.k + 1:
            return float(closes[-1])
        drift = float(np.mean(np.diff(closes[-(self.k + 1):])))
        return float(closes[-1] + drift)


class MeanReversionForecaster(BaseForecaster):
    """Pull the price a fraction of the way back toward a moving average."""

    def __init__(self, n: int = 20, alpha: float = 0.25):
        self.n = n
        self.alpha = alpha
        self.name = f"MeanRev{n}"

    def predict_from(self, closes: np.ndarray) -> float:
        n = min(self.n, len(closes))
        sma = float(np.mean(closes[-n:]))
        last = float(closes[-1])
        return last + self.alpha * (sma - last)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


class EmaMacdForecaster(BaseForecaster):
    """Technical model using EMA + MACD momentum (per the paper's indicators)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast, self.slow, self.signal = fast, slow, signal
        self.name = "EMA_MACD"

    def predict_from(self, closes: np.ndarray) -> float:
        if len(closes) < self.slow + self.signal:
            return float(closes[-1])
        macd = _ema(closes, self.fast) - _ema(closes, self.slow)
        signal_line = _ema(macd, self.signal)
        hist = float(macd[-1] - signal_line[-1])  # MACD histogram (momentum)
        return float(closes[-1] + hist)


class BayesianForecaster(BaseForecaster):
    """Wraps the Shah & Zhang Bayesian-regression estimator as a base model."""

    name = "Bayesian"

    def __init__(self, config: Optional[Config] = None, split: float = 0.60):
        self.config = config or Config(
            data_source="csv",
            window_preset=WindowPreset.DAILY_SHORT,
            n_clusters=60,
            n_selected=15,
            smoothing_c=0.25,
            weight_method="ols",
            random_state=0,
        )
        self.split = split
        self._model: Optional[BayesianRegressionModel] = None

    def fit(self, closes: np.ndarray) -> "BayesianForecaster":
        cut = int(len(closes) * self.split)
        self._model = BayesianRegressionModel(config=self.config)
        self._model.fit(closes[:cut], closes[cut:])
        return self

    def predict_from(self, closes: np.ndarray) -> float:
        if self._model is None or not self._model.is_fitted():
            self.fit(closes)
        delta = self._model.predict_delta(closes)
        return float(closes[-1] + delta)

    # The Bayesian walk-forward is comparatively expensive; estimate its
    # uncertainty from realized one-step volatility instead of re-fitting.
    def residual_std(self, closes: np.ndarray, window: int = 60) -> float:
        span = min(window, len(closes) - 1)
        diffs = np.diff(closes[-(span + 1):]) if span > 1 else np.diff(closes)
        s = float(np.std(diffs))
        return s if s > _EPS else 1.0


# --------------------------------------------------------------------------
# Combinatorial Fusion Analysis
# --------------------------------------------------------------------------

def _normal_score(grid: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Truncated-normal density over ``grid``, normalized so the peak is 1."""
    std = max(std, _EPS)
    z = (grid - mean) / std
    score = np.exp(-0.5 * z * z)
    # Truncate at 2 std (the paper keeps ~95% of the mass), as a soft mask.
    score = np.where(np.abs(z) <= 2.0, score, 0.0)
    peak = score.max()
    if peak <= _EPS:
        return np.zeros_like(grid)
    return score / peak


def _rsc(scores: np.ndarray) -> np.ndarray:
    """Rank-Score Characteristic function: scores sorted high-to-low."""
    return np.sort(scores)[::-1]


def cognitive_diversity(f_a: np.ndarray, f_b: np.ndarray) -> float:
    """RMS distance between two RSC functions (Eq. 1 of Wu et al.)."""
    return float(np.sqrt(np.mean((f_a - f_b) ** 2)))


@dataclass
class NwachukwuResult:
    predicted_price: float
    delta: float                 # predicted change vs. the last close
    direction: str
    base_means: dict             # model name -> predicted price
    diversity_strength: dict     # model name -> ds(A_j)
    weights: dict                # model name -> fusion weight used
    combination: str
    grid_low: float
    grid_high: float
    n_points: int                # history length used


@dataclass
class NwachukwuModel:
    """Blend of the Bayesian estimator and CFA over diverse base models."""

    base_models: List[BaseForecaster] = field(default_factory=list)
    weighting: str = "diversity"   # "diversity" (WCDS) or "average"
    combination: str = "score"     # "score" (SC) or "rank" (RC)
    grid_size: int = 401
    trunc_std: float = 2.0

    @classmethod
    def default(cls, config: Optional[Config] = None,
                weighting: str = "diversity") -> "NwachukwuModel":
        return cls(
            base_models=[
                BayesianForecaster(config=config),
                MomentumForecaster(k=10),
                MeanReversionForecaster(n=20, alpha=0.25),
                EmaMacdForecaster(),
                RandomWalkForecaster(),
            ],
            weighting=weighting,
        )

    def fit(self, closes: np.ndarray) -> "NwachukwuModel":
        closes = np.asarray(closes, dtype=np.float64).ravel()
        for m in self.base_models:
            m.fit(closes)
        self._closes = closes
        return self

    def predict(self, closes: Optional[np.ndarray] = None) -> NwachukwuResult:
        closes = np.asarray(
            self._closes if closes is None else closes, dtype=np.float64
        ).ravel()
        last = float(closes[-1])

        # 1) Each base model -> a (mean, std) prediction.
        preds = [m.predict(closes) for m in self.base_models]

        # 2) Build a shared candidate-price grid spanning all truncated normals.
        lo = min(p.mean - self.trunc_std * p.std for p in preds)
        hi = max(p.mean + self.trunc_std * p.std for p in preds)
        lo = max(0.0, lo)
        if hi - lo < _EPS:
            hi = lo + 1.0
        grid = np.linspace(lo, hi, self.grid_size)

        # 3) Score each system over the grid; derive RSC functions.
        scores = np.vstack([_normal_score(grid, p.mean, p.std) for p in preds])
        rscs = np.vstack([_rsc(scores[i]) for i in range(len(preds))])

        # 4) Cognitive diversity -> diversity strength per model.
        t = len(preds)
        ds = np.zeros(t)
        if t > 1:
            for j in range(t):
                ds[j] = np.mean([
                    cognitive_diversity(rscs[j], rscs[k])
                    for k in range(t) if k != j
                ])

        # 5) Fusion weights.
        if self.weighting == "diversity" and ds.sum() > _EPS:
            w = ds.copy()
        else:
            w = np.ones(t)

        # 6) Combine. Score combination: weighted average of scores ->
        #    arg-max candidate. Rank combination: weighted average of ranks
        #    (lower is better) using 1/w weights, per the paper.
        if self.combination == "rank":
            ranks = np.vstack([
                (len(grid) - np.argsort(np.argsort(scores[i])))  # rank 1 = best
                for i in range(t)
            ]).astype(np.float64)
            inv = 1.0 / np.maximum(w, _EPS)
            combined_rank = (inv[:, None] * ranks).sum(axis=0) / inv.sum()
            best = int(np.argmin(combined_rank))
        else:
            combined = (w[:, None] * scores).sum(axis=0) / max(w.sum(), _EPS)
            best = int(np.argmax(combined))

        predicted_price = float(grid[best])
        delta = predicted_price - last

        return NwachukwuResult(
            predicted_price=predicted_price,
            delta=delta,
            direction="UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT",
            base_means={p.name: p.mean for p in preds},
            diversity_strength={preds[i].name: float(ds[i]) for i in range(t)},
            weights={preds[i].name: float(w[i]) for i in range(t)},
            combination=f"{self.combination}/{self.weighting}",
            grid_low=lo,
            grid_high=hi,
            n_points=len(closes),
        )
