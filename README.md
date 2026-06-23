# BTC Tracker · Nwachukwu Model

A small Flask webpage that, **every time you refresh**, shows the live BTC/USD
price and forecasts the price **15 minutes from now** and **at the top of the
hour** — recomputed on each load by running the **Nwachukwu Model**.

## The Nwachukwu Model

The Nwachukwu Model blends two published approaches:

1. **Bayesian Regression and Bitcoin** — Shah & Zhang (2014). A pattern-mining
   Bayesian estimator (the faithful Python port in
   [`btc_bayesian_predictor.py`](btc_bayesian_predictor.py)) that predicts the
   next-step price change from RBF-kernel-weighted historical patterns.

2. **Combinatorial Fusion Analysis (CFA)** — Wu, Ye, Xu & Hsu,
   *"Bitcoin Price Prediction using Machine Learning and Combinatorial Fusion
   Analysis"* (IEEE CAI). Rather than trusting one model, CFA combines a set of
   **diverse**, individually-decent scoring systems and reliably beats any one
   of them.

How the blend works (in [`nwachukwu_model.py`](nwachukwu_model.py)):

- **Five diverse base forecasters** each predict the next price with an
  uncertainty: the **Bayesian** regressor (the expert), **Momentum**,
  **Mean-Reversion**, an **EMA/MACD** technical model, and a **Random-Walk**
  baseline.
- Each prediction is spread into a truncated normal distribution over a grid of
  candidate prices — the density is that model's **score** `s_A(d_i)`,
  normalized to `[0, 1]` (exactly the paper's construction).
- From the scores we derive each model's **rank function** and **Rank-Score
  Characteristic (RSC) function** `f_A(i) = s_A(r_A⁻¹(i))`.
- **Cognitive diversity** between two systems is the RMS area between their RSC
  functions, `CD(A,B) = sqrt(mean_i (f_A(i) − f_B(i))²)`; a model's **diversity
  strength** is its mean CD to the others.
- The systems are fused by **score combination weighted by diversity strength**
  (WCDS). The fused score's arg-max candidate price is the Nwachukwu forecast.

The forecast change is anchored to the **live spot price** on each refresh so
the number reflects the price you actually see now.

## Run it

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Press **Refresh now** (or tick *Auto-refresh every 60s*) to recompute.

## Data sources

- **Spot price:** Kraken public ticker (Coinbase fallback).
- **Candles:** Kraken public OHLC — 15-minute candles for the 15-minute
  forecast, 1-hour candles for the top-of-the-hour forecast. No API key needed.

Fitted models are cached per timeframe for ~90s so rapid refreshes stay instant;
the live spot price is always re-fetched.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask web app (`/` page, `/api/analysis` JSON). |
| `predictor_service.py` | Live data feed + caching + builds the forecasts. |
| `nwachukwu_model.py` | The Nwachukwu Model (Bayesian + CFA fusion). |
| `btc_bayesian_predictor.py` | Shah & Zhang Bayesian-regression port (base model). |
| `templates/index.html` | The page UI, including the CFA fusion breakdown. |

## Disclaimer

Research and education only. **Not financial advice.** Modeled performance does
not imply live results.
