# Umeh Jr — Accuracy Analysis & Fine-Tuning Roadmap (CONTINUE HERE)

> This is the open task we stopped on. It captures **how Umeh Jr's projection
> under-performed**, backed by the live data we collected, and a concrete plan
> to fine-tune it. Everything you need is in `umeh_full.py`
> (`compute_umeh_jr`, `fetch_benchmark_spot`) and the `data/` CSVs.
>
> **EDUCATIONAL ONLY — NOT FINANCIAL ADVICE.**

---

## 1. What we measured (3 live Kalshi windows, all settled DOWN)

Because all three monitored windows settled **DOWN** (BTC below target), the
ideal "P(up)" is ~0, so lower P(up) and lower Brier score = better.

| Window | n | mean Jr P(up) | mean Kalshi P(up) | Brier Jr | Brier Kalshi |
|---|---|---|---|---|---|
| 01:11 (30s) | 11 | 11.0% | 9.1% | **0.014** | 0.009 |
| 01:30 (30s) | 20 | 24.5% | 26.5% | **0.106** | 0.129 |
| 01:45 (1s)  | 174 | 33.9% | 32.6% | **0.147** | 0.145 |

Plus, on the 1-second window:
- **Forecast hugs the benchmark**: mean |forecast − benchmark| = **$3.31**.
- **P(up) is jumpy**: mean tick-to-tick change 2.4pp, but **max single-tick swing
  47pp**.
- **Slightly under-confident near close**: last 25% of samples Jr 19.1% vs
  Kalshi 15.8%.

**Direction was 3/3 correct.** The problem isn't direction — it's the *quality of
the projection and the probability*.

---

## 2. How Umeh Jr "failed in accuracy" (root causes)

1. **It follows, it doesn't lead.** `expected_move` is tiny, so
   `forecast_settle ≈ benchmark` (|err| ~$3). It reacts to moves already in the
   benchmark rather than anticipating them. Defensible for a ~15-min random walk,
   but it offers little edge over "price stays put."

2. **The probability is over-sensitive / jumpy.** `σ_settle = σ1·√N·0.85`
   shrinks as `N → 0`, so near settlement a few-dollar benchmark wiggle flips
   P(up) by tens of points (observed 47pp on one tick). The benchmark-carry
   (`bench_base + (binance − binance_ref)`) injects noise when the **Binance.US
   ticker goes stale** (it repeated the same value for many seconds).

3. **Mild miscalibration vs. the market.** Brier is close to Kalshi's (even beat
   it in the 01:30 window), but Jr is a touch under-confident into the close,
   where the market sharpens correctly.

4. **The settlement mechanic isn't modeled.** Kalshi settles on the **60-second
   average** of the CF index near expiry; Umeh Jr forecasts the instantaneous
   value with a flat 0.85 damping factor that was never fit to data.

---

## 3. The fine-tuning plan (do these, measure each)

### 3.1 Add a validation harness FIRST (so changes are measured, not guessed)
- New function `backtest_umeh_jr()` in `umeh_full.py`: walk historical 1-minute
  candles; for each simulated 15-minute window, set `target = price at window
  open`, step minute-by-minute computing the model's `P(up)`, and compare to the
  **realized** outcome (close-of-window vs target).
- Report **Brier score, log-loss, and a calibration curve** (bucket predicted
  P(up) into deciles, plot realized frequency). This needs only candle history —
  no Kalshi data — so it can run offline and repeatedly.
- Target: beat the "always 0.5" and "random-walk normal" baselines, and approach
  the market's Brier.

### 3.2 Calibrate the volatility `σ_settle`
- Replace `σ1·√N·0.85` with an **EWMA realized-vol** of 1-minute benchmark
  returns (more responsive to regime), times `√N`.
- **Fit the averaging factor** (currently 0.85) against the realized dispersion
  of the 60-second-average vs the instantaneous close, from candle history.
- Add a **σ floor** (e.g. `max(σ_settle, k·σ1)`) so P(up) can't saturate to 0/1
  or swing 47pp on one tick.
- **EWMA-smooth P(up)** across ticks (e.g. `P_t = 0.6·P_t + 0.4·P_{t-1}`) to kill
  single-tick jumpiness.

### 3.3 Make the drift *lead* a little
- Scale `expected_move` by a **momentum/volume regime** factor: amplify when
  recent realized momentum is volume-confirmed (use `VolThrust`/order flow),
  shrink toward 0 in chop.
- Consider an explicit **EWMA-of-returns drift** term blended with the current
  fusion deltas.
- Keep it bounded — over-extrapolation is worse than random-walk on this horizon.

### 3.4 Model the 60-second-average settlement explicitly
- Forecast the **average over [T−60s, T]**, which ≈ the value at **T−30s**.
  Practically: use effective horizon `N_eff = max(N − 0.5, 0)` and integrate the
  drift to the midpoint of the last minute. This both lowers variance and removes
  the ad-hoc 0.85.

### 3.5 Fix the live data feed
- Use **Coinbase** (or a **websocket** tick stream) for the per-second price;
  Binance.US `ticker/price` is too stale for sub-minute sampling.
- Recompute the benchmark from the 4 exchanges as often as rate limits allow;
  only carry-interpolate as a fallback.

### 3.6 (Optional, display-only) market-aware shrinkage
- For a steadier *displayed* probability, shrink toward the market:
  `P_show = α·P_model + (1−α)·P_market`. Keep the **raw** model P(up) separate so
  the **edge** stays an independent signal.

---

## 4. Exact code pointers

- `umeh_full.py` → **SECTION 10b** → `compute_umeh_jr()` — the projection +
  probability formula to change (§3.2–3.4).
- `umeh_full.py` → `fetch_benchmark_spot()` / `fetch_spot()` — the data feed
  (§3.5).
- `umeh_full.py` → **SECTION 11** `backtest_roster()` — pattern to copy for the
  new `backtest_umeh_jr()` (§3.1).
- `monitor.py` — already writes labeled CSVs to validate live behavior after
  changes.

## 5. Suggested first step in Claude Code

> "Implement `backtest_umeh_jr()` in `umeh_full.py` per UMEH_NEXT_STEPS §3.1,
> then use it to calibrate `σ_settle` (§3.2). Report Brier/log-loss and a
> calibration curve before and after."

Get the harness in first; then every tuning change in §3.2–3.4 is a measured
win or loss instead of a guess.

**Educational use only. Not financial advice.**
