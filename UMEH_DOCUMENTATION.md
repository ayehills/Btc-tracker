# The Umeh Formula — Full Documentation, Logic & Reasoning

> **EDUCATIONAL ONLY — NOT FINANCIAL ADVICE.** The long-horizon valuation models
> (power law, stock-to-flow, top cap) are contested public fits and are
> frequently wrong. Backtested/modeled performance does not imply live results.

This document explains every model, formula, design decision, and piece of code
in the Umeh engine so you (or a fresh Claude Code session) can continue from
exactly where we left off. It is paired with:

- `umeh_full.py` — the complete, self-contained Python engine.
- `monitor.py` — the live sampler (Umeh Jr vs Kalshi).
- `UMEH_NEXT_STEPS.md` — the open task: fine-tuning Umeh Jr's projection.
- `CLAUDE.md` — orientation for Claude Code.

---

## 1. The big picture

The **Umeh formula** fuses four independent views of Bitcoin's price into one
composite, then specializes a fast variant ("**Umeh Jr**") for the Kalshi
15-minute prediction market.

| Pillar | What it answers | Horizon | File location |
|---|---|---|---|
| **Nwachukwu** | Where is price headed *right now*? | seconds–minutes | `kernel_bayesian_*`, base forecasters, `cfa_fuse` |
| **Power Law** | What is BTC "worth" long-term? | years | `power_law()` |
| **Stock-to-Flow** | What does scarcity imply? | years | `stock_to_flow()` |
| **Top Cap** | Where is the cycle ceiling? | years | `top_cap()` |

These combine into the **Umeh fair value** (a weighted geometric blend) and a
**0–100 Umeh score** (accumulate ↔ distribute). **Umeh Jr** ignores the
long-term pillars and focuses entirely on forecasting the **CF-Benchmarks index**
that Kalshi settles on, over the next 15-minute window.

---

## 2. Data sources (all public, no API key)

| Source | Endpoint | Used for |
|---|---|---|
| Binance.US | `/api/v3/klines` | 1m / 15m / 1h OHLCV (paginated for ~1 week of 1m) |
| Binance.US | `/api/v3/ticker/price` | live spot |
| Coinbase | `/products/BTC-USD/book?level=2` | order-book imbalance (buy/sell pressure) |
| Coinbase / Kraken / Bitstamp / Gemini | tickers | **benchmark composite** (CF-Benchmarks BRTI proxy) |
| Kalshi | `/trade-api/v2/markets?series_ticker=KXBTC15M` | 15-min "BTC up?" market (target + probability) |

> **Important data-quality note:** Binance.US (not global Binance) has thin
> liquidity and its `ticker/price` can go **stale** for many seconds at a time
> (observed during 1-second monitoring: the same value repeated across samples).
> For sub-minute work prefer Coinbase or a websocket feed. See `UMEH_NEXT_STEPS.md`.

Networking uses only the standard library (`urllib`); the sole third-party
dependency is **numpy**.

---

## 3. The Nwachukwu short-term predictor

### 3.1 Kernel-Bayesian estimator (Shah & Zhang, 2014)

For a window length `L`, take the most recent **z-scored** price window and
compare it to every historical z-scored window of the same length via an RBF
kernel. The predicted next-step change is the kernel-weighted average of those
windows' realized next moves:

```
w_i      = exp(-c * || zscore(window_i) - zscore(current) ||^2)
delta_i  = close[i+L] - close[i+L-1]
E[Δ]     = Σ w_i · delta_i / Σ w_i
```

We average `E[Δ]` over several window lengths (default `L ∈ {30,60,120}`),
subsampling up to `kb_n_ref` (1500) reference windows for speed. This is
**KMeans-free** (unlike the original BTCpredictor), which makes it deterministic
and trivially portable (the GitHub-Pages JS mirrors it exactly; see the web app).

Code: `kernel_bayesian_delta()`, `kernel_bayesian_price()`.

### 3.2 Diverse base forecasters

Each returns a next-price estimate; rosters differ by horizon:

| Model | Idea |
|---|---|
| `Bayesian` | the kernel-Bayesian estimate above |
| `Momentum{k}` | persist recent average drift |
| `MeanRev{n}` | pull a fraction toward the SMA |
| `EMA_MACD` | MACD histogram momentum |
| `RSI` | lean with momentum, fade extremes (>70/<30) |
| `ROC{n}` | rate-of-change momentum |
| `VolThrust` | volume-confirmed thrust (catches quick buys) |
| `OrderFlow` | live bid/ask imbalance tilt (live-only) |

- **`fast_roster()`** (1-minute horizon): Bayesian, Momentum(3), VolThrust, ROC,
  RSI, EMA/MACD, + OrderFlow when a live imbalance is supplied.
- **`slow_roster()`** (15m / hourly): Bayesian, Momentum(10), MeanRev(20), EMA/MACD.

**Why two rosters?** Backtests (Section 6) showed the thrust/volume models *help*
at 1 minute but *add noise* at 15 minutes, so they are excluded from the slow
horizons.

### 3.3 Combinatorial Fusion Analysis (CFA) — Wu, Ye, Xu & Hsu

The base predictions are fused, not averaged. Each model becomes a **scoring
system** over a grid of candidate prices (a truncated normal centered on its
prediction, peak-normalized). From the scores we derive:

- **Rank-Score Characteristic (RSC)** functions = scores sorted high→low.
- **Cognitive diversity** between two systems = RMS distance between their RSC
  functions: `CD(A,B) = sqrt(mean_i (f_A(i) - f_B(i))²)`.
- **Diversity strength** `ds(A)` = mean CD of A to the others.

The fused score is the **diversity-strength-weighted** sum of scores; the fused
prediction is its arg-max candidate price. Intuition: models that are both
decent *and* cognitively different get more say, because they correct each
other's errors.

Code: `cfa_fuse()`.

---

## 4. The long-term valuation pillars

All three are deterministic closed forms (no live price needed); constants live
in `Config`.

### 4.1 Power Law (Santostasi)
```
log10(price) = PL_A + PL_B · log10(days_since_genesis)      # PL_A=-17.0, PL_B=5.8
support  = center · 0.42      resistance = center · 2.10
```
`days_since_genesis` uses the 2009-01-03 genesis block. The intercept is
calibrated so the center line is ~$60k around mid-2024.

### 4.2 Stock-to-Flow (PlanB)
Supply and flow come from the **halving schedule** (deterministic):
```
height  = days · 144                      # ~10-min blocks
supply  = Σ (210,000 · reward_era)        # reward halves each 210k blocks
flow    = current_reward · 144 · 365
S2F     = supply / flow
price   = 0.40 · S2F^3.30
```
The S2F model runs hot post-2024-halving (it implies multi-$M prices), so it is
**down-weighted to 0.05** in the Umeh blend.

### 4.3 Top Cap (Willy Woo)
Average Cap is approximated from the power-law price integral:
```
average_price = center / (PL_B + 1)
top_price     = 35 · average_price
```
This is a transparent proxy; true Top Cap uses on-chain realized cap.

---

## 5. The Umeh composite

### 5.1 Fair value (weighted geometric blend of ln-anchors)
```
ln(fair) = Σ w_k · ln(anchor_k) / Σ w_k
anchors  = { market: spot, power_law: center, top_cap: top_price, s2f: model_price }
weights  = { market: 0.45, power_law: 0.38, top_cap: 0.12, s2f: 0.05 }
```

### 5.2 The 0–100 Umeh score
```
valuation = 1 - band_position           # band_position: 0=support, 1=resistance
momentum  = 0.5 + 0.5·tanh((st_price - spot)/(0.001·spot))
flow      = 0.5 + 0.5·clamp(order_flow_r, -1, 1)
ceiling   = clamp(headroom_to_top_cap, 0, 1)
score = 100·(0.40·valuation + 0.25·momentum + 0.20·flow + 0.15·ceiling)
```
High score → cheap vs. fair + buy pressure + room below the ceiling → "accumulate".

Code: `umeh_score()`, `compute_umeh()`.

---

## 6. Backtesting

`backtest_roster()` runs a **walk-forward**: for each held-out candle it fits the
roster on all prior data, predicts the next candle, and scores direction. It
reports directional hit-rate, **up-move recall** (fraction of up-moves caught),
MAPE, and mean absolute error. `python umeh_full.py backtest` runs it on 1m and
15m for both rosters.

**Findings that drove the design (live runs):**

| Roster / horizon | Direction | Up-move recall |
|---|---|---|
| 1m price-only | ~51% | ~40% |
| **1m fast/thrust** | **~60%** | **~70%** |
| **15m price-only** | **~59%** | ~59% |
| 15m fast/thrust | ~41% | ~31% |

→ thrust models on 1m, price-only on 15m. This is *why* there are two rosters.

---

## 7. Umeh Jr — the CF-Benchmarks 15-minute model

Kalshi's `KXBTC15M` market ("BTC price up in next 15 mins?") settles on **CF
Benchmarks' BRTI** — a multi-exchange index averaged over the final ~60 seconds —
**not** Binance's last price. Umeh Jr is built specifically for this.

### 7.1 The benchmark composite (BRTI proxy)
`fetch_benchmark_spot()` = simple mean of live spot from **Coinbase, Kraken,
Bitstamp, Gemini** (four BRTI constituents). During monitoring this ran
**~$50–120 below Binance**, which is exactly why a Binance-only model misjudges
Kalshi bets.

### 7.2 The forecast
```
near_delta   = fast-roster CFA fusion delta (1m, momentum/thrust/flow)
trend_delta  = slow-roster CFA fusion delta (15m)
expected_move = 0.40·near_delta·min(N,3) + 0.60·trend_delta·(N/15)
forecast_settle = benchmark + expected_move          # N = minutes to settle
```

### 7.3 The probability
Settlement volatility is grown over the horizon and damped for the 60-second
averaging:
```
σ_settle = per_minute_vol · sqrt(N) · 0.85
P(up)    = Φ((forecast_settle - target) / σ_settle)      # Φ = normal CDF
edge     = P(up)_model − P(up)_market
```
`edge > +0.07 → LEAN YES`, `edge < −0.07 → LEAN NO`, else no edge.

Code: `compute_umeh_jr()`, `render_jr()`. CLI: `python umeh_full.py jr`.

### 7.4 Kalshi market structure (`KXBTC15M`)
One active binary per 15-minute window:
- `floor_strike` = the **target** ("To Beat" level).
- `yes_bid/ask` (dollars, 0–1) = market **probability BTC finishes above target**.
- `close_time` = settlement (aligned to :00/:15/:30/:45).

Code: `fetch_kalshi_btc()` → `KalshiImplied`.

---

## 8. File map

```
umeh_full.py          # the whole engine (data + all models + CLI). ~1,400 lines.
monitor.py            # live sampler -> CSV (Umeh Jr vs Kalshi, until-close or N minutes)
requirements.txt      # numpy
UMEH_DOCUMENTATION.md # this file
UMEH_NEXT_STEPS.md    # the open task: fine-tune Umeh Jr's projection
CLAUDE.md             # orientation for Claude Code
data/                 # collected monitoring CSVs (evidence from live windows)
```

`umeh_full.py` internal sections (search for "SECTION"):
1 Config · 2 numeric utils · 3 data layer (incl. Kalshi + benchmark) ·
4 long-term models · 5 kernel-Bayesian · 6 base forecasters · 7 CFA ·
8 Nwachukwu forecasts · 9 projection fan · 10 Umeh composite ·
10b **Umeh Jr** · 11 backtester · 12 reporting · 13 orchestration · 14 CLI.

---

## 9. CLI reference

```bash
python umeh_full.py                 # full live report (all four pillars + Kalshi)
python umeh_full.py jr              # focused Umeh Jr 15-min Kalshi forecast
python umeh_full.py jr --every 10   # Umeh Jr live loop
python umeh_full.py monitor --every 10   # full-report live loop
python umeh_full.py backtest        # walk-forward accuracy
python umeh_full.py anchors         # long-term valuation only
python umeh_full.py --json          # machine-readable

python monitor.py --until-close             # sample until the 15m window closes
python monitor.py --minutes 5 --cadence 30  # fixed 5-minute run
python monitor.py --cadence 1 --until-close --csv run.csv
```

---

## 10. Known limitations (read before trusting any number)

1. **Benchmark is a proxy**, not the official CF Benchmarks BRTI (4 public
   exchanges, simple mean). Small differences at settlement are expected.
2. **Binance.US ticker can be stale** at sub-minute resolution — biases fast
   sampling. Prefer Coinbase / websockets for 1s work.
3. **Umeh Jr probability is not yet calibrated.** `σ_settle` is a heuristic; the
   model P(up) was observed to be jumpy and sometimes over/under-confident vs.
   the (well-calibrated) Kalshi market. **This is the open task** — see
   `UMEH_NEXT_STEPS.md`.
4. **Forecast ≈ benchmark + small drift**, i.e. close to a random walk. Defensible
   for 15-min horizons but it follows more than it leads.
5. Long-horizon pillars (power law / S2F / top cap) are **contested** and often
   wrong; they are context, not predictions.

---

## 11. References

- Shah & Zhang (2014), *Bayesian Regression and Bitcoin*. arXiv:1410.1231.
- Wu, Ye, Xu & Hsu, *Bitcoin Price Prediction using ML and Combinatorial Fusion
  Analysis*, IEEE CAI (2025). Hsu et al. on CFA / RSC / cognitive diversity.
- Santostasi, the Bitcoin Power Law.
- PlanB, the Stock-to-Flow model.
- Willy Woo, Top Cap / Average Cap.
- CF Benchmarks, Bitcoin Real-Time Index (BRTI) methodology.
- Kalshi trade-api v2 (public market data).

**Educational use only. Not financial advice.**
