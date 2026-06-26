# CLAUDE.md — project orientation for Claude Code

> Read this first. It tells a fresh Claude Code session what this project is,
> where things live, and what to work on next.
>
> **EDUCATIONAL ONLY — NOT FINANCIAL ADVICE.** This is a research/learning
> project about Bitcoin price modeling. Do not present outputs as trading advice.

## What this project is

A Bitcoin price-analysis engine — the **Umeh formula** — that fuses four models
(a live short-term predictor called **Nwachukwu**, plus **Power Law**,
**Stock-to-Flow**, and **Top Cap**), and a specialized fast variant **Umeh Jr**
that forecasts the **CF-Benchmarks index** Kalshi settles on, for the Kalshi
15-minute "BTC up?" market.

There are two surfaces:
1. **Python engine (the current focus)** — runs locally, fetches live data, no
   browser. This is what the user runs each session.
2. **A web app + GitHub Pages site** (older, still in the repo) — the Flask app
   (`app.py`, `predictor_service.py`, `nwachukwu_model.py`,
   `btc_bayesian_predictor.py`) and the client-side page in `docs/`
   (`docs/index.html`, `docs/umeh.js`), deployed at
   `https://ayehills.github.io/Btc-tracker/`.

## Key files (Python engine — start here)

| File | What it is |
| --- | --- |
| `umeh_full.py` | **The whole engine** — data layer + all models + CLI. Self-contained (numpy + stdlib only). ~1,400 lines, organized into numbered SECTIONs. |
| `monitor.py` | Live sampler → CSV: Umeh Jr vs Kalshi, `--until-close` or `--minutes N` at any `--cadence`. |
| `UMEH_DOCUMENTATION.md` | Full math/logic/reasoning for every model + file map + CLI. |
| `UMEH_NEXT_STEPS.md` | **The open task** — Umeh Jr accuracy analysis + fine-tuning roadmap. |
| `data/*.csv` | Collected live-monitoring windows (evidence; all settled DOWN). |
| `requirements.txt` | `numpy` (engine). The web app's deps are separate. |

## How to run

```bash
pip install numpy
python umeh_full.py              # full live report (4 pillars + Kalshi)
python umeh_full.py jr           # focused Umeh Jr 15-min Kalshi forecast
python umeh_full.py backtest     # walk-forward directional accuracy
python monitor.py --until-close  # sample live until the 15m window closes -> CSV
```

Live data: Binance.US (klines/spot), Coinbase (order book), Coinbase/Kraken/
Bitstamp/Gemini (benchmark composite), Kalshi public API (`KXBTC15M`). No keys.

## Current state / what to do next

Direction calls are solid (3/3 correct on live windows), but **Umeh Jr's
projection and probability need fine-tuning** — the forecast follows rather than
leads, and P(up) is jumpy/over-sensitive near settlement. The full analysis and
a step-by-step plan are in **`UMEH_NEXT_STEPS.md`**.

**Recommended first move:** implement `backtest_umeh_jr()` (UMEH_NEXT_STEPS §3.1)
so every tuning change is measured (Brier / log-loss / calibration), then
calibrate `σ_settle` and the drift. Code to change: `compute_umeh_jr()` in
`umeh_full.py` SECTION 10b.

## Conventions

- Keep `umeh_full.py` **self-contained** (numpy + stdlib only) so it stays a
  single downloadable file.
- The web app's JS (`docs/umeh.js`) mirrors the Python long-term math and the
  KMeans-free kernel-Bayesian; if you change shared formulas, keep them in sync
  (`tools/parity_check.py` validates the page vs the Python reference).
- Everything is **educational** — preserve the disclaimers.
