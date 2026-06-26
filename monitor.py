#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor.py — live sampler for the Umeh Jr / Kalshi 15-minute comparison
=======================================================================

Samples, on a fixed cadence, the live Binance price, the CF-Benchmarks proxy,
the Kalshi 15-minute target + market probability, and the Umeh Jr forecast +
model probability. Writes every sample to a CSV and prints a running table.

It can run for a fixed number of minutes OR until the current Kalshi 15-minute
window closes (and then print the settlement outcome vs. the target).

To sample faster than the exchange APIs comfortably allow, the *candle history*
is fetched once and tail-refreshed periodically, while the *benchmark + Kalshi*
are refreshed on a slower sub-cadence and carried live between refreshes using
the per-tick Binance move. (REST round-trips here are ~1-3s, so true sub-second
cadence needs a websocket feed — see NEXT_STEPS.md.)

Usage
-----
    python monitor.py --until-close                 # until the 15m window closes
    python monitor.py --minutes 5 --cadence 30      # 5 minutes, every 30s
    python monitor.py --cadence 1 --until-close      # ~1s cadence (latency-bound)
    python monitor.py --until-close --csv out.csv    # custom CSV path

Requires: numpy, and umeh_full.py in the same folder.

EDUCATIONAL ONLY. NOT FINANCIAL ADVICE.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umeh_full as u


def parse_settle(settle_utc: str) -> datetime:
    """Parse Kalshi 'HH:MM UTC' against today's date (UTC)."""
    base = datetime.now(timezone.utc).strftime("%Y-%m-%d ")
    return datetime.fromisoformat(base + settle_utc.replace(" UTC", "") + ":00+00:00")


def run(cadence: float, minutes: float, until_close: bool, csv_path: str) -> int:
    cfg = u.Config()

    k0 = u.fetch_kalshi_btc(cfg)
    if until_close and k0.ok:
        end_dt = parse_settle(k0.settle_utc)
        target0 = k0.target_price
        end_desc = f"Kalshi close {k0.settle_utc}"
    else:
        end_dt = datetime.now(timezone.utc).timestamp() + minutes * 60
        end_dt = datetime.fromtimestamp(end_dt, tz=timezone.utc)
        target0 = k0.target_price if k0.ok else float("nan")
        end_desc = f"{minutes:.0f} minutes"
    print(f"Monitoring ({end_desc}) at {cadence:g}s cadence.  "
          f"Target = ${target0:,.2f}" if not math.isnan(target0) else
          f"Monitoring ({end_desc}) at {cadence:g}s cadence.", flush=True)

    # Fetch candle history once; tail-refresh later.
    c1 = u.fetch_candles("1m", 300, cfg)
    c15 = u.fetch_candles("15m", 300, cfg)

    # Initial slow-source refresh.
    flow = u.fetch_order_flow(cfg)
    kalshi = u.fetch_kalshi_btc(cfg)
    bench_base, comps = u.fetch_benchmark_spot(cfg)
    binance_ref = u.fetch_spot(cfg)
    if math.isnan(bench_base):
        bench_base = binance_ref
    last_bench_t = time.time()
    bench_refresh = max(5.0, cadence)   # never refresh the heavy sources faster than 5s

    hdr = (f"{'time(UTC)':>9}  {'Binance':>11}  {'Benchmark':>11}  {'Target':>11}  "
           f"{'K.P(up)':>7}  {'UmehJrFcst':>11}  {'Jr.P(up)':>8}  {'dir':>4}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    tick = 0
    print_every = max(1, int(round(5.0 / cadence)))  # keep console readable at fast cadence
    while tick < 100000:
        t0 = time.time()
        now = datetime.now(timezone.utc)
        try:
            if t0 - last_bench_t >= bench_refresh:
                nb, nc = u.fetch_benchmark_spot(cfg)
                if not math.isnan(nb):
                    bench_base, comps = nb, nc
                    binance_ref = u.fetch_spot(cfg)
                flow = u.fetch_order_flow(cfg)
                kalshi = u.fetch_kalshi_btc(cfg)
                last_bench_t = t0
            if tick > 0 and tick % max(1, int(round(30.0 / cadence))) == 0:
                tail = u.fetch_candles("1m", 3, cfg)
                for t, cl, vol in zip(tail.times, tail.closes, tail.volumes):
                    idx = np.where(c1.times == t)[0]
                    if idx.size:
                        c1.closes[idx[-1]] = cl
                        c1.volumes[idx[-1]] = vol
                    else:
                        c1.times = np.append(c1.times, t)
                        c1.closes = np.append(c1.closes, cl)
                        c1.volumes = np.append(c1.volumes, vol)

            binance = u.fetch_spot(cfg)
            live_bench = bench_base + (binance - binance_ref)
            jr = u.compute_umeh_jr(cfg, c1, c15, live_bench, comps, binance, flow, kalshi)
            kpu = kalshi.prob_up if (kalshi and kalshi.ok) else float("nan")
            tgt = kalshi.target_price if (kalshi and kalshi.ok) else target0
            rows.append({"time": now.strftime("%H:%M:%S"), "binance": round(binance, 2),
                         "benchmark": round(live_bench, 2), "target": round(tgt, 2),
                         "kalshi_p_up": (round(kpu, 4) if kpu == kpu else None),
                         "jr_forecast": round(jr.forecast_settle, 2),
                         "jr_p_up": (round(jr.model_p_up, 4) if jr.model_p_up == jr.model_p_up else None),
                         "dir": jr.direction})
            if tick % print_every == 0:
                print(f"{now.strftime('%H:%M:%S'):>9}  {binance:>11,.2f}  {live_bench:>11,.2f}  "
                      f"{tgt:>11,.2f}  {(100*kpu if kpu==kpu else float('nan')):>6.1f}%  "
                      f"{jr.forecast_settle:>11,.2f}  "
                      f"{(100*jr.model_p_up if jr.model_p_up==jr.model_p_up else float('nan')):>7.1f}%  "
                      f"{jr.direction:>4}", flush=True)
        except Exception as e:
            print(f"{now.strftime('%H:%M:%S'):>9}  ERROR: {e}", flush=True)

        tick += 1
        secs_left = (end_dt - datetime.now(timezone.utc)).total_seconds()
        if secs_left <= 0.5:
            break
        time.sleep(min(cadence, max(0.2, secs_left), max(0.05, cadence - (time.time() - t0))))

    # Settlement reading (only meaningful for --until-close).
    settled_up = None
    if until_close and not math.isnan(target0):
        time.sleep(1)
        binance = u.fetch_spot(cfg)
        bench, _ = u.fetch_benchmark_spot(cfg)
        if math.isnan(bench):
            bench = binance
        settled_up = bench >= target0
        print("\n" + "=" * 70, flush=True)
        print(f"  WINDOW CLOSED at {k0.settle_utc}   (samples: {len(rows)})", flush=True)
        print(f"  Target (to beat)   : ${target0:,.2f}", flush=True)
        print(f"  Benchmark at close : ${bench:,.2f}   (Binance ${binance:,.2f})", flush=True)
        print(f"  Settled vs target  : "
              f"{'ABOVE (UP wins)' if settled_up else 'BELOW (DOWN wins)'}  "
              f"({bench - target0:+,.2f})", flush=True)
        print("  (Benchmark-at-close is a proxy; Kalshi settles on the official CF", flush=True)
        print("   Benchmarks index averaged over the final 60 seconds.)", flush=True)
        print("=" * 70, flush=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "binance", "benchmark", "target",
                                          "kalshi_p_up", "jr_forecast", "jr_p_up", "dir"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} samples -> {csv_path}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live Umeh Jr / Kalshi 15-minute monitor.")
    p.add_argument("--cadence", type=float, default=30.0, help="seconds between samples")
    p.add_argument("--minutes", type=float, default=5.0, help="run length if not --until-close")
    p.add_argument("--until-close", action="store_true",
                   help="run until the current Kalshi 15-minute window closes")
    p.add_argument("--csv", default=None, help="output CSV path")
    a = p.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    csv_path = a.csv or f"monitor_{stamp}.csv"
    return run(a.cadence, a.minutes, a.until_close, csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
