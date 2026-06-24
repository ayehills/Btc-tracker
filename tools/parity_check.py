#!/usr/bin/env python3
"""
parity_check.py — verify docs/umeh.js matches umeh_model.py exactly.

Builds a fixed, deterministic snapshot (synthetic price/volume series + a frozen
timestamp), computes the Umeh formula in Python, runs the same snapshot through
the JavaScript via Node, and asserts the key outputs agree to a tight tolerance.
This is the guarantee that the GitHub Pages page is a faithful mirror of the
Python reference.

    python tools/parity_check.py
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import umeh_model as um  # noqa: E402

from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_snapshot():
    rng = np.random.default_rng(42)
    n = 800
    # Deterministic random-walk-ish price series around 60k.
    steps = rng.normal(0, 25, n)
    closes = 60000 + np.cumsum(steps)
    volumes = np.abs(rng.normal(5, 2, n)) + 0.5
    spot = float(closes[-1] + 12.3)
    order_flow_r = 0.137
    now = datetime(2026, 6, 23, 20, 0, 0, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    return closes, volumes, spot, order_flow_r, now, now_ms


def main():
    closes, volumes, spot, r, now, now_ms = build_snapshot()

    py = um.compute_umeh(closes, volumes, spot, r, now=now)
    py_slow = um.nwachukwu_slow(closes, spot)

    snap = {
        "closes": closes.tolist(),
        "volumes": volumes.tolist(),
        "spot": spot,
        "order_flow_r": r,
        "now_ms": now_ms,
    }
    node_runner = os.path.join(ROOT, "tools", "parity_run.js")
    proc = subprocess.run(
        ["node", node_runner],
        input=json.dumps(snap),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("NODE ERROR:\n", proc.stderr)
        sys.exit(1)
    js = json.loads(proc.stdout)

    checks = [
        ("power_law.center", py.power_law["center"], js["power_law"]["center"]),
        ("power_law.support", py.power_law["support"], js["power_law"]["support"]),
        ("s2f.supply", py.stock_to_flow["supply"], js["stock_to_flow"]["supply"]),
        ("s2f.ratio", py.stock_to_flow["s2f_ratio"], js["stock_to_flow"]["s2f_ratio"]),
        ("s2f.model_price", py.stock_to_flow["model_price"], js["stock_to_flow"]["model_price"]),
        ("top_cap.top_price", py.top_cap["top_price"], js["top_cap"]["top_price"]),
        ("short_term_price", py.short_term_price, js["short_term_price"]),
        ("umeh_fair_value", py.umeh_fair_value, js["umeh_fair_value"]),
        ("pl_band_position", py.pl_band_position, js["pl_band_position"]),
        ("umeh_score", py.umeh_score, js["umeh_score"]),
        ("slow.predicted_price", py_slow["predicted_price"], js["slow"]["predicted_price"]),
        ("slow.delta", py_slow["delta"], js["slow"]["delta"]),
    ]

    print(f"{'field':22s} {'python':>16s} {'javascript':>16s}  status")
    ok = True
    for name, a, b in checks:
        rel = abs(a - b) / max(abs(a), 1e-9)
        passed = rel < 1e-6
        ok = ok and passed
        print(f"{name:22s} {a:16.6f} {b:16.6f}  {'OK' if passed else 'MISMATCH (%.2e)' % rel}")

    # Per-base-model agreement.
    for k in py.base_means:
        a, b = py.base_means[k], js["base_means"][k]
        rel = abs(a - b) / max(abs(a), 1e-9)
        if rel >= 1e-6:
            ok = False
            print(f"base_mean[{k}] MISMATCH: {a} vs {b} ({rel:.2e})")

    print("\nPARITY:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
