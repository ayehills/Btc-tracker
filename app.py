"""
BTC Bayesian Tracker — web app
==============================

A single-page Flask app. Every time you refresh the page it:
  * fetches the live BTC/USD spot price,
  * runs the Bayesian-regression model (Shah & Zhang 2014) on recent
    intraday candles, and
  * shows a forecast for the price in 15 minutes and at the top of the hour.

Run:
    pip install -r requirements.txt
    python app.py
    # open http://localhost:5000

Research and education only. Not financial advice.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template

from predictor_service import get_analysis

app = Flask(__name__)


@app.route("/")
def index():
    analysis = get_analysis()
    return render_template("index.html", a=analysis)


@app.route("/api/analysis")
def api_analysis():
    """JSON version of the same analysis (handy for the auto-refresh toggle)."""
    a = get_analysis()
    return jsonify(
        {
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
                }
                for f in a.forecasts
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
