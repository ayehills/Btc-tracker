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

import json

from flask import Flask, jsonify, render_template

from predictor_service import get_live

app = Flask(__name__)


@app.route("/")
def index():
    # Embed an initial payload so the page paints instantly, then the page
    # keeps itself live by polling /api/live on an interval.
    initial = get_live()
    return render_template("index.html", initial_json=json.dumps(initial))


@app.route("/api/live")
def api_live():
    """Live bundle the front-end polls: spot price, forecasts, price history."""
    return jsonify(get_live())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
