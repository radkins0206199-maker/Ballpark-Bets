"""
BallparkBets API Server
Runs on Render alongside the Python ML scheduler.
Serves predictions, tracker data, and status to the Expo app and Netlify website.
"""

import os
import json
import csv
import datetime
import pytz
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from Netlify and Expo Go

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ET = pytz.timezone("America/New_York")


def et_today():
    return datetime.datetime.now(ET).strftime("%Y-%m-%d")


def read_json(filename, default=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def parse_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        pass
    return rows


# ── Health ─────────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# ── Status ─────────────────────────────────────────────────────
@app.route("/api/mlb/status")
def status():
    preds = read_json("daily_predictions.json", {})
    today = et_today()
    pred_date = preds.get("date")
    fresh = pred_date == today
    return jsonify({
        "predictions_fresh": fresh,
        "generated_at": preds.get("generated_at"),
        "prediction_date": pred_date,
        "today": today,
        "game_count": len(preds.get("games", [])),
        "best_bet_count": len(preds.get("best_bets", [])),
        "stale_reason": None if fresh else f"predictions are for {pred_date}, today is {today}",
    })


# ── Predictions ─────────────────────────────────────────────────
@app.route("/api/mlb/predictions")
def predictions():
    data = read_json("daily_predictions.json", {
        "date": None,
        "generated_at": None,
        "best_bets": [],
        "games": [],
    })
    return jsonify(data)


# ── Tracker ─────────────────────────────────────────────────────
@app.route("/api/mlb/tracker")
def tracker():
    rows = parse_csv("results_tracker.csv")
    total = len(rows)
    resolved = [r for r in rows if r.get("Actual Winner", "").strip()]
    correct = [r for r in resolved if r.get("Correct?", "").strip().lower().startswith("correct")]
    wins = len(correct)
    losses = len(resolved) - wins
    pct = round(wins / len(resolved) * 100, 1) if resolved else None

    # Confidence breakdown
    conf_stats = {}
    for conf in ["High", "Medium", "Low"]:
        c_resolved = [r for r in resolved if r.get("Confidence", "") == conf]
        c_correct = [r for r in c_resolved if r.get("Correct?", "").lower().startswith("correct")]
        conf_stats[conf] = {
            "wins": len(c_correct),
            "losses": len(c_resolved) - len(c_correct),
            "pct": round(len(c_correct) / len(c_resolved) * 100, 1) if c_resolved else None,
        }

    return jsonify({
        "records": rows[-100:],  # last 100 records
        "summary": {
            "total": total,
            "resolved": len(resolved),
            "wins": wins,
            "losses": losses,
            "pct": pct,
            "by_confidence": conf_stats,
        }
    })


# ── Tracker chart ────────────────────────────────────────────────
@app.route("/api/mlb/tracker/chart")
def tracker_chart():
    rows = parse_csv("results_tracker.csv")

    def parse_csv_line_safe(line):
        """Simple RFC4180 field parser."""
        fields, field, in_quotes = [], "", False
        for ch in line:
            if in_quotes:
                if ch == '"':
                    in_quotes = False
                else:
                    field += ch
            elif ch == '"':
                in_quotes = True
            elif ch == ',':
                fields.append(field.strip()); field = ""
            else:
                field += ch
        fields.append(field.strip())
        return fields

    by_date = {}
    for row in rows:
        date = row.get("Date", "").strip()
        actual = row.get("Actual Winner", "").strip()
        predicted = row.get("Predicted Winner", "").strip()
        correct = row.get("Correct?", "").strip().lower().startswith("correct")
        if not date or not actual:
            continue
        entry = by_date.setdefault(date, {"wins": 0, "losses": 0})
        if correct:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

    sorted_dates = sorted(by_date.keys())
    cum_w = cum_l = 0
    points = []
    for date in sorted_dates:
        e = by_date[date]
        cum_w += e["wins"]; cum_l += e["losses"]
        total = cum_w + cum_l
        points.append({
            "date": date,
            "wins": e["wins"],
            "losses": e["losses"],
            "cumWins": cum_w,
            "cumLosses": cum_l,
            "pct": round(cum_w / total * 100) if total else 0,
        })

    return jsonify({"points": points})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
