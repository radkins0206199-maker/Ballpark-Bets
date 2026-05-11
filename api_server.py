"""
BallparkBets API Server — Enhanced
New endpoints: injuries, line movement, performance history,
parlays, fade the public, weather alerts, pitcher recent form.
"""

import os, json, csv, datetime, re, pytz, requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app      = Flask(__name__)
CORS(app)
ET       = pytz.timezone("America/New_York")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MLB_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "")


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

def write_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[write_json] {e}")

def parse_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except Exception:
        pass
    return rows

def _vig_free(hml, aml):
    def raw(ml):
        return (-ml / (-ml + 100)) if ml < 0 else (100 / (ml + 100))
    h, a = raw(hml), raw(aml)
    total = h + a
    return h / total if total else 0.5


# ── Health ──────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "date": et_today()})


# ── Status ──────────────────────────────────────────────────
@app.route("/api/mlb/status")
def status():
    preds     = read_json("daily_predictions.json", {})
    today     = et_today()
    pred_date = preds.get("date")
    fresh     = pred_date == today
    return jsonify({
        "predictions_fresh": fresh,
        "generated_at": preds.get("generated_at"),
        "prediction_date": pred_date,
        "today": today,
        "game_count": len(preds.get("games", [])),
        "best_bet_count": len(preds.get("best_bets", [])),
        "stale_reason": None if fresh else f"predictions for {pred_date}, today is {today}",
    })


# ── Predictions ─────────────────────────────────────────────
@app.route("/api/mlb/predictions")
def predictions():
    return jsonify(read_json("daily_predictions.json", {
        "date": None, "generated_at": None, "best_bets": [], "games": []
    }))


# ── Tracker ─────────────────────────────────────────────────
@app.route("/api/mlb/tracker")
def tracker():
    rows     = parse_csv("results_tracker.csv")
    resolved = [r for r in rows if r.get("Actual Winner", "").strip()]
    correct  = [r for r in resolved if r.get("Correct?", "").lower().startswith("correct")]
    wins, losses = len(correct), len(resolved) - len(correct)
    pct = round(wins / len(resolved) * 100, 1) if resolved else None
    conf_stats = {}
    for conf in ["High", "Medium", "Low"]:
        sub  = [r for r in resolved if r.get("Confidence", "") == conf]
        cor  = [r for r in sub if r.get("Correct?", "").lower().startswith("correct")]
        conf_stats[conf] = {"wins": len(cor), "losses": len(sub)-len(cor),
                            "pct": round(len(cor)/len(sub)*100, 1) if sub else None}
    return jsonify({
        "records": rows[-100:],
        "summary": {"total": len(rows), "resolved": len(resolved),
                    "wins": wins, "losses": losses, "pct": pct,
                    "by_confidence": conf_stats},
    })


# ── Historical performance breakdown ────────────────────────
@app.route("/api/mlb/performance")
def performance():
    rows     = parse_csv("results_tracker.csv")
    resolved = [r for r in rows if r.get("Actual Winner","").strip() and r.get("Correct?","").strip()]

    def stats(subset):
        cor   = [r for r in subset if r.get("Correct?","").lower().startswith("correct")]
        total = len(subset)
        w     = len(cor)
        return {"wins": w, "losses": total-w, "total": total,
                "pct": round(w/total*100, 1) if total else None}

    # By confidence tier
    by_conf = {c: stats([r for r in resolved if r.get("Confidence","") == c])
               for c in ["High","Medium","Low"]}

    # By pick side (home vs away)
    by_side = {
        "home_picks": stats([r for r in resolved if r.get("Predicted Winner","") == r.get("Home Team","")]),
        "away_picks": stats([r for r in resolved if r.get("Predicted Winner","") == r.get("Away Team","")]),
    }

    # By month
    by_month_raw = {}
    for r in resolved:
        month = r.get("Date","")[:7] or "unknown"
        by_month_raw.setdefault(month, []).append(r)
    by_month = {m: stats(v) for m, v in sorted(by_month_raw.items())}

    # By model edge bucket
    def edge_bucket(r):
        try:
            e = float(r.get("Model Edge", 0))
            if e >= 10: return "10%+"
            if e >= 5:  return "5-10%"
            if e >= 0:  return "0-5%"
            if e >= -5: return "-5-0%"
            return "<-5%"
        except Exception:
            return "unknown"
    by_edge_raw = {}
    for r in resolved:
        b = edge_bucket(r)
        by_edge_raw.setdefault(b, []).append(r)
    by_edge = {k: stats(v) for k, v in by_edge_raw.items()}

    # Streak
    def streak(rows):
        if not rows: return {"type": None, "count": 0}
        last = rows[-1].get("Correct?","").lower().startswith("correct")
        count = 1
        for r in reversed(rows[:-1]):
            if r.get("Correct?","").lower().startswith("correct") == last:
                count += 1
            else:
                break
        return {"type": "win" if last else "loss", "count": count}

    return jsonify({
        "total_resolved": len(resolved),
        "overall": stats(resolved),
        "by_confidence": by_conf,
        "by_side": by_side,
        "by_month": by_month,
        "by_edge_bucket": by_edge,
        "last_10": stats(resolved[-10:]) if len(resolved) >= 10 else stats(resolved),
        "streak": streak(resolved),
    })


# ── Injuries & IL ────────────────────────────────────────────
@app.route("/api/mlb/injuries")
def injuries():
    today  = et_today()
    cached = read_json("injuries_cache.json", {})
    if cached.get("date") == today:
        return jsonify(cached)
    try:
        r = requests.get(f"{MLB_BASE}/transactions",
                         params={"sportId": 1, "date": today, "limit": 200}, timeout=10)
        il_moves = []
        if r.status_code == 200:
            for tx in r.json().get("transactions", []):
                td = tx.get("typeDesc", "")
                if any(k in td.lower() for k in ["injured","il ","disable","60-day"]):
                    il_moves.append({
                        "player": tx.get("player", {}).get("fullName", ""),
                        "team":   (tx.get("fromTeam") or tx.get("toTeam") or {}).get("name", ""),
                        "type":   td,
                        "date":   tx.get("date", today),
                    })
        result = {"date": today, "il_moves": il_moves, "count": len(il_moves)}
        write_json("injuries_cache.json", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"date": today, "il_moves": [], "count": 0, "error": str(e)})


# ── Line movement & sharp money ──────────────────────────────
@app.route("/api/mlb/lines")
def lines():
    today  = et_today()
    cached = read_json("lines_cache.json", {})
    if cached.get("date") == today and cached.get("games"):
        return jsonify(cached)
    if not ODDS_KEY:
        return jsonify({"date": today, "games": [], "error": "ODDS_API_KEY not set"})
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={"apiKey": ODDS_KEY, "regions": "us",
                    "markets": "h2h", "oddsFormat": "american"}, timeout=10)
        if r.status_code != 200:
            return jsonify({"date": today, "games": [], "error": f"Odds API {r.status_code}"})

        games = []
        prev_games = {g.get("home"): g for g in cached.get("games", [])}
        for game in r.json():
            home, away = game.get("home_team",""), game.get("away_team","")
            home_mls, away_mls = [], []
            for bm in game.get("bookmakers", []):
                h2h = next((m for m in bm.get("markets",[]) if m["key"]=="h2h"), None)
                if not h2h: continue
                oc = {o["name"]: o["price"] for o in h2h.get("outcomes",[])}
                if home in oc and away in oc:
                    home_mls.append(oc[home])
                    away_mls.append(oc[away])
            if not home_mls: continue
            avg_hml = round(sum(home_mls)/len(home_mls))
            avg_aml = round(sum(away_mls)/len(away_mls))
            prev = prev_games.get(home, {})
            movement = None
            sharp_signal = None
            if prev.get("home_ml") is not None:
                diff = avg_hml - prev["home_ml"]
                if abs(diff) >= 5:
                    movement = diff
                    if avg_hml < 0 and diff > 0:
                        sharp_signal = f"Line moving away from {home} — sharps may be fading"
                    elif avg_hml < 0 and diff < 0:
                        sharp_signal = f"Steam on {home} — sharp money incoming"
                    elif avg_hml > 0 and diff < 0:
                        sharp_signal = f"{home} (underdog) attracting sharp action"
            games.append({
                "home": home, "away": away,
                "home_ml": avg_hml, "away_ml": avg_aml,
                "books": len(home_mls),
                "line_movement": movement,
                "sharp_signal": sharp_signal,
                "commence_time": game.get("commence_time"),
            })
        result = {"date": today, "games": games, "count": len(games)}
        write_json("lines_cache.json", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"date": today, "games": [], "error": str(e)})


# ── Pitcher last 3 starts ────────────────────────────────────
@app.route("/api/mlb/pitcher/<int:pitcher_id>/recent")
def pitcher_recent(pitcher_id):
    season = datetime.datetime.now(ET).year
    try:
        r = requests.get(f"{MLB_BASE}/people/{pitcher_id}/stats",
                         params={"stats":"gameLog","group":"pitching",
                                 "season":season,"sportId":1}, timeout=10)
        if r.status_code != 200:
            return jsonify({"last_3_starts": [], "error": f"MLB API {r.status_code}"})
        splits = r.json().get("stats",[{}])[0].get("splits",[])
        starts = [s for s in splits if float(s.get("stat",{}).get("inningsPitched",0) or 0) > 1.0]
        last3  = list(reversed(starts[-3:])) if starts else []
        formatted = []
        for s in last3:
            st = s.get("stat",{})
            ip = float(st.get("inningsPitched",0) or 0)
            er = int(st.get("earnedRuns",0) or 0)
            formatted.append({
                "date":      s.get("date",""),
                "opponent":  s.get("opponent",{}).get("name",""),
                "ip":        st.get("inningsPitched","—"),
                "hits":      st.get("hits",0),
                "earned_runs": er,
                "strikeouts": st.get("strikeOuts",0),
                "walks":     st.get("baseOnBalls",0),
                "era_game":  round(er / max(ip,0.1) * 9, 2),
            })
        return jsonify({"pitcher_id": pitcher_id, "last_3_starts": formatted})
    except Exception as e:
        return jsonify({"last_3_starts": [], "error": str(e)})


# ── Parlays ─────────────────────────────────────────────────
@app.route("/api/mlb/parlays")
def parlays():
    preds  = read_json("daily_predictions.json", {})
    games  = preds.get("best_bets") or preds.get("games") or []

    def edge_score(e):
        if e is None: return 0
        if e >= 15: return 10
        if e >= 12: return 9
        if e >= 9:  return 8
        if e >= 6:  return 7
        if e >= 4:  return 6
        if e >= 2:  return 5
        return max(0, int(e))

    def pick_prob(g):
        hp = (g.get("Home Win Probability") or 50) / 100
        return hp if g.get("Predicted Winner") == g.get("Home Team") else 1 - hp

    strong = sorted(
        [g for g in games if edge_score(g.get("Model Edge")) >= 6 and g.get("Predicted Winner")],
        key=lambda g: edge_score(g.get("Model Edge")), reverse=True
    )

    result = []
    # 3-leg parlay
    if len(strong) >= 3:
        legs3 = strong[:3]
        probs = [pick_prob(g) for g in legs3]
        result.append({
            "legs": [{"pick": g["Predicted Winner"],
                      "game": f"{g['Away Team']} @ {g['Home Team']}",
                      "time": g.get("Game Time",""),
                      "edge_score": edge_score(g.get("Model Edge")),
                      "win_prob": round(p*100,1)} for g,p in zip(legs3,probs)],
            "combined_prob": round(probs[0]*probs[1]*probs[2]*100,1),
            "combined_score": round(sum(edge_score(g.get("Model Edge")) for g in legs3)/3,1),
            "label": "3-leg parlay",
        })

    # All 2-leg combos
    for i in range(len(strong)):
        for j in range(i+1, len(strong)):
            g1, g2 = strong[i], strong[j]
            p1, p2 = pick_prob(g1), pick_prob(g2)
            result.append({
                "legs": [
                    {"pick": g1["Predicted Winner"], "game": f"{g1['Away Team']} @ {g1['Home Team']}",
                     "time": g1.get("Game Time",""), "edge_score": edge_score(g1.get("Model Edge")), "win_prob": round(p1*100,1)},
                    {"pick": g2["Predicted Winner"], "game": f"{g2['Away Team']} @ {g2['Home Team']}",
                     "time": g2.get("Game Time",""), "edge_score": edge_score(g2.get("Model Edge")), "win_prob": round(p2*100,1)},
                ],
                "combined_prob": round(p1*p2*100,1),
                "combined_score": round((edge_score(g1.get("Model Edge"))+edge_score(g2.get("Model Edge")))/2,1),
                "label": "2-leg parlay",
            })

    result.sort(key=lambda p: p["combined_score"], reverse=True)
    return jsonify({"date": preds.get("date"), "parlays": result[:5], "strong_picks": len(strong)})


# ── Fade the public ─────────────────────────────────────────
@app.route("/api/mlb/public")
def public_bets():
    today = et_today()
    if not ODDS_KEY:
        return jsonify({"date": today, "games": [], "error": "ODDS_API_KEY not set"})
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={"apiKey": ODDS_KEY,"regions":"us","markets":"h2h","oddsFormat":"american"}, timeout=10)
        if r.status_code != 200:
            return jsonify({"date": today, "games": [], "error": f"API {r.status_code}"})
        games = []
        for game in r.json():
            home, away = game.get("home_team",""), game.get("away_team","")
            home_mls = []
            for bm in game.get("bookmakers",[]):
                h2h = next((m for m in bm.get("markets",[]) if m["key"]=="h2h"), None)
                if not h2h: continue
                oc = {o["name"]: o["price"] for o in h2h.get("outcomes",[])}
                if home in oc and away in oc:
                    home_mls.append(oc[home])
            if not home_mls: continue
            avg_ml   = sum(home_mls) / len(home_mls)
            implied  = (-avg_ml)/(-avg_ml+100) if avg_ml < 0 else 100/(avg_ml+100)
            fade = None
            if implied > 0.68:
                fade = {"signal": f"Public heavy on {home} ({round(implied*100)}%) — fade {away}", "fade_team": away}
            elif implied < 0.32:
                fade = {"signal": f"Public heavy on {away} ({round((1-implied)*100)}%) — fade {home}", "fade_team": home}
            if fade:
                games.append({"home": home, "away": away, "home_ml": round(avg_ml),
                               "home_implied": round(implied*100,1), **fade})
        return jsonify({"date": today, "games": games})
    except Exception as e:
        return jsonify({"date": today, "games": [], "error": str(e)})


# ── Weather alerts ───────────────────────────────────────────
@app.route("/api/mlb/weather")
def weather_alerts():
    preds  = read_json("daily_predictions.json", {})
    alerts = []
    for g in preds.get("games", []):
        w = g.get("Weather","") or ""
        if not w: continue
        wm = re.search(r'(\d+)mph\s*(OUT|IN)', w, re.IGNORECASE)
        tm = re.search(r'(\d+)F', w)
        ws  = int(wm.group(1)) if wm else 0
        wd  = wm.group(2).upper() if wm else ""
        tmp = int(tm.group(1)) if tm else 72
        alert = None
        if wd == "OUT" and ws >= 12:
            alert = f"💨 Wind OUT {ws}mph — expect high scoring"
        elif wd == "IN" and ws >= 10:
            alert = f"🌬 Wind IN {ws}mph — pitcher-friendly"
        elif tmp <= 45:
            alert = f"🥶 {tmp}°F — suppressed offense"
        elif tmp >= 92 and "dome" not in w.lower():
            alert = f"🥵 {tmp}°F heat — late-inning fatigue factor"
        if alert:
            alerts.append({"home": g.get("Home Team"), "away": g.get("Away Team"),
                           "weather": w, "alert": alert,
                           "wind_speed": ws, "wind_dir": wd, "temp": tmp})
    return jsonify({"date": preds.get("date"), "alerts": alerts})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
