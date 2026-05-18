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
@app.route("/api/mlb/retrain", methods=["GET", "POST"])
def trigger_retrain():
    """Manually trigger model retraining — returns detailed status."""
    import io, sys
    try:
        from mlb_daily_report import retrain_model_if_needed, RETRAIN_FLAG, _load_resolved_from_supabase
        
        # Clear the flag so retrain isn't skipped
        if os.path.exists(RETRAIN_FLAG):
            os.remove(RETRAIN_FLAG)
            print("[Retrain] Cleared retrain flag", flush=True)

        # Check what data is available
        df = _load_resolved_from_supabase()
        if df is None:
            # Try to get error details
            import psycopg2
            db_url = os.environ.get("DATABASE_URL", "NOT SET")
            db_url_safe = db_url[:30] + "..." if len(db_url) > 30 else db_url
            return jsonify({
                "success": False, 
                "error": "Could not load Supabase data",
                "db_url_prefix": db_url_safe,
                "rows": 0
            })
        
        total_rows = len(df)
        has_actual = df['Actual Winner'].notna().sum()
        
        # Count rows with real features
        feat_cols = ['pitcher_era', 'opp_pitcher_era', 'bullpen_era', 'opp_bullpen_era']
        available_feat_cols = [c for c in feat_cols if c in df.columns]
        if available_feat_cols:
            has_features = (df[available_feat_cols].notna().any(axis=1)).sum()
        else:
            has_features = 0

        print(f"[Retrain] Total rows: {total_rows}, with actual: {has_actual}, with features: {has_features}", flush=True)

        # Run retrain
        retrain_model_if_needed()
        
        return jsonify({
            "success": True,
            "total_rows": int(total_rows),
            "rows_with_actual": int(has_actual),
            "rows_with_features": int(has_features),
            "message": "Check Render logs for [ML] output"
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()[-500:]}), 500


@app.route("/api/mlb/backfill-features", methods=["GET", "POST"])
def backfill_features():
    """Push feature data from CSV into Supabase results table."""
    try:
        import psycopg2, psycopg2.extras, csv
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return jsonify({"success": False, "error": "DATABASE_URL not set"})

        csv_path = os.path.join(os.path.dirname(__file__), "results_tracker.csv")
        if not os.path.exists(csv_path):
            return jsonify({"success": False, "error": "results_tracker.csv not found"})

        def safe_float(v):
            try: return float(v) if v and str(v).strip() not in ('','nan','None') else None
            except: return None

        def safe_bool(v):
            if str(v).strip().lower() in ('true','1'): return True
            if str(v).strip().lower() in ('false','0'): return False
            return None

        rows = []
        with open(csv_path, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))

        conn = psycopg2.connect(db_url, sslmode='require', connect_timeout=15)
        cur  = conn.cursor()
        updated = 0

        for r in rows:
            date = r.get('Date','').strip()
            home = r.get('Home Team','').strip()
            away = r.get('Away Team','').strip()
            if not date or not home or not away:
                continue

            # Only update if we have at least one feature value
            pitcher_era = safe_float(r.get('pitcher_era') or r.get('Home SP ERA (at Home)'))
            bullpen_era = safe_float(r.get('bullpen_era') or r.get('Home Bullpen ERA'))
            park_factor = safe_float(r.get('park_factor') or r.get('Park Factor'))

            if pitcher_era is None and bullpen_era is None and park_factor is None:
                continue

            cur.execute("""
                UPDATE results SET
                    home_sp_era = %s,
                    away_sp_era = %s,
                    bullpen_era = %s,
                    opp_bullpen_era = %s,
                    park_factor = %s,
                    temp = %s,
                    wind_speed = %s,
                    wind_dir_out = %s,
                    home = %s,
                    pitcher_recent_delta = %s,
                    opp_pitcher_recent_delta = %s,
                    predicted_winner = COALESCE(NULLIF(predicted_winner,''), %s),
                    actual_winner = COALESCE(NULLIF(actual_winner,''), %s),
                    correct = COALESCE(NULLIF(correct,''), %s),
                    confidence = COALESCE(NULLIF(confidence,''), %s),
                    model_edge = COALESCE(model_edge, %s),
                    home_win_prob = COALESCE(home_win_prob, %s)
                WHERE date=%s AND home_team=%s AND away_team=%s
            """, (
                pitcher_era,
                safe_float(r.get('opp_pitcher_era') or r.get('Away SP ERA (on Road)')),
                bullpen_era,
                safe_float(r.get('opp_bullpen_era') or r.get('Away Bullpen ERA')),
                park_factor,
                safe_float(r.get('temp')),
                safe_float(r.get('wind_speed')),
                safe_bool(r.get('wind_dir_out')),
                safe_bool(r.get('home')),
                safe_float(r.get('pitcher_recent_delta')),
                safe_float(r.get('opp_pitcher_recent_delta')),
                r.get('Predicted Winner','').strip() or None,
                r.get('Actual Winner','').strip() or None,
                r.get('Correct?','').strip() or None,
                r.get('Confidence','').strip() or None,
                safe_float(r.get('Model Edge')),
                safe_float(r.get('Home Win %')),
                date, home, away
            ))
            if cur.rowcount > 0:
                updated += 1

        conn.commit()
        cur.close(); conn.close()

        return jsonify({
            "success": True,
            "csv_rows": len(rows),
            "updated": updated,
            "message": f"Updated {updated} rows with feature data"
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()[-500:]})
def resolve_yesterday_endpoint():
    """Resolve yesterday's games — fetch scores from MLB API and write to Supabase."""
    from mlb_daily_report import auto_fill_results
    try:
        auto_fill_results()
        return jsonify({"success": True, "message": "Yesterday's games resolved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/mlb/tracker")
def tracker():
    """Read results from Supabase first, fall back to CSV."""
    rows = []
    
    # Try Supabase first
    try:
        import psycopg2, psycopg2.extras
        db_url = os.environ.get("DATABASE_URL", "")
        if db_url:
            conn = psycopg2.connect(db_url, sslmode='require')
            cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT date, home_team, away_team, predicted_winner,
                       actual_winner, correct, confidence, model_edge,
                       home_win_prob, game_time
                FROM results ORDER BY date ASC LIMIT 500
            """)
            db_rows = cur.fetchall()
            cur.close(); conn.close()
            rows = [{
                "Date":             r["date"] or "",
                "Home Team":        r["home_team"] or "",
                "Away Team":        r["away_team"] or "",
                "Predicted Winner": r["predicted_winner"] or "",
                "Actual Winner":    r["actual_winner"] or "",
                "Correct?":         r["correct"] or "",
                "Confidence":       r["confidence"] or "",
                "Model Edge":       str(r["model_edge"]) if r["model_edge"] is not None else "",
                "Home Win %":       str(r["home_win_prob"]) if r["home_win_prob"] is not None else "",
                "Game Time":        r["game_time"] or "",
            } for r in db_rows]
            print(f"[tracker] Loaded {len(rows)} rows from Supabase")
    except Exception as e:
        print(f"[tracker] Supabase failed: {e} — falling back to CSV")
        rows = parse_csv("results_tracker.csv")

    resolved = [r for r in rows if r.get("Actual Winner", "").strip()]
    correct  = [r for r in resolved if str(r.get("Correct?", "")).startswith("✓") or 
                str(r.get("Correct?","")).lower().startswith("correct")]
    wins, losses = len(correct), len(resolved) - len(correct)
    pct = round(wins / len(resolved) * 100, 1) if resolved else None

    conf_stats = {}
    for conf in ["High", "Medium", "Low"]:
        sub = [r for r in resolved if r.get("Confidence", "") == conf]
        cor = [r for r in sub if str(r.get("Correct?","")).startswith("✓") or 
               str(r.get("Correct?","")).lower().startswith("correct")]
        conf_stats[conf] = {
            "wins": len(cor), "losses": len(sub) - len(cor),
            "pct": round(len(cor)/len(sub)*100, 1) if sub else None
        }

    return jsonify({
        "records": rows[-200:],
        "summary": {
            "total": len(rows), "resolved": len(resolved),
            "wins": wins, "losses": losses, "pct": pct,
            "by_confidence": conf_stats,
            "source": "supabase" if rows and "date" not in str(rows[0].keys()) else "supabase",
        },
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
    # Use ALL games with predictions, not just the Python-labeled best_bets
    # Edge score filter is applied here, same as the frontend
    all_games = preds.get("games") or []
    best_bets  = preds.get("best_bets") or []

    # Merge, dedup by matchup
    merged = list(all_games)
    for bb in best_bets:
        exists = any(
            g.get("Home Team") == bb.get("Home Team") and
            g.get("Away Team") == bb.get("Away Team")
            for g in merged
        )
        if not exists:
            merged.append(bb)

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
        pick = g.get("Predicted Winner","")
        home = g.get("Home Team","")
        return hp if pick == home else 1 - hp

    # Only use games with full predictions and edge score 6+
    strong = sorted(
        [g for g in merged
         if g.get("Predicted Winner") and
         g.get("Model Edge") is not None and
         edge_score(g.get("Model Edge")) >= 6],
        key=lambda g: edge_score(g.get("Model Edge")),
        reverse=True
    )

    result = []

    # Best 3-leg parlay
    if len(strong) >= 3:
        legs3  = strong[:3]
        probs  = [pick_prob(g) for g in legs3]
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




# ── Export results_tracker.csv ───────────────────────────────
@app.route("/api/mlb/export/results")
def export_results():
    """Download the current results_tracker.csv directly from Render."""
    path = os.path.join(DATA_DIR, "results_tracker.csv")
    if not os.path.exists(path):
        return jsonify({"error": "No results file found"}), 404
    from flask import send_file
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"results_tracker_{et_today()}.csv",
    )


@app.route("/api/mlb/props")
def player_props():
    today  = et_today()
    cached = read_json("props_cache.json", {})
    if cached.get("date") == today and cached.get("props"):
        return jsonify(cached)

    if not ODDS_KEY:
        return jsonify({"date": today, "props": [], "error": "ODDS_API_KEY not set"})

    try:
        # Step 1 — get today's MLB event IDs from The Odds API
        events_r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
            params={"apiKey": ODDS_KEY, "dateFormat": "iso"},
            timeout=10,
        )
        if events_r.status_code != 200:
            return jsonify({"date": today, "props": [], "error": f"Events API {events_r.status_code}"})

        events = events_r.json()
        # Filter to today's games only
        today_events = [e for e in events if e.get("commence_time","")[:10] == today]

        all_props = []

        for event in today_events[:8]:  # cap at 8 games to save API quota
            event_id   = event.get("id")
            home_team  = event.get("home_team","")
            away_team  = event.get("away_team","")

            # Step 2 — fetch player props for this event
            props_r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
                params={
                    "apiKey": ODDS_KEY,
                    "regions": "us",
                    "markets": "batter_hits,batter_home_runs,batter_total_bases,pitcher_strikeouts",
                    "oddsFormat": "american",
                },
                timeout=10,
            )
            if props_r.status_code != 200:
                continue

            data       = props_r.json()
            bookmakers = data.get("bookmakers", [])
            if not bookmakers:
                continue

            # Collect lines from all bookmakers and average them
            prop_lines = {}  # {(player, market, point): [prices]}
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    mkey = market.get("key","")
                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description","")
                        name   = outcome.get("name","")   # Over / Under
                        point  = outcome.get("point", 0)
                        price  = outcome.get("price", 0)
                        if name == "Over" and player:
                            key = (player, mkey, point)
                            prop_lines.setdefault(key, []).append(price)

            # Build prop objects
            for (player, market, line), prices in prop_lines.items():
                avg_price = round(sum(prices)/len(prices))
                market_label = {
                    "pitcher_strikeouts": "Strikeouts",
                    "batter_hits":        "Hits",
                    "batter_home_runs":   "Home Runs",
                    "batter_total_bases": "Total Bases",
                }.get(market, market)

                # Step 3 — get player stats from MLB API
                player_stats = _get_player_stats(player, market)
                season_avg   = player_stats.get("season_avg")
                l10_avg      = player_stats.get("l10_avg")
                player_id    = player_stats.get("player_id")

                # Step 4 — compute edge score
                edge_score   = _prop_edge_score(line, l10_avg, season_avg, market)

                all_props.append({
                    "player":       player,
                    "player_id":    player_id,
                    "market":       market,
                    "market_label": market_label,
                    "line":         line,
                    "bet":          f"Over {line}",
                    "avg_price":    avg_price,
                    "season_avg":   season_avg,
                    "l10_avg":      l10_avg,
                    "edge_score":   edge_score,
                    "home_team":    home_team,
                    "away_team":    away_team,
                    "game":         f"{away_team} @ {home_team}",
                })

        # Sort by edge score descending
        all_props.sort(key=lambda p: p["edge_score"] or 0, reverse=True)

        result = {"date": today, "props": all_props[:30]}  # top 30 props
        write_json("props_cache.json", result)
        return jsonify(result)

    except Exception as e:
        print(f"[Props] Error: {e}")
        return jsonify({"date": today, "props": [], "error": str(e)})


def _get_player_stats(player_name, market):
    """Fetch season avg and last 10 game avg for a player from MLB Stats API."""
    try:
        # Search for player by name
        search_r = requests.get(
            f"{MLB_BASE}/people/search",
            params={"names": player_name, "sportId": 1},
            timeout=8,
        )
        if search_r.status_code != 200:
            return {}
        people = search_r.json().get("people", [])
        if not people:
            return {}
        player_id = people[0]["id"]

        # Determine stat group
        group    = "pitching" if "pitcher" in market else "hitting"
        stat_key = {
            "pitcher_strikeouts":  "strikeOuts",
            "batter_hits":         "hits",
            "batter_home_runs":    "homeRuns",
            "batter_total_bases":  "totalBases",
        }.get(market, "hits")

        season = datetime.datetime.now(ET).year

        # Season stats
        season_r = requests.get(
            f"{MLB_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": group, "season": season, "sportId": 1},
            timeout=8,
        )
        season_avg = None
        games_played = 1
        if season_r.status_code == 200:
            splits = season_r.json().get("stats",[{}])[0].get("splits",[])
            if splits:
                st = splits[0].get("stat", {})
                total = int(st.get(stat_key, 0) or 0)
                gp    = int(st.get("gamesPlayed", 1) or 1)
                games_played = max(gp, 1)
                season_avg = round(total / games_played, 1)

        # Last 10 game log
        log_r = requests.get(
            f"{MLB_BASE}/people/{player_id}/stats",
            params={"stats": "gameLog", "group": group, "season": season, "sportId": 1},
            timeout=8,
        )
        l10_avg = None
        if log_r.status_code == 200:
            splits = log_r.json().get("stats",[{}])[0].get("splits",[])
            last10 = splits[-10:] if len(splits) >= 10 else splits
            if last10:
                vals = [int(s.get("stat",{}).get(stat_key, 0) or 0) for s in last10]
                l10_avg = round(sum(vals)/len(vals), 1)

        return {"player_id": player_id, "season_avg": season_avg, "l10_avg": l10_avg}

    except Exception as e:
        print(f"[PlayerStats] {player_name}: {e}")
        return {}


def _prop_edge_score(line, l10_avg, season_avg, market):
    """
    Compute edge score 1-10 for a prop Over bet.
    Uses L10 average as primary signal, season avg as secondary.
    """
    if l10_avg is None and season_avg is None:
        return 5  # neutral when no data

    primary = l10_avg if l10_avg is not None else season_avg
    backup  = season_avg if season_avg is not None else primary

    if line == 0:
        return 5

    # How much does the player's recent avg exceed the line?
    # Express as a percentage above/below the line
    over_pct  = (primary - line) / line * 100 if line else 0
    over_pct2 = (backup  - line) / line * 100 if line else 0

    # Both signals agree = stronger score
    both_over = over_pct > 0 and over_pct2 > 0

    if over_pct >= 40 and both_over:  return 10
    if over_pct >= 30 and both_over:  return 9
    if over_pct >= 20 and both_over:  return 8
    if over_pct >= 10 and both_over:  return 7
    if over_pct >= 10:                return 6
    if over_pct >= 0:                 return 5
    if over_pct >= -10:               return 4
    if over_pct >= -20:               return 3
    if over_pct >= -30:               return 2
    return 1


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
