import math
import json
import os
import pickle
import datetime
import io
import requests
import pandas as pd
import numpy as np
import pytz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import smtplib
from email.message import EmailMessage

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# ── Always use Eastern Time for dates ─────────────────────────
# Render runs in UTC. Games run until ~1 AM ET on the West Coast.
# Using UTC causes tomorrow's date to appear after ~8 PM ET.
ET = pytz.timezone("America/New_York")

def today_et():
    """Return today's date in Eastern Time as YYYY-MM-DD string."""
    return datetime.datetime.now(ET).strftime("%Y-%m-%d")

def yesterday_et():
    """Return yesterday's date in Eastern Time as YYYY-MM-DD string."""
    yd = datetime.datetime.now(ET) - datetime.timedelta(days=1)
    return yd.strftime("%Y-%m-%d")

CURRENT_SEASON = datetime.datetime.now(ET).year


class CalibratedModel:
    """Wraps a fitted classifier with an isotonic regression calibrator.
    Trained on a held-out calibration set so predict_proba() outputs
    well-calibrated probabilities without expensive cross-validation."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _train_calibrated(estimator, X_tr, y_tr, X_cal, y_cal):
    """Fit estimator on training split, then fit isotonic calibrator on cal split."""
    estimator.fit(X_tr, y_tr)
    raw_cal = estimator.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)
    return CalibratedModel(estimator, iso)

# In-process cache so we don't hit the API twice for the same team/pitcher
_team_stats_cache       = {}
_pitcher_stats_cache    = {}
_standings_cache        = {}
_savant_cache           = {}
_vegas_cache            = {}
_lineup_cache           = {}
_bullpen_fatigue_cache  = {}
_weather_cache          = {}

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
TRACKER_CSV       = os.path.join(SCRIPT_DIR, "results_tracker.csv")
RETRAIN_FLAG      = os.path.join(SCRIPT_DIR, ".last_retrain_date")
MODEL_CACHE       = os.path.join(SCRIPT_DIR, "ml_model_cache.pkl")
PREDICTIONS_JSON  = os.path.join(SCRIPT_DIR, "daily_predictions.json")
TRACKER_JSON      = os.path.join(SCRIPT_DIR, "results_tracker.json")

# Core display columns + ML feature columns stored for auto-recalibration
TRACKER_COLS = [
    "Date", "Game Time", "Home Team", "Away Team", "Predicted Winner",
    "Home Win %", "Model Edge", "Confidence", "Actual Winner",
    # ML training features
    "team_wpct", "opp_wpct", "team_rpg", "opp_rpg",
    "pitcher_era", "opp_pitcher_era", "bullpen_era", "opp_bullpen_era",
    "park_factor", "temp", "wind_speed", "wind_dir_out", "home",
    "pitcher_recent_delta", "opp_pitcher_recent_delta",
]

# ============================================================
# A. LIVE DATA FROM MLB STATS API
# ============================================================

def _get(url, params=None):
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [API] WARNING: {e}")
        return None


def fetch_standings():
    """Return dict of team_id -> (wpct, rpg) from live standings."""
    global _standings_cache
    if _standings_cache:
        return _standings_cache

    data = _get(f"{MLB_BASE}/standings",
                params={"leagueId": "103,104", "season": CURRENT_SEASON, "standingsTypes": "regularSeason"})
    if not data:
        return {}

    result = {}
    for division in data.get("records", []):
        for rec in division.get("teamRecords", []):
            team_id = rec["team"]["id"]
            wpct = float(rec.get("winningPercentage", "0.500"))
            games = rec.get("gamesPlayed", 1) or 1
            runs = rec.get("runsScored", 0) or 0
            rpg = round(runs / games, 2)
            result[team_id] = {"wpct": wpct, "rpg": rpg}

    _standings_cache = result
    return result


# ============================================================
# B. BASEBALL SAVANT / STATCAST DATA
# ============================================================

def fetch_savant_team_batting():
    """
    Aggregate Baseball Savant batting Statcast (hard hit %, barrel rate, xwOBA)
    to the team level by joining player IDs with MLB API roster data.
    Results are cached for the run.
    """
    global _savant_cache
    if _savant_cache:
        return _savant_cache

    # 1. player_id → team_id from MLB API active roster
    players_data = _get(f"{MLB_BASE}/sports/1/players",
                        {"season": CURRENT_SEASON, "gameType": "R"})
    if not players_data:
        return {}
    player_to_team = {}
    for p in players_data.get("people", []):
        pid = p.get("id")
        tid = p.get("currentTeam", {}).get("id")
        if pid and tid:
            player_to_team[pid] = tid

    # 2. team_id → full team name
    teams_data = _get(f"{MLB_BASE}/teams", {"sportId": 1, "season": CURRENT_SEASON})
    team_id_to_name = {}
    if teams_data:
        for t in teams_data.get("teams", []):
            team_id_to_name[t["id"]] = t["name"]

    # 3. Baseball Savant batting leaderboard (player-level)
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/leaderboard/custom",
            params={
                "year": CURRENT_SEASON, "type": "batter", "filter": "",
                "sort": "4", "sortDir": "desc", "min": "25",
                "selections": "hard_hit_percent,barrel_batted_rate,xwoba",
                "csv": "true",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"  [Savant] WARNING: Batting leaderboard failed: {e}")
        return {}

    # 4. Join and aggregate by team
    df["team_id"]   = df["player_id"].map(player_to_team)
    df["team_name"] = df["team_id"].map(team_id_to_name)
    df = df.dropna(subset=["team_name"])

    result = {}
    for team_name, grp in df.groupby("team_name"):
        result[team_name] = {
            "hard_hit":    round(float(grp["hard_hit_percent"].mean()) / 100, 4),
            "barrel_rate": round(float(grp["barrel_batted_rate"].mean()) / 100, 4),
            "xwoba":       round(float(grp["xwoba"].mean()), 4),
        }

    _savant_cache = result
    print(f"  [Savant] Team batting stats loaded for {len(result)} teams")
    return result


def get_savant_pitcher_stats(pitcher_id):
    """
    Fetch Baseball Savant Statcast data for a specific pitcher:
    xwOBA allowed, hard hit % allowed, barrel rate against.
    Returns None if unavailable.
    """
    if not pitcher_id:
        return None
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/statcast_search/csv",
            params={
                "all": "true", "hfGT": "R|", "hfSea": f"{CURRENT_SEASON}|",
                "player_type": "pitcher",
                "pitchers_lookup[]": pitcher_id,
                "group_by": "name",
                "sort_col": "pitches", "sort_order": "desc", "min_pas": "0",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "xwoba":       float(row.get("xwoba",                  0.320)),
            "hard_hit_pct": float(row.get("hardhit_percent",       35.0)),
            "barrel_rate":  float(row.get("barrels_per_bbe_percent", 7.0)),
        }
    except Exception:
        return None


def xwoba_to_era_adj(xwoba, base_era):
    """
    Blend a pitcher's ERA with a Statcast xwOBA-implied ERA for a more
    predictive quality metric.
      - League avg: xwOBA ≈ 0.315, ERA ≈ 4.20
      - Each 0.020 xwOBA ≈ 0.50 ERA difference
      - Final = 60% ERA + 40% xwOBA-derived ERA
    """
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA   = 4.20
    xwoba_era = LEAGUE_ERA + (xwoba - LEAGUE_XWOBA) * 25
    return round(base_era * 0.60 + xwoba_era * 0.40, 2)


_FORMULA_CHARS = ("=", "+", "-", "@", "\t", "\r")

# Only these CSV columns contain free-form text from untrusted third-party
# feeds (team names, pitcher names, weather strings).  Numeric columns such
# as "Model Edge" or "Home Win %" are intentionally excluded so that negative
# numeric strings stored as text (e.g. "-3.2") are not prefixed and can still
# be parsed correctly by downstream consumers.
_CSV_TEXT_COLS = frozenset({
    "Date", "Game Time", "Home Team", "Away Team",
    "Predicted Winner", "Confidence", "Actual Winner",
})


def sanitize_cell(val):
    """Neutralize spreadsheet formula injection in untrusted string values.

    Spreadsheet applications (Excel, Google Sheets, LibreOffice) treat cell
    values that begin with '=', '+', '-', '@', TAB, or CR as formulas and
    execute them on open.  This function prefixes any such string with a
    leading single-quote so the application renders it as plain text instead.
    Strings that start with '+' or '-' but parse as valid numbers are left
    untouched because they represent legitimate numeric text, not formulas.
    Non-string values (numbers, None, booleans) are returned unchanged.
    """
    if not isinstance(val, str):
        return val
    if val.startswith(("=", "@", "\t", "\r")):
        return "'" + val
    if val.startswith(("+", "-")):
        try:
            float(val)
            return val  # Legitimate signed-number string; not a formula
        except (ValueError, TypeError):
            return "'" + val
    return val


def _sanitize_dataframe(df):
    """Apply sanitize_cell() to free-form text columns of a DataFrame.

    Only columns listed in _CSV_TEXT_COLS are sanitized.  Numeric columns
    (e.g. 'Model Edge', 'Home Win %') are left untouched so that values such
    as '-3.2' remain parseable as numbers by downstream consumers.
    """
    for col in _CSV_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: sanitize_cell(v) if isinstance(v, str) else v)
    return df


def update_results_tracker(games_list):
    """
    Append today's predictions to results_tracker.csv.
    Removes any existing rows for today first (safe to re-run).
    Uses Eastern Time date — not UTC — to prevent midnight date rollover.
    """
    today    = today_et()  # ET date, not UTC
    new_rows = []
    for g in games_list:
        new_rows.append({
            "Date":             today,
            "Home Team":        g.get("Home Team", ""),
            "Away Team":        g.get("Away Team", ""),
            "Predicted Winner": g.get("Predicted Winner", ""),
            "Home Win %":       g.get("Home Win Probability", ""),
            "Confidence":       g.get("Confidence", ""),
            "Actual Winner":    "",
            # ML training features — used for monthly auto-recalibration
            "team_wpct":        g.get("team_wpct", ""),
            "opp_wpct":         g.get("opp_wpct", ""),
            "team_rpg":         g.get("team_rpg", ""),
            "opp_rpg":          g.get("opp_rpg", ""),
            "pitcher_era":      g.get("pitcher_era", ""),
            "opp_pitcher_era":  g.get("opp_pitcher_era", ""),
            "bullpen_era":      g.get("bullpen_era", ""),
            "opp_bullpen_era":  g.get("opp_bullpen_era", ""),
            "park_factor":      g.get("park_factor", ""),
            "temp":             g.get("temp", ""),
            "wind_speed":       g.get("wind_speed", ""),
            "wind_dir_out":     g.get("wind_dir_out", ""),
            "home":             1,
            "pitcher_recent_delta":     g.get("pitcher_recent_delta", 0.0),
            "opp_pitcher_recent_delta": g.get("opp_pitcher_recent_delta", 0.0),
        })
    new_df = pd.DataFrame(new_rows).reindex(columns=TRACKER_COLS, fill_value="")

    if os.path.exists(TRACKER_CSV):
        existing = pd.read_csv(TRACKER_CSV, dtype=str)
        existing = existing[existing["Date"] != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = _sanitize_dataframe(combined)
    combined.to_csv(TRACKER_CSV, index=False)
    return combined


# ============================================================
# C. FEATURES 1–6: AUTO-RESULTS, VEGAS, LINEUP, BULLPEN, FORM, RETRAIN
# ============================================================

def auto_fill_results():
    """
    Feature 1 — Auto-fill Results Tracker.
    Fetches final scores from MLB API and writes the actual winner
    into any unresolved rows. Backfills up to 7 past days to recover
    from Render restarts. Uses Eastern Time throughout.
    Also writes Correct? column and removes future game rows.
    """
    if not os.path.exists(TRACKER_CSV):
        return

    df = pd.read_csv(TRACKER_CSV, dtype=str)
    today = today_et()

    # Remove future game rows (dates after today ET)
    df = df[df["Date"] <= today]

    # Find all past dates with unresolved rows
    unresolved = df[
        df["Actual Winner"].isna() |
        (df["Actual Winner"].str.strip() == "") |
        (df["Actual Winner"].str.lower() == "nan")
    ]
    unresolved_dates = [d for d in unresolved["Date"].dropna().unique() if d < today]

    if not unresolved_dates:
        print(f"  [Tracker] All past rows resolved (today ET={today})")
        df.to_csv(TRACKER_CSV, index=False)
        return

    print(f"  [Tracker] Backfilling: {sorted(unresolved_dates)}")
    total_updated = 0

    for date_str in sorted(unresolved_dates):
        data = _get(f"{MLB_BASE}/schedule",
                    {"sportId": 1, "date": date_str, "hydrate": "linescore,team"})
        if not data:
            continue

        results = {}
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                state  = game.get("status", {}).get("abstractGameState", "")
                detail = game.get("status", {}).get("detailedState", "").lower()
                if state != "Final" or "postponed" in detail:
                    continue
                teams      = game.get("teams", {})
                home       = teams.get("home", {})
                away       = teams.get("away", {})
                home_name  = home.get("team", {}).get("name", "")
                away_name  = away.get("team", {}).get("name", "")
                home_score = home.get("score", 0) or 0
                away_score = away.get("score", 0) or 0
                if home_score == 0 and away_score == 0:
                    continue
                winner = home_name if home_score > away_score else away_name
                results[(home_name, away_name)] = winner

        updated = 0
        for idx, row in df.iterrows():
            if str(row.get("Date", "")).strip() != date_str:
                continue
            if str(row.get("Actual Winner", "")).strip() not in ("", "nan", "None"):
                continue
            key = (str(row.get("Home Team", "")).strip(),
                   str(row.get("Away Team", "")).strip())
            if key in results:
                actual    = results[key]
                predicted = str(row.get("Predicted Winner", "")).strip()
                df.at[idx, "Actual Winner"] = actual
                df.at[idx, "Correct?"]      = "✓ Correct" if actual == predicted else "✗ Wrong"
                updated += 1
                total_updated += 1

        print(f"    {date_str}: filled {updated} result(s) from {len(results)} final games")

    df.to_csv(TRACKER_CSV, index=False)
    print(f"  [Tracker] Done. Total updated: {total_updated}")

    # Sync resolved results to Supabase
    if total_updated:
        _sync_results_supabase(df)


def _sync_results_supabase(df):
    """Upsert resolved results from CSV into Supabase."""
    import urllib.request as _req
    sb_url = os.environ.get("SUPABASE_URL", "https://wkxpdmfabiepkfdxbdie.supabase.co")
    sb_key = os.environ.get("SUPABASE_KEY", "")

    def safe_float(v):
        try: return float(v) if v and str(v).strip() not in ('', 'nan', 'None') else None
        except: return None

    resolved = df[df['Actual Winner'].notna() & (df['Actual Winner'].str.strip() != '')]
    rows = []
    for _, r in resolved.iterrows():
        rows.append({
            'date':             str(r.get('Date','')).strip(),
            'home_team':        str(r.get('Home Team','')).strip(),
            'away_team':        str(r.get('Away Team','')).strip(),
            'predicted_winner': str(r.get('Predicted Winner','')).strip() or None,
            'actual_winner':    str(r.get('Actual Winner','')).strip() or None,
            'correct':          str(r.get('Correct?','')).strip() or None,
            'confidence':       str(r.get('Confidence','')).strip() or None,
            'model_edge':       safe_float(r.get('Model Edge')),
            'home_win_prob':    safe_float(r.get('Home Win %')),
            'game_time':        str(r.get('Game Time','')).strip() or None,
        })

    if not rows:
        return

    try:
        body = json.dumps(rows, default=str).encode()
        url = f"{sb_url}/rest/v1/results"
        request = _req.Request(url, data=body, method='POST')
        request.add_header('apikey', sb_key)
        # Authorization not needed for new sb_ keys
        request.add_header('Content-Type', 'application/json')
        request.add_header('Prefer', 'resolution=merge-duplicates,return=minimal')
        with _req.urlopen(request, timeout=15) as resp:
            print(f"  [Supabase] {len(rows)} results synced (HTTP {resp.status})")
    except Exception as e:
        print(f"  [Supabase] Results sync failed: {e}")


def _vig_free_prob(home_ml, away_ml):
    """
    Convert American moneylines to a vig-removed (true) home win probability.
    Removes the bookmaker's juice so both sides sum to 100%.
    """
    def to_raw(ml):
        return (100 / (ml + 100)) if ml > 0 else (abs(ml) / (abs(ml) + 100))
    h = to_raw(home_ml)
    a = to_raw(away_ml)
    total = h + a
    return h / total if total > 0 else 0.500


def fetch_vegas_odds():
    """
    Feature 2 — Live Vegas Odds (enhanced).
    Pulls today's MLB moneylines from The Odds API (free tier: 500 req/mo).
    Averages probabilities across ALL available bookmakers and applies
    vig removal so the implied probability is accurate, not inflated.
    Requires ODDS_API_KEY environment variable. Falls back gracefully.
    Returns dict keyed by (home_name, away_name).
    """
    global _vegas_cache
    if _vegas_cache:
        return _vegas_cache

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        print("  [Vegas] ODDS_API_KEY not set — win% fallback active (set it for live lines)")
        return {}

    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h",
                "oddsFormat":  "american",
                "dateFormat":  "iso",
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"  [Vegas] Odds API returned {r.status_code} — win% fallback active")
            return {}

        result = {}
        for game in r.json():
            home_team  = game.get("home_team", "")
            away_team  = game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            # Collect vig-free probs from every bookmaker for consensus accuracy
            probs    = []
            home_mls = []
            for bm in bookmakers:
                h2h = next((m for m in bm.get("markets", []) if m["key"] == "h2h"), None)
                if not h2h:
                    continue
                outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
                hml = outcomes.get(home_team)
                aml = outcomes.get(away_team)
                if hml is not None and aml is not None:
                    probs.append(_vig_free_prob(hml, aml))
                    home_mls.append(hml)

            if not probs:
                continue

            consensus_prob    = sum(probs) / len(probs)
            consensus_home_ml = round(sum(home_mls) / len(home_mls))
            result[(home_team, away_team)] = {
                "home_moneyline":  consensus_home_ml,
                "away_moneyline":  -consensus_home_ml,
                "home_prob_novig": round(consensus_prob, 4),
                "is_live":         True,
            }

        _vegas_cache = result
        print(f"  [Vegas] Live consensus odds loaded for {len(result)} games (vig removed)")
        return result

    except Exception as e:
        print(f"  [Vegas] WARNING: {e} — win% fallback active")
        return {}


def get_vegas_line(home_name, away_name, all_odds=None, home_wpct=0.500, away_wpct=0.500):
    """
    Return odds dict for one game.
    Priority 1 — Live consensus odds (vig-removed) from The Odds API.
    Priority 2 — Synthesized probability from team win% + home-field advantage
                 (much smarter than flat 50/50 when no API key is set).
    """
    if all_odds:
        if (home_name, away_name) in all_odds:
            return all_odds[(home_name, away_name)]
        for (h, a), odds in all_odds.items():
            if (home_name in h or h in home_name) and (away_name in a or a in away_name):
                return odds

    # Synthesized fallback: convert season win% + HFA to an implied probability
    total = (home_wpct + away_wpct) or 1.0
    home_prior     = home_wpct / total
    home_with_hfa  = min(max(home_prior + 0.035, 0.01), 0.99)
    if home_with_hfa >= 0.5:
        home_ml = round(-home_with_hfa / (1 - home_with_hfa) * 100)
    else:
        home_ml = round((1 - home_with_hfa) / home_with_hfa * 100)
    return {
        "home_moneyline":  home_ml,
        "away_moneyline":  -home_ml,
        "home_prob_novig": round(home_with_hfa, 4),
        "is_live":         False,
    }


def get_lineup_strength(team_id, team_name, fallback_strength):
    """
    Feature 3 — Starting Lineup Awareness.
    Fetches today's confirmed batting order from MLB API.
    Uses batch player lookup to get season OPS for each batter.
    Scales average OPS to a 1-10 lineup strength score.
    Falls back to team-average lineup_strength if lineup not yet posted.
    """
    global _lineup_cache
    if team_id in _lineup_cache:
        return _lineup_cache[team_id]

    today = today_et()
    data  = _get(f"{MLB_BASE}/schedule",
                 {"sportId": 1, "date": today, "hydrate": "lineups,team"})
    if not data:
        _lineup_cache[team_id] = fallback_strength
        return fallback_strength

    batting_order = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            for side in ("home", "away"):
                t = game.get("teams", {}).get(side, {})
                if t.get("team", {}).get("id") == team_id:
                    lineups    = game.get("lineups", {})
                    side_key   = "homePlayers" if side == "home" else "awayPlayers"
                    side_lineup = lineups.get(side_key, [])
                    if side_lineup:
                        batting_order = side_lineup
                    break

    if not batting_order:
        _lineup_cache[team_id] = fallback_strength
        return fallback_strength

    # Batch-fetch season hitting stats for the batting order
    player_ids = ",".join(str(p["id"]) for p in batting_order[:9] if p.get("id"))
    if not player_ids:
        _lineup_cache[team_id] = fallback_strength
        return fallback_strength

    try:
        pdata = _get(f"{MLB_BASE}/people",
                     {"personIds": player_ids,
                      "hydrate": f"stats(group=hitting,type=season,season={CURRENT_SEASON})"})
        ops_list = []
        for person in (pdata or {}).get("people", []):
            for stat_group in person.get("stats", []):
                for split in stat_group.get("splits", []):
                    ops_str = split.get("stat", {}).get("ops")
                    if ops_str:
                        ops_list.append(float(ops_str))
                        break

        if not ops_list:
            _lineup_cache[team_id] = fallback_strength
            return fallback_strength

        avg_ops  = sum(ops_list) / len(ops_list)
        # Scale: OPS 0.600 → 1, OPS 0.900 → 10
        strength = round(min(max((avg_ops - 0.600) / 0.300 * 10, 1), 10), 2)
        print(f"  [{team_name}] Live lineup: avg OPS={avg_ops:.3f} → strength={strength}")
        _lineup_cache[team_id] = strength
        return strength

    except Exception as e:
        print(f"  [{team_name}] Lineup lookup failed: {e}")
        _lineup_cache[team_id] = fallback_strength
        return fallback_strength


def get_bullpen_fatigue(team_id, team_name):
    """
    Feature 4 — Bullpen Fatigue Tracking.
    Sums non-starter pitch counts from the last 3 completed games.
    Returns a 0.0–1.0 fatigue score (higher = more tired bullpen).
    """
    global _bullpen_fatigue_cache
    if team_id in _bullpen_fatigue_cache:
        return _bullpen_fatigue_cache[team_id]

    today      = datetime.datetime.now(ET).date()
    start_date = (today - datetime.timedelta(days=4)).strftime("%Y-%m-%d")
    end_date   = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    data = _get(f"{MLB_BASE}/schedule", {
        "sportId": 1, "startDate": start_date, "endDate": end_date,
        "teamId": team_id, "hydrate": "boxscore,team",
    })

    if not data:
        _bullpen_fatigue_cache[team_id] = 0.35
        return 0.35

    total_bullpen_pitches = 0
    games_counted         = 0

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            boxscore = game.get("boxscore", {})
            for side in ("home", "away"):
                t = boxscore.get("teams", {}).get(side, {})
                if t.get("team", {}).get("id") != team_id:
                    continue
                pitchers = t.get("pitchers", [])
                players  = t.get("players", {})
                for pid in pitchers[1:]:          # skip starter (index 0)
                    pstats = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
                    total_bullpen_pitches += pstats.get("pitchesThrown", 0) or 0
                games_counted += 1

    if games_counted == 0:
        _bullpen_fatigue_cache[team_id] = 0.35
        return 0.35

    avg_pitches = total_bullpen_pitches / games_counted
    # 0 pitches/game → 0.0, 60+ pitches/game → 0.85
    fatigue = round(min(avg_pitches / 60, 1.0) * 0.85, 3)
    print(f"  [{team_name}] Bullpen: {avg_pitches:.0f} avg bp-pitches/game → fatigue={fatigue:.3f}")
    _bullpen_fatigue_cache[team_id] = fatigue
    return fatigue


def get_recent_pitcher_era(pitcher_id, season_era):
    """
    Feature 5 — Rolling Pitcher Form (last 5 starts).
    Computes ERA from the pitcher's most recent 5 game log entries,
    then returns a (blended_era, rolling_era) tuple.
    blended = 50/50 season + rolling for stability.
    rolling_era is returned separately so callers can compute the delta
    (rolling - season) as a trend feature for the ML model.
    """
    if not pitcher_id:
        return season_era, season_era

    try:
        data = _get(f"{MLB_BASE}/people/{pitcher_id}/stats",
                    {"stats": "gameLog", "group": "pitching",
                     "season": CURRENT_SEASON})
        if not data:
            return season_era, season_era

        splits = data.get("stats", [{}])[0].get("splits", [])
        recent = splits[-5:] if len(splits) >= 5 else splits
        if not recent:
            return season_era, season_era

        total_er, total_ip = 0, 0.0
        for game in recent:
            stat = game.get("stat", {})
            er   = stat.get("earnedRuns", 0) or 0
            ip_s = str(stat.get("inningsPitched", "0"))
            parts = ip_s.split(".")
            ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
            total_er += er
            total_ip += ip

        if total_ip < 3:
            return season_era, season_era

        rolling_era = round((total_er / total_ip) * 9, 2)
        blended     = round(season_era * 0.50 + rolling_era * 0.50, 2)
        return blended, rolling_era

    except Exception:
        return season_era, season_era


def retrain_model_if_needed():
    """
    Daily Learning — retrain whenever new resolved games have been added
    since the last training run (count-based, not time-based).
    Minimum of 10 resolved games to start learning.
    Saves the updated model to disk so future runs load instantly.
    """
    global ml_model

    if not os.path.exists(TRACKER_CSV):
        return

    df       = pd.read_csv(TRACKER_CSV, dtype=str)
    resolved = df[df["Actual Winner"].notna() & (df["Actual Winner"].str.strip() != "")]
    current_count = len(resolved)

    # Read last-retrain game count from flag file (format: "YYYY-MM-DD:count")
    last_count = 0
    if os.path.exists(RETRAIN_FLAG):
        try:
            content = open(RETRAIN_FLAG).read().strip()
            parts   = content.split(":")
            last_count = int(parts[1]) if len(parts) >= 2 else 0
        except Exception:
            last_count = 0

    if current_count < 10:
        print(f"  [ML] Daily learning: {current_count} resolved games (need ≥10 to start)")
        return
    if current_count <= last_count:
        print(f"  [ML] Daily learning: no new resolved games since last run ({last_count} games)")
        return

    print(f"  [ML] Daily learning: {current_count - last_count} new game(s) → retraining…")
    try:
        ml_feat_cols    = ["team_wpct", "opp_wpct", "team_rpg", "opp_rpg",
                           "pitcher_era", "opp_pitcher_era", "bullpen_era", "opp_bullpen_era",
                           "park_factor", "temp", "wind_speed", "wind_dir_out", "home",
                           "pitcher_recent_delta", "opp_pitcher_recent_delta"]
        available_feats = [c for c in ml_feat_cols if c in resolved.columns]
        if not available_feats:
            return

        live_rows = []
        for _, row in resolved.iterrows():
            try:
                feat           = {f: float(row.get(f, 0) or 0) for f in available_feats}
                feat["result"] = 1 if str(row["Actual Winner"]).strip() == str(row["Home Team"]).strip() else 0
                live_rows.append(feat)
            except Exception:
                continue

        if not live_rows:
            return

        live_df   = pd.DataFrame(live_rows)
        hist_path = os.path.join(SCRIPT_DIR, "mlb_games_history.csv")
        if os.path.exists(hist_path):
            hist        = pd.read_csv(hist_path)
            common_cols = list(set(live_df.columns) & set(hist.columns))
            combined    = pd.concat([hist[common_cols], live_df[common_cols]], ignore_index=True)
        else:
            combined = live_df

        combined = combined.fillna(0)
        features = combined.drop(columns=["result"])
        target   = combined["result"]

        # 3-way split: 70% train | 15% calibration | 15% holdout test
        X_tr, X_rest, y_tr, y_rest = train_test_split(features, target, test_size=0.30, random_state=42)
        X_cal, X_te, y_cal, y_te   = train_test_split(X_rest,   y_rest,  test_size=0.50, random_state=42)

        cal_rf  = _train_calibrated(
            RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
            X_tr, y_tr, X_cal, y_cal,
        )
        cal_xgb = _train_calibrated(
            XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=42, verbosity=0),
            X_tr, y_tr, X_cal, y_cal,
        )

        ensemble = {"rf": cal_rf, "xgb": cal_xgb, "feature_cols": list(features.columns)}
        ml_model = ensemble

        accuracy = sum(cal_rf.predict(X_te) == y_te) / len(y_te)

        with open(MODEL_CACHE, "wb") as f:
            pickle.dump(ensemble, f)

        with open(RETRAIN_FLAG, "w") as f:
            f.write(f"{today_et()}:{current_count}")

        print(f"  [ML] Ensemble updated (RF+XGB calibrated) — {accuracy:.1%} holdout accuracy "
              f"({current_count} live + {len(combined) - current_count} historical games)")

    except Exception as e:
        print(f"  [ML] Retrain failed: {e}")


def get_team_stats(team_id, team_name="Team", savant_batting=None):
    """Fetch live team stats: win pct, runs per game, bullpen ERA, batting avg."""
    if team_id in _team_stats_cache:
        return _team_stats_cache[team_id]

    defaults = {
        "wpct": 0.500, "rpg": 4.5, "bullpen_era": 4.00,
        "hard_hit": 0.38, "defense": 5.0,
        "lineup_strength": 5.0, "vs_hand_split": 0.0
    }

    standings = fetch_standings()
    if team_id in standings:
        defaults["wpct"] = standings[team_id]["wpct"]
        defaults["rpg"] = standings[team_id]["rpg"]

    # Team pitching ERA (used as overall pitching proxy incl. bullpen)
    pitch_data = _get(f"{MLB_BASE}/teams/{team_id}/stats",
                      params={"stats": "season", "group": "pitching", "season": CURRENT_SEASON})
    if pitch_data:
        splits = pitch_data.get("stats", [{}])[0].get("splits", [])
        if splits:
            era_str = splits[0].get("stat", {}).get("era", None)
            if era_str:
                defaults["bullpen_era"] = float(era_str)

    # Team hitting avg (proxy for lineup strength)
    hit_data = _get(f"{MLB_BASE}/teams/{team_id}/stats",
                    params={"stats": "season", "group": "hitting", "season": CURRENT_SEASON})
    if hit_data:
        splits = hit_data.get("stats", [{}])[0].get("splits", [])
        if splits:
            stat = splits[0].get("stat", {})
            avg_str = stat.get("avg", None)
            if avg_str:
                avg = float(avg_str)
                # Scale batting avg to lineup_strength 1-10
                defaults["lineup_strength"] = round(min(max((avg - 0.200) / (0.290 - 0.200) * 10, 1), 10), 2)

    # Replace placeholder hard_hit with real Baseball Savant data if available
    if savant_batting and team_name in savant_batting:
        sv = savant_batting[team_name]
        defaults["hard_hit"]    = sv["hard_hit"]
        defaults["barrel_rate"] = sv["barrel_rate"]
        defaults["xwoba_off"]   = sv["xwoba"]

    print(f"  [{team_name}] wpct={defaults['wpct']:.3f}, rpg={defaults['rpg']}, bullpen_era={defaults['bullpen_era']}")
    _team_stats_cache[team_id] = defaults
    return defaults


def get_pitcher_stats(pitcher_id, pitcher_name="TBD"):
    """Fetch live pitcher ERA with home/away splits. Falls back to overall ERA."""
    if not pitcher_id or pitcher_name == "TBD":
        return {"home_era": 4.00, "away_era": 4.50}

    if pitcher_id in _pitcher_stats_cache:
        return _pitcher_stats_cache[pitcher_id]

    result = {"home_era": 4.00, "away_era": 4.50}

    # Overall season ERA as base
    season_data = _get(f"{MLB_BASE}/people/{pitcher_id}/stats",
                       params={"stats": "season", "group": "pitching", "season": CURRENT_SEASON})
    if season_data:
        splits = (season_data.get("stats") or [{}])[0].get("splits", [])
        if splits:
            era_str = splits[0].get("stat", {}).get("era")
            if era_str:
                overall_era = float(era_str)
                result["home_era"] = overall_era
                result["away_era"] = overall_era

    # Home/away splits
    split_data = _get(f"{MLB_BASE}/people/{pitcher_id}/stats",
                      params={"stats": "homeAndAway", "group": "pitching", "season": CURRENT_SEASON})
    if split_data:
        for split in (split_data.get("stats") or [{}])[0].get("splits", []):
            era_str = split.get("stat", {}).get("era")
            if not era_str:
                continue
            is_home = split.get("isHome", None)
            split_code = split.get("split", {}).get("code", "")
            if is_home is True or split_code == "H":
                result["home_era"] = float(era_str)
            elif is_home is False or split_code == "A":
                result["away_era"] = float(era_str)

    # Feature 5 — blend each split ERA with rolling last-5-start ERA (50/50)
    # Also capture the raw rolling ERA to compute a trend delta for the ML model
    season_home_era = result["home_era"]
    season_away_era = result["away_era"]
    result["home_era"], home_rolling = get_recent_pitcher_era(pitcher_id, season_home_era)
    result["away_era"], away_rolling = get_recent_pitcher_era(pitcher_id, season_away_era)
    # Positive delta = pitcher struggling recently vs season avg; negative = hot streak
    result["home_recent_delta"] = round(home_rolling - season_home_era, 2)
    result["away_recent_delta"] = round(away_rolling - season_away_era, 2)

    # Blend ERA with Baseball Savant xwOBA for a more predictive quality metric
    sv = get_savant_pitcher_stats(pitcher_id)
    if sv:
        result["home_era"] = xwoba_to_era_adj(sv["xwoba"], result["home_era"])
        result["away_era"] = xwoba_to_era_adj(sv["xwoba"], result["away_era"])
        result["xwoba"]    = sv["xwoba"]
        print(f"  [{pitcher_name}] home_era={result['home_era']}, away_era={result['away_era']}  (xwOBA={sv['xwoba']:.3f})")
    else:
        print(f"  [{pitcher_name}] home_era={result['home_era']}, away_era={result['away_era']}")
    _pitcher_stats_cache[pitcher_id] = result
    return result


def fetch_todays_schedule():
    """
    Fetch today's MLB schedule with probable pitchers and team IDs.
    Returns a list of game dicts.
    """
    today = today_et()
    print(f"Fetching MLB schedule for {today}...")
    data = _get(f"{MLB_BASE}/schedule",
                params={"sportId": 1, "date": today,
                        "hydrate": "probablePitcher,team,venue"})
    if not data:
        return None

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home_team = game["teams"]["home"]["team"]
            away_team = game["teams"]["away"]["team"]
            home_prob = game["teams"]["home"].get("probablePitcher", {})
            away_prob = game["teams"]["away"].get("probablePitcher", {})

            games.append({
                "home_name": home_team.get("name", "Home Team"),
                "home_id": home_team.get("id"),
                "away_name": away_team.get("name", "Away Team"),
                "away_id": away_team.get("id"),
                "venue": game.get("venue", {}).get("name", "Unknown Park"),
                "home_pitcher_id": home_prob.get("id"),
                "home_pitcher_name": home_prob.get("fullName", "TBD"),
                "away_pitcher_id": away_prob.get("id"),
                "away_pitcher_name": away_prob.get("fullName", "TBD"),
                "game_date": game.get("gameDate", ""),
            })

    return games if games else None


# ============================================================
# B. WEATHER — Live via Open-Meteo (free, no API key required)
# ============================================================

# (latitude, longitude, CF bearing degrees from north, indoor/retractable)
BALLPARK_COORDS = {
    "Yankee Stadium":           (40.8296,  -73.9262,  40,  False),
    "Fenway Park":              (42.3467,  -71.0972,  65,  False),
    "Wrigley Field":            (41.9484,  -87.6553, 315,  False),
    "Dodger Stadium":           (34.0739, -118.2400,  10,  False),
    "Oracle Park":              (37.7786, -122.3893,  60,  False),
    "Chase Field":              (33.4453, -112.0667, 340,  True),
    "T-Mobile Park":            (47.5914, -122.3324, 315,  False),
    "Camden Yards":             (39.2838,  -76.6218,   5,  False),
    "Nationals Park":           (38.8730,  -77.0074,   5,  False),
    "Truist Park":              (33.8907,  -84.4678, 315,  False),
    "American Family Field":    (43.0280,  -87.9712,  20,  True),
    "Minute Maid Park":         (29.7573,  -95.3555, 280,  True),
    "Angel Stadium":            (33.8003, -117.8827, 350,  False),
    "Petco Park":               (32.7077, -117.1569, 330,  False),
    "Busch Stadium":            (38.6226,  -90.1928,  10,  False),
    "Great American Ball Park": (39.0974,  -84.5080,  40,  False),
    "Progressive Field":        (41.4962,  -81.6852, 345,  False),
    "Kauffman Stadium":         (39.0517,  -94.4803, 350,  False),
    "Target Field":             (44.9817,  -93.2781, 300,  False),
    "Comerica Park":            (42.3390,  -83.0485, 345,  False),
    "Globe Life Field":         (32.7474,  -97.0828,  20,  True),
    "Coors Field":              (39.7559, -104.9942, 340,  False),
    "loanDepot park":           (25.7781,  -80.2197, 325,  True),
    "Citizens Bank Park":       (39.9057,  -75.1665,   5,  False),
    "Citi Field":               (40.7571,  -73.8458, 330,  False),
    "PNC Park":                 (40.4469,  -80.0057, 350,  False),
    "Guaranteed Rate Field":    (41.8300,  -87.6339, 350,  False),
    "Oakland Coliseum":         (37.7516, -122.2005, 320,  False),
    "Tropicana Field":          (27.7683,  -82.6534,   0,  True),
    "Rogers Centre":            (43.6414,  -79.3894,   0,  True),
    "Sutter Health Park":       (38.5802, -121.5002, 350,  False),
}


def _angle_diff(a, b):
    """Smallest angular difference between two compass bearings."""
    return abs((a - b + 180) % 360 - 180)


def _wind_category(wind_deg, cf_bearing, indoor):
    """
    Classify wind as 'out' (helps hitters), 'in' (hurts), or 'neutral'.
    Meteorological convention: wind_deg is the direction the wind is COMING FROM.
    Wind blows OUT when it comes from behind home plate (≈ cf_bearing + 180°).
    Wind blows IN  when it comes from center field (≈ cf_bearing).
    """
    if indoor:
        return "neutral"
    out_source = (cf_bearing + 180) % 360
    if _angle_diff(wind_deg, out_source) < 45:
        return "out"
    if _angle_diff(wind_deg, cf_bearing) < 45:
        return "in"
    return "neutral"


def get_weather(ballpark, game_date_utc=""):
    """
    Fetch live ballpark weather from Open-Meteo (free, no API key required).
    Returns real temperature, wind speed/direction (relative to CF), and
    rain probability at game start time and mid-game (+3 h).
    Falls back to league-average conditions if the park is unknown.
    """
    global _weather_cache
    cache_key = f"{ballpark}:{game_date_utc[:13]}"
    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    default = {
        "temp_start": 70, "temp_mid": 68,
        "wind_start_speed": 7,  "wind_start_dir": "neutral",
        "wind_mid_speed":   6,  "wind_mid_dir":   "neutral",
        "rain_chance": 10,
    }

    coords = BALLPARK_COORDS.get(ballpark)
    if not coords:
        _weather_cache[cache_key] = default
        return default

    lat, lon, cf_bearing, indoor = coords

    # Parse game start hour (UTC); default to 23:00 (≈ 7 PM ET)
    game_hour_utc = 23
    if game_date_utc:
        try:
            dt = datetime.datetime.strptime(game_date_utc, "%Y-%m-%dT%H:%M:%SZ")
            game_hour_utc = dt.hour
        except Exception:
            pass

    try:
        today = today_et()
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":         lat,
                "longitude":        lon,
                "hourly":           "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "windspeed_unit":   "mph",
                "timezone":         "UTC",
                "start_date":       today,
                "end_date":         today,
            },
            timeout=10,
        )
        d      = r.json()
        hours  = d.get("hourly", {})
        times  = hours.get("time", [])
        temps  = hours.get("temperature_2m", [])
        winds  = hours.get("windspeed_10m", [])
        wdirs  = hours.get("winddirection_10m", [])
        precs  = hours.get("precipitation_probability", [])

        # Map UTC hours to list indices
        start_idx = next(
            (i for i, t in enumerate(times) if f"T{game_hour_utc:02d}:00" in t),
            min(game_hour_utc, len(times) - 1)
        )
        mid_game_hour = (game_hour_utc + 3) % 24
        mid_idx = next(
            (i for i, t in enumerate(times) if f"T{mid_game_hour:02d}:00" in t),
            min(start_idx + 3, len(times) - 1)
        )

        def _v(lst, idx, fallback):
            return lst[idx] if lst and idx < len(lst) else fallback

        if indoor:
            t = _v(temps, start_idx, 72)
            result = {
                "temp_start": t, "temp_mid": t,
                "wind_start_speed": 0, "wind_start_dir": "neutral",
                "wind_mid_speed":   0, "wind_mid_dir":   "neutral",
                "rain_chance": 0,
            }
        else:
            ws  = _v(winds, start_idx, 7);   wm  = _v(winds, mid_idx, 6)
            wds = _v(wdirs, start_idx, 180);  wdm = _v(wdirs, mid_idx, 180)
            result = {
                "temp_start":       round(_v(temps, start_idx, 70), 1),
                "temp_mid":         round(_v(temps, mid_idx,   68), 1),
                "wind_start_speed": round(ws, 1),
                "wind_start_dir":   _wind_category(wds, cf_bearing, indoor),
                "wind_mid_speed":   round(wm, 1),
                "wind_mid_dir":     _wind_category(wdm, cf_bearing, indoor),
                "rain_chance":      _v(precs, start_idx, 10),
            }

        tag = "indoor/dome" if indoor else f"{result['wind_start_speed']}mph {result['wind_start_dir']}"
        print(f"  [{ballpark}] Weather: {result['temp_start']}°F, wind {tag}, rain {result['rain_chance']}%")
        _weather_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"  [Weather] {ballpark} API error: {e} — using defaults")
        _weather_cache[cache_key] = default
        return default

def get_park_factor(ballpark):
    park_factors = {
        "Coors Field": 1.30,
        "Great American Ball Park": 1.15,
        "Yankee Stadium": 1.08,
        "Dodger Stadium": 0.96,
        "Oracle Park": 0.92,
        "Petco Park": 0.90,
        "T-Mobile Park": 0.93,
    }
    return park_factors.get(ballpark, 1.00)

def get_umpire_factor(umpire):
    return 0

def get_travel(team):
    return 300


# ============================================================
# C. ADVANCED MODEL
# ============================================================

def advanced_model(inputs):
    (
        team_wpct, opp_wpct,
        team_rpg, opp_rpg,
        pitcher_home_era, pitcher_away_era,
        opp_pitcher_home_era, opp_pitcher_away_era,
        bullpen_era, opp_bullpen_era,
        park_factor,
        umpire_favor,
        travel_miles, opp_travel_miles,
        bullpen_usage, opp_bullpen_usage,
        is_home,
        temp_start, temp_mid,
        wind_start_speed, wind_start_dir,
        wind_mid_speed, wind_mid_dir,
        rain_chance,
        team_hard_hit, opp_hard_hit,
        team_defense, opp_defense,
        team_lineup_strength, opp_lineup_strength,
        team_vs_hand_split, opp_vs_hand_split
    ) = inputs

    home_adv = 0.04 if is_home else 0

    team_pitcher_era = pitcher_home_era if is_home else pitcher_away_era
    opp_pitcher_era = opp_pitcher_home_era if not is_home else opp_pitcher_away_era
    era_diff = (opp_pitcher_era - team_pitcher_era) / 10

    bullpen_diff = (opp_bullpen_era - bullpen_era) / 10
    bullpen_fatigue = (opp_bullpen_usage - bullpen_usage) * 0.5

    park_adj = (park_factor - 1.00) * 0.5
    umpire_adj = umpire_favor * 0.03
    travel_adj = (opp_travel_miles - travel_miles) / 3000

    def weather_effect(temp, wind_speed, wind_dir):
        effect = 0
        if temp >= 85:
            effect += 0.05
        elif temp <= 55:
            effect -= 0.05
        if wind_dir == "out":
            effect += wind_speed * 0.01
        elif wind_dir == "in":
            effect -= wind_speed * 0.01
        return effect

    weather_avg = (
        weather_effect(temp_start, wind_start_speed, wind_start_dir) +
        weather_effect(temp_mid, wind_mid_speed, wind_mid_dir)
    ) / 2

    rain_adj = -rain_chance * 0.001
    wpct_diff = team_wpct - opp_wpct
    rpg_diff = (team_rpg - opp_rpg) / 10
    hard_hit_diff = (team_hard_hit - opp_hard_hit) * 0.5
    defense_diff = (team_defense - opp_defense) * 0.5
    lineup_diff = (team_lineup_strength - opp_lineup_strength) * 0.7
    split_diff = (team_vs_hand_split - opp_vs_hand_split) * 0.6

    score = (
        wpct_diff * 2.0 +
        era_diff * 1.8 +
        bullpen_diff * 1.2 +
        bullpen_fatigue +
        rpg_diff * 1.2 +
        home_adv +
        park_adj +
        umpire_adj +
        travel_adj +
        weather_avg +
        rain_adj +
        hard_hit_diff +
        defense_diff +
        lineup_diff +
        split_diff
    )

    return 1 / (1 + math.exp(-score))


# ============================================================
# D. MACHINE LEARNING MODEL
# ============================================================

def load_ml_model():
    """
    Load the RF + XGBoost calibrated ensemble.
    Checks for a cached pickle first (fast, includes any daily retraining).
    Falls back to training from the historical CSV if no cache exists.
    Returns a dict {"rf": calibrated_rf, "xgb": calibrated_xgb, "feature_cols": [...]}
    or None if no data is available.
    """
    if os.path.exists(MODEL_CACHE):
        try:
            with open(MODEL_CACHE, "rb") as f:
                model = pickle.load(f)
            # Accept both old single-model cache and new ensemble dict
            if isinstance(model, dict) and "rf" in model:
                print("  [ML] Loaded cached RF+XGB ensemble from disk")
                return model
            # Old cache format — discard and retrain below
        except Exception:
            pass

    try:
        data = pd.read_csv(os.path.join(SCRIPT_DIR, "mlb_games_history.csv")).fillna(0)
    except Exception:
        return None

    features = data.drop(columns=["result"])
    target   = data["result"]

    # 3-way split: 70% train | 15% calibration | 15% holdout test
    X_tr, X_rest, y_tr, y_rest = train_test_split(features, target, test_size=0.30, random_state=42)
    X_cal, X_te, y_cal, y_te  = train_test_split(X_rest,   y_rest,  test_size=0.50, random_state=42)

    print("  [ML] Training RF + XGBoost ensemble on historical data…")
    cal_rf  = _train_calibrated(
        RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
        X_tr, y_tr, X_cal, y_cal,
    )
    cal_xgb = _train_calibrated(
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      use_label_encoder=False, eval_metric="logloss",
                      random_state=42, verbosity=0),
        X_tr, y_tr, X_cal, y_cal,
    )

    ensemble = {"rf": cal_rf, "xgb": cal_xgb, "feature_cols": list(features.columns)}

    accuracy = sum(cal_rf.predict(X_te) == y_te) / len(y_te)
    print(f"  [ML] Ensemble ready — {accuracy:.1%} holdout accuracy ({len(data)} games)")

    try:
        with open(MODEL_CACHE, "wb") as f:
            pickle.dump(ensemble, f)
    except Exception:
        pass

    return ensemble

ml_model = load_ml_model()


# ============================================================
# FINAL COMBINED PROBABILITY
# ============================================================

def final_win_probability(adv_prob, ml_prob, vegas_prob=None, vegas_is_live=False):
    """
    Blend three signal sources.
    Live (vig-removed) Vegas odds get 16% weight; synthesized win%-based
    odds get 10%.  The two models absorb the remaining weight equally.
    """
    w_vegas = (0.16 if vegas_is_live else 0.10) if vegas_prob is not None else 0.0
    w_each  = (1.0 - w_vegas) / 2.0

    combined = (
        adv_prob   * w_each +
        ml_prob    * w_each +
        (vegas_prob * w_vegas if vegas_prob is not None else 0)
    )
    return round(combined * 100, 2)


# ============================================================
# MOBILE APP JSON EXPORT
# ============================================================

_MOBILE_KEYS = [
    "Game Time", "Home Team", "Away Team", "Venue", "Weather",
    "Home Starting Pitcher", "Away Starting Pitcher",
    "Home Win %", "Away Win %",
    "Home Runs / Game", "Away Runs / Game",
    "Home Bullpen ERA", "Away Bullpen ERA",
    "Home SP ERA (at Home)", "Away SP ERA (on Road)",
    "Park Factor", "Adv Model %", "ML Model %",
    "Vegas Implied %", "Model Edge",
    "Home Win Probability", "Predicted Winner", "Confidence",
]


def _save_predictions_json(games_list):
    """Export today's game predictions as JSON for the mobile app."""
    today_str = today_et()

    def clean(g):
        return {k: g.get(k) for k in _MOBILE_KEYS}

    conf_rank = {"High": 0, "Medium": 1}
    best_bets = sorted(
        [g for g in games_list if g.get("Confidence") in ("High", "Medium")],
        key=lambda g: (conf_rank.get(g.get("Confidence", ""), 99),
                       -abs(g.get("Home Win Probability", 50))),
    )[:6]

    payload = {
        "date":         today_str,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_bets":    [clean(g) for g in best_bets],
        "games":        [clean(g) for g in games_list],
    }
    with open(PREDICTIONS_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Predictions JSON saved → {PREDICTIONS_JSON}")

    # Also write to Supabase so Netlify functions can serve it even when Render is asleep
    _save_predictions_supabase(payload)


def _save_predictions_supabase(payload):
    """Write today's predictions to Supabase predictions table."""
    import urllib.request as _req
    sb_url = os.environ.get("SUPABASE_URL", "https://wkxpdmfabiepkfdxbdie.supabase.co")
    sb_key = os.environ.get("SUPABASE_KEY", "")
    try:
        body = json.dumps({
            "date":         payload["date"],
            "games":        payload["games"],
            "best_bets":    payload["best_bets"],
            "generated_at": payload["generated_at"],
        }, default=str).encode()
        url = f"{sb_url}/rest/v1/predictions"
        request = _req.Request(url, data=body, method="POST")
        request.add_header("apikey", sb_key)
        request.add_header("Authorization", f"Bearer {sb_key}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Prefer", "resolution=merge-duplicates,return=minimal")
        with _req.urlopen(request, timeout=10) as resp:
            print(f"  [Supabase] Predictions cached for {payload['date']} (HTTP {resp.status})")
    except Exception as e:
        print(f"  [Supabase] Predictions cache failed: {e}")


def _save_tracker_json(tracker_df):
    """Export tracker data as JSON for the mobile app (display columns only)."""
    display_cols = [
        "Date", "Game Time", "Home Team", "Away Team",
        "Predicted Winner", "Home Win %", "Model Edge",
        "Confidence", "Actual Winner", "Correct?",
    ]
    cols = [c for c in display_cols if c in tracker_df.columns]
    df   = tracker_df[cols].copy()

    records = df.to_dict(orient="records")
    records = [
        {k: (None if str(v) == "nan" else v) for k, v in r.items()}
        for r in records
    ]
    records.reverse()   # most-recent first

    def tier(conf_name):
        tier_recs = [r for r in records if r.get("Confidence") == conf_name]
        resolved  = [r for r in tier_recs
                     if r.get("Correct?") and str(r["Correct?"]) not in ("", "None")]
        correct   = sum(1 for r in resolved if "Correct" in str(r.get("Correct?", "")))
        return {
            "total":   len(resolved),
            "correct": correct,
            "pct":     round(correct / len(resolved) * 100) if resolved else 0,
        }

    resolved = [r for r in records
                if r.get("Correct?") and str(r["Correct?"]) not in ("", "None")]
    correct  = sum(1 for r in resolved if "Correct" in str(r.get("Correct?", "")))

    payload = {
        "records": records,
        "summary": {
            "total":   len(resolved),
            "correct": correct,
            "pct":     round(correct / len(resolved) * 100) if resolved else 0,
            "high":    tier("High"),
            "medium":  tier("Medium"),
            "low":     tier("Low"),
        },
    }
    with open(TRACKER_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Tracker JSON saved → {TRACKER_JSON}")


# ============================================================
# DAILY REPORT GENERATION
# ============================================================

def save_daily_report(games_list):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "mlb_daily_report.xlsx")
    today_str   = datetime.datetime.now(ET).strftime("%B %d, %Y")

    # ── Colour palette ──────────────────────────────────────────
    NAVY         = "1B3A6B"
    WHITE        = "FFFFFF"
    ROW_ODD      = "F4F6FA"
    ROW_EVEN     = "FFFFFF"
    MID_GRAY     = "C8CDD8"
    BEST_BET_BG  = "FFF8E1"   # warm gold tint for Best Bets rows
    EDGE_POS     = "1A6B3A"   # strong positive edge — dark green
    EDGE_NEG     = "8B1A1A"   # negative edge — dark red
    # Section header accent colours
    HDR_GAME     = "2C4F8A"
    HDR_PITCHER  = "1A6B50"
    HDR_STATS    = "5A3A8A"
    HDR_MODEL    = "7A4A00"
    HDR_PRED     = "1A5A1A"
    HDR_EDGE     = "8B4500"
    # Confidence
    GREEN_DARK   = "1A6B3A";  GREEN_LIGHT  = "D6F0E0"
    YELLOW_DARK  = "7A6000";  YELLOW_FILL  = "FFF3C4"
    RED_DARK     = "8B1A1A";  RED_FILL     = "FFD6D6"
    WINNER_FILL  = "C6EFCE"
    PROB_HIGH    = "C6EFCE"
    PROB_MID     = "DBEAFE"
    PROB_LOW     = "FFCCCC"

    thin   = Side(style="thin",   color=MID_GRAY)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def fill(c):  return PatternFill("solid", fgColor=c)
    def al(h="center", v="center", wrap=False):
        return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

    # ── Column definitions ──────────────────────────────────────
    # (key, display header, alignment, width, header-colour)
    # Widths are calibrated to the longest realistic data value in each column.
    columns = [
        ("Game Time",             "Game\nTime",             "center", 13, HDR_GAME),
        ("Home Team",             "Home Team",              "left",   22, HDR_GAME),
        ("Away Team",             "Away Team",              "left",   22, HDR_GAME),
        ("Venue",                 "Venue",                  "left",   27, HDR_GAME),
        ("Weather",               "Weather",                "left",   32, HDR_GAME),
        ("Home Starting Pitcher", "Home Starter",           "left",   22, HDR_PITCHER),
        ("Away Starting Pitcher", "Away Starter",           "left",   22, HDR_PITCHER),
        ("Home Win %",            "Home\nWin %",            "center", 11, HDR_STATS),
        ("Away Win %",            "Away\nWin %",            "center", 11, HDR_STATS),
        ("Home Runs / Game",      "Home\nRuns/G",           "center", 10, HDR_STATS),
        ("Away Runs / Game",      "Away\nRuns/G",           "center", 10, HDR_STATS),
        ("Home Bullpen ERA",      "Home\nBull ERA",         "center", 12, HDR_STATS),
        ("Away Bullpen ERA",      "Away\nBull ERA",         "center", 12, HDR_STATS),
        ("Home SP ERA (at Home)", "Home SP\nERA",           "center", 11, HDR_PITCHER),
        ("Away SP ERA (on Road)", "Away SP\nERA",           "center", 11, HDR_PITCHER),
        ("Park Factor",           "Park\nFactor",           "center", 10, HDR_GAME),
        ("Adv Model %",           "Adv\nModel %",           "center", 12, HDR_MODEL),
        ("ML Model %",            "ML\nModel %",            "center", 12, HDR_MODEL),
        ("Vegas Implied %",       "Vegas\nImplied %",       "center", 13, HDR_MODEL),
        ("Model Edge",            "Model\nEdge",            "center", 12, HDR_EDGE),
        ("Home Win Probability",  "Home Win\nProbability",  "center", 15, HDR_PRED),
        ("Predicted Winner",      "Predicted Winner",       "left",   22, HDR_PRED),
        ("Confidence",            "Confidence",             "center", 13, HDR_PRED),
    ]
    num_cols = len(columns)
    last_col = get_column_letter(num_cols)

    wb = Workbook()
    ws = wb.active
    ws.title = "MLB Predictions"

    # ── Identify best bets (High or Medium confidence) ──
    _conf_rank = {"High": 0, "Medium": 1}
    best_bets = sorted(
        [g for g in games_list if g.get("Confidence") in ("High", "Medium")],
        key=lambda g: (_conf_rank.get(g.get("Confidence", ""), 99),
                       -abs(g.get("Home Win Probability", 50))),
    )[:6]
    best_bet_keys = {(g["Home Team"], g["Away Team"]) for g in best_bets}

    current_row = 1

    # ── Title row ───────────────────────────────────────────────
    ws.merge_cells(f"A{current_row}:{last_col}{current_row}")
    tc = ws[f"A{current_row}"]
    tc.value     = f"⚾   MLB Daily Prediction Report   —   {today_str}"
    tc.font      = Font(color=WHITE, bold=True, size=15, name="Calibri")
    tc.fill      = fill(NAVY)
    tc.alignment = al("center")
    ws.row_dimensions[current_row].height = 36
    current_row += 1

    # ── Best Bets section ────────────────────────────────────────
    GOLD         = "B8860B"
    GOLD_LIGHT   = "FFF8DC"
    GOLD_BORDER  = "DAA520"
    BEST_HDR_COL = "8B6914"

    if best_bets:
        # Section header
        ws.merge_cells(f"A{current_row}:{last_col}{current_row}")
        bh = ws[f"A{current_row}"]
        bh.value     = "★  TODAY'S BEST BETS  —  Highest model edge with model agreement"
        bh.font      = Font(color=WHITE, bold=True, size=11, name="Calibri")
        bh.fill      = fill(GOLD)
        bh.alignment = al("left")
        ws.row_dimensions[current_row].height = 22
        current_row += 1

        # Best bet sub-headers
        bb_cols = [
            ("Game Time", "Time",      9),
            ("Away Team", "Away",     20),
            ("Home Team", "Home",     20),
            ("Predicted Winner", "Pick",     20),
            ("Home Win Probability", "Win Prob", 11),
            ("Model Edge", "Edge",     10),
            ("Vegas Implied %", "Vegas %",  10),
            ("Confidence", "Confidence", 12),
            ("Weather", "Weather",   22),
        ]
        for ci, (_, hdr, w) in enumerate(bb_cols, start=1):
            cell = ws.cell(row=current_row, column=ci, value=hdr)
            cell.font      = Font(color=WHITE, bold=True, size=9, name="Calibri")
            cell.fill      = fill(BEST_HDR_COL)
            cell.alignment = al("center", wrap=True)
            cell.border    = Border(
                left=Side(style="thin", color=GOLD_BORDER),
                right=Side(style="thin", color=GOLD_BORDER),
                top=Side(style="thin", color=GOLD_BORDER),
                bottom=Side(style="thin", color=GOLD_BORDER),
            )
            ws.column_dimensions[get_column_letter(ci)].width = w
        ws.row_dimensions[current_row].height = 22
        current_row += 1

        for g in best_bets:
            edge      = g.get("Model Edge", 0)
            edge_str  = f"+{edge}%" if edge >= 0 else f"{edge}%"
            conf      = g.get("Confidence", "")
            conf_bg   = GREEN_LIGHT if conf == "High" else YELLOW_FILL
            conf_fg   = GREEN_DARK  if conf == "High" else YELLOW_DARK
            bb_vals   = [
                g.get("Game Time", ""),
                g.get("Away Team", ""),
                g.get("Home Team", ""),
                g.get("Predicted Winner", ""),
                f"{g.get('Home Win Probability', '')}%",
                edge_str,
                f"{g.get('Vegas Implied %', '')}%",
                conf,
                g.get("Weather", ""),
            ]
            for ci, val in enumerate(bb_vals, start=1):
                cell = ws.cell(row=current_row, column=ci, value=sanitize_cell(val))
                cell.border    = Border(
                    left=Side(style="thin", color=GOLD_BORDER),
                    right=Side(style="thin", color=GOLD_BORDER),
                    top=Side(style="thin", color=GOLD_BORDER),
                    bottom=Side(style="thin", color=GOLD_BORDER),
                )
                cell.alignment = al("center" if ci not in (2, 3, 4, 9) else "left")
                if ci == 4:
                    cell.fill = fill(WINNER_FILL)
                    cell.font = Font(name="Calibri", size=10, bold=True, color=GREEN_DARK)
                elif ci == 6:
                    cell.fill = fill(GREEN_LIGHT if edge >= 0 else RED_FILL)
                    cell.font = Font(name="Calibri", size=10, bold=True,
                                     color=EDGE_POS if edge >= 0 else EDGE_NEG)
                elif ci == 8:
                    cell.fill = fill(conf_bg)
                    cell.font = Font(name="Calibri", size=10, bold=True, color=conf_fg)
                else:
                    cell.fill = fill(GOLD_LIGHT)
                    cell.font = Font(name="Calibri", size=10)
            ws.row_dimensions[current_row].height = 20
            current_row += 1

        # Spacer
        ws.merge_cells(f"A{current_row}:{last_col}{current_row}")
        ws.row_dimensions[current_row].height = 8
        current_row += 1

    # ── All-games section header ─────────────────────────────────
    ws.merge_cells(f"A{current_row}:{last_col}{current_row}")
    ah = ws[f"A{current_row}"]
    ah.value     = "ALL GAMES TODAY"
    ah.font      = Font(color=WHITE, bold=True, size=10, name="Calibri")
    ah.fill      = fill(NAVY)
    ah.alignment = al("left")
    ws.row_dimensions[current_row].height = 18
    current_row += 1

    # ── Column headers ───────────────────────────────────────────
    hdr_row = current_row
    for ci, (key, header, align, width, hdr_color) in enumerate(columns, start=1):
        cell = ws.cell(row=hdr_row, column=ci, value=header)
        cell.font      = Font(color=WHITE, bold=True, size=9, name="Calibri")
        cell.fill      = fill(hdr_color)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = Border(left=thin, right=thin,
                                top=Side(style="medium", color=WHITE),
                                bottom=Side(style="medium", color=WHITE))
        ws.column_dimensions[get_column_letter(ci)].width = width
    ws.row_dimensions[hdr_row].height = 32
    current_row += 1

    data_start_row = current_row

    # Number formats per column key
    NUM_FMT = {
        "Home Win %":            "0.0%",    # stored as 0.684 → displays 68.4%
        "Away Win %":            "0.0%",
        "Home Runs / Game":      "0.0",
        "Away Runs / Game":      "0.0",
        "Home Bullpen ERA":      "0.00",
        "Away Bullpen ERA":      "0.00",
        "Home SP ERA (at Home)": "0.00",
        "Away SP ERA (on Road)": "0.00",
        "Park Factor":           "0.00",
        "Adv Model %":           '0.0"%"',  # stored as 82.44 → displays 82.4%
        "ML Model %":            '0.0"%"',
        "Vegas Implied %":       '0.0"%"',
        "Home Win Probability":  '0.0"%"',
    }

    # ── Data rows ────────────────────────────────────────────────
    for game in games_list:
        ri      = current_row
        is_bb   = (game.get("Home Team"), game.get("Away Team")) in best_bet_keys
        row_bg  = BEST_BET_BG if is_bb else (ROW_ODD if ri % 2 == 1 else ROW_EVEN)
        conf    = game.get("Confidence", "Medium")
        home_wp = game.get("Home Win Probability", 50)
        edge    = game.get("Model Edge", 0)

        for ci, (key, _, align, __, ___) in enumerate(columns, start=1):
            val  = game.get(key, "")
            cell = ws.cell(row=ri, column=ci, value=sanitize_cell(val))
            cell.border    = border
            cell.font      = Font(name="Calibri", size=10)
            cell.alignment = al(align)

            # Apply number format where defined
            if key in NUM_FMT:
                cell.number_format = NUM_FMT[key]

            if key == "Predicted Winner":
                cell.fill = fill(WINNER_FILL)
                cell.font = Font(name="Calibri", size=10, bold=True, color=GREEN_DARK)

            elif key == "Home Win Probability":
                cell.font = Font(name="Calibri", size=10, bold=True)
                cell.fill = fill(PROB_HIGH if home_wp >= 65 else
                                 PROB_LOW  if home_wp <= 40 else PROB_MID)

            elif key == "Confidence":
                if conf == "High":
                    cell.fill = fill(GREEN_LIGHT)
                    cell.font = Font(name="Calibri", size=10, bold=True, color=GREEN_DARK)
                elif conf == "Medium":
                    cell.fill = fill(YELLOW_FILL)
                    cell.font = Font(name="Calibri", size=10, bold=True, color=YELLOW_DARK)
                else:
                    cell.fill = fill(RED_FILL)
                    cell.font = Font(name="Calibri", size=10, bold=True, color=RED_DARK)

            elif key == "Model Edge":
                edge_disp = f"+{val}%" if isinstance(val, (int, float)) and val >= 0 else f"{val}%"
                cell.value = edge_disp
                if isinstance(val, (int, float)):
                    if val >= 5:
                        cell.fill = fill(GREEN_LIGHT)
                        cell.font = Font(name="Calibri", size=10, bold=True, color=EDGE_POS)
                    elif val <= -5:
                        cell.fill = fill(RED_FILL)
                        cell.font = Font(name="Calibri", size=10, bold=True, color=EDGE_NEG)
                    else:
                        cell.fill = fill(row_bg)
                        cell.font = Font(name="Calibri", size=10)
                else:
                    cell.fill = fill(row_bg)

            else:
                cell.fill = fill(row_bg)

            if is_bb and key not in ("Predicted Winner", "Home Win Probability",
                                     "Confidence", "Model Edge"):
                cell.font = Font(name="Calibri", size=10, bold=True)

        ws.row_dimensions[ri].height = 22
        current_row += 1

    # ── Freeze & filter ──────────────────────────────────────────
    ws.freeze_panes = f"A{data_start_row}"
    ws.auto_filter.ref = (
        f"A{hdr_row}:{last_col}{hdr_row + len(games_list)}"
    )

    # ── Legend ───────────────────────────────────────────────────
    legend_start = current_row + 1
    ws.merge_cells(f"A{legend_start}:{last_col}{legend_start}")
    lh = ws[f"A{legend_start}"]
    lh.value     = "Confidence & Colour Legend"
    lh.font      = Font(bold=True, size=10, name="Calibri", color=WHITE)
    lh.fill      = fill(NAVY)
    lh.alignment = al("left")
    ws.row_dimensions[legend_start].height = 20

    legend_items = [
        ("★ Best Bet",  GOLD_LIGHT,  GOLD,       "High or Medium confidence game with model edge ≥ 5% over Vegas implied — top picks of the day"),
        ("High",        GREEN_LIGHT, GREEN_DARK, "All three models agree within 8% — strongest conviction"),
        ("Medium",      YELLOW_FILL, YELLOW_DARK,"Models agree within 18% — reasonable edge, some caution"),
        ("Low",         RED_FILL,    RED_DARK,   "Models diverge more than 18% — conflicting signals, use caution"),
        ("Edge +5%+",   GREEN_LIGHT, EDGE_POS,   "Model predicts home team at least 5% more likely to win than Vegas implies"),
        ("Edge −5%−",   RED_FILL,    EDGE_NEG,   "Vegas implies home team 5%+ more likely than model does"),
    ]
    for offset, (label, bg, fg, desc) in enumerate(legend_items, start=1):
        row = legend_start + offset
        lc  = ws.cell(row=row, column=1, value=label)
        lc.fill      = fill(bg)
        lc.font      = Font(bold=True, size=10, name="Calibri", color=fg)
        lc.alignment = al("center")
        lc.border    = border
        ws.merge_cells(f"B{row}:{last_col}{row}")
        dc = ws.cell(row=row, column=2, value=desc)
        dc.font      = Font(size=10, name="Calibri")
        dc.alignment = al("left")
        dc.fill      = fill(ROW_ODD)
        ws.row_dimensions[row].height = 18

    # ── Column Definitions ───────────────────────────────────────
    def_start = legend_start + len(legend_items) + 2
    ws.merge_cells(f"A{def_start}:{last_col}{def_start}")
    dh = ws[f"A{def_start}"]
    dh.value     = "Column Definitions"
    dh.font      = Font(bold=True, size=10, name="Calibri", color=WHITE)
    dh.fill      = fill(NAVY)
    dh.alignment = al("left")
    ws.row_dimensions[def_start].height = 20

    definitions = [
        ("Game Time",               "The scheduled first pitch time in Eastern Time."),
        ("Weather",                 "Live ballpark conditions from Open-Meteo at game start: temperature, wind speed & direction relative to center field (OUT = helps hitters, IN = hurts), and rain %. Domes show 'dome'."),
        ("Home SP ERA (at Home)",   "Home starter's ERA in home games only this season."),
        ("Away SP ERA (on Road)",   "Away starter's ERA in road games only this season."),
        ("Park Factor",             "Ballpark scoring effect. >1.0 = hitter-friendly (Coors 1.30). <1.0 = pitcher-friendly (Petco 0.90)."),
        ("Adv Model %",             "Formula model: win %, runs/game, bullpen ERA, pitcher ERA, park factor, weather. 42% of final probability."),
        ("ML Model %",              "Random Forest trained on 7,300+ real MLB games. Learns patterns from history. 42% of final probability."),
        ("Vegas Implied %",         "Vig-removed consensus probability averaged across all sportsbooks. 16% weight when live, 10% from win% fallback."),
        ("Model Edge",              "Final blended model % minus Vegas implied %. Positive = model favours home team more than the market does. This is the key number for identifying value."),
        ("Home Win Probability",    "Final blended probability. Green ≥ 65%, blue 40–65%, red ≤ 40%."),
        ("Predicted Winner",        "Team predicted to win. Home team if probability ≥ 50%, away otherwise."),
        ("Confidence",              "Model agreement. High = within 8% (strongest). Medium = within 18%. Low = diverge >18%."),
    ]
    for offset, (col_name, desc) in enumerate(definitions, start=1):
        row = def_start + offset
        bg  = ROW_ODD if offset % 2 == 1 else ROW_EVEN
        nc  = ws.cell(row=row, column=1, value=col_name)
        nc.font      = Font(name="Calibri", size=10, bold=True, color=NAVY)
        nc.fill      = fill(bg)
        nc.alignment = al("left")
        nc.border    = border
        ws.merge_cells(f"B{row}:{last_col}{row}")
        dc = ws.cell(row=row, column=2, value=desc)
        dc.font      = Font(name="Calibri", size=10)
        dc.fill      = fill(bg)
        dc.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        dc.border    = border
        ws.row_dimensions[row].height = 28

    # ── Results Tracker tab ──────────────────────────────────────
    tracker_df = update_results_tracker(games_list)
    _save_tracker_json(tracker_df)

    wt = wb.create_sheet("Results Tracker")

    TR_NAVY     = "1B3A6B"
    TR_WHITE    = "FFFFFF"
    TR_ODD      = "F4F6FA"
    TR_EVEN     = "FFFFFF"
    TR_GREEN_BG = "C6EFCE";  TR_GREEN_FG = "276221"
    TR_RED_BG   = "FFD6D6";  TR_RED_FG   = "8B1A1A"
    TR_PENDING  = "FFF3C4";  TR_PEND_FG  = "7A6000"

    tr_thin   = Side(style="thin", color="C8CDD8")
    tr_border = Border(left=tr_thin, right=tr_thin, top=tr_thin, bottom=tr_thin)

    # Compute running record by confidence tier
    resolved_rows = tracker_df[
        tracker_df["Actual Winner"].notna() &
        (tracker_df["Actual Winner"].str.strip() != "") &
        (tracker_df["Actual Winner"].str.lower() != "nan")
    ]

    def _record(subset):
        correct = sum(
            1 for _, r in subset.iterrows()
            if str(r.get("Predicted Winner", "")).strip() == str(r.get("Actual Winner", "")).strip()
        )
        total = len(subset)
        pct   = round(correct / total * 100) if total else 0
        return correct, total, pct

    overall_c, overall_t, overall_pct = _record(resolved_rows)
    high_c,    high_t,    high_pct    = _record(resolved_rows[resolved_rows["Confidence"] == "High"])
    med_c,     med_t,     med_pct     = _record(resolved_rows[resolved_rows["Confidence"] == "Medium"])
    low_c,     low_t,     low_pct     = _record(resolved_rows[resolved_rows["Confidence"] == "Low"])

    # Title row
    wt.merge_cells("A1:I1")
    ttl = wt["A1"]
    ttl.value     = f"⚾   MLB Results Tracker   —   Updated {today_str}"
    ttl.font      = Font(color=TR_WHITE, bold=True, size=14, name="Calibri")
    ttl.fill      = PatternFill("solid", fgColor=TR_NAVY)
    ttl.alignment = Alignment(horizontal="center", vertical="center")
    wt.row_dimensions[1].height = 30

    # Record summary row
    wt.merge_cells("A2:I2")
    rec_note = wt["A2"]
    if overall_t > 0:
        rec_note.value = (
            f"Overall: {overall_c}–{overall_t - overall_c} ({overall_pct}%)     "
            f"High confidence: {high_c}–{high_t - high_c} ({high_pct}%)     "
            f"Medium confidence: {med_c}–{med_t - med_c} ({med_pct}%)     "
            f"Low confidence: {low_c}–{low_t - low_c} ({low_pct}%)"
        )
    else:
        rec_note.value = "No resolved games yet — record will appear here once results are filled in."
    rec_note.font      = Font(size=10, bold=True, name="Calibri", color=TR_NAVY)
    rec_note.alignment = Alignment(horizontal="left", vertical="center")
    rec_note.fill      = PatternFill("solid", fgColor="D6E4F0")
    wt.row_dimensions[2].height = 22

    # Instruction row
    wt.merge_cells("A3:I3")
    note = wt["A3"]
    note.value     = 'Fill in "Actual Winner" with the winning team name — the "Correct?" column updates automatically.'
    note.font      = Font(size=9, italic=True, name="Calibri", color="444444")
    note.alignment = Alignment(horizontal="left", vertical="center")
    note.fill      = PatternFill("solid", fgColor="EEF1F8")
    wt.row_dimensions[3].height = 16

    # Column headers
    tr_cols = [
        ("Date",             11),
        ("Game Time",        13),
        ("Home Team",        22),
        ("Away Team",        22),
        ("Predicted Winner", 22),
        ("Home Win %",       12),
        ("Model Edge",       12),
        ("Confidence",       13),
        ("Actual Winner",    22),
        ("Correct?",         13),
    ]
    for col_idx, (col_name, col_width) in enumerate(tr_cols, start=1):
        cell = wt.cell(row=4, column=col_idx, value=col_name)
        cell.font      = Font(bold=True, size=10, name="Calibri", color=TR_WHITE)
        cell.fill      = PatternFill("solid", fgColor=TR_NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = tr_border
        wt.column_dimensions[get_column_letter(col_idx)].width = col_width
    wt.row_dimensions[4].height = 22

    # Data rows
    for row_idx, (_, tr_row) in enumerate(tracker_df.iterrows(), start=5):
        bg        = TR_ODD if row_idx % 2 == 0 else TR_EVEN
        predicted = str(tr_row.get("Predicted Winner", "")).strip()
        actual    = str(tr_row.get("Actual Winner", "")).strip()
        edge_val  = tr_row.get("Model Edge", "")

        if actual and actual.lower() not in ("", "nan"):
            is_correct     = actual == predicted
            correct_val    = "✓  Correct" if is_correct else "✗  Wrong"
            correct_fill   = TR_GREEN_BG if is_correct else TR_RED_BG
            correct_font_c = TR_GREEN_FG if is_correct else TR_RED_FG
        else:
            correct_val    = "Pending"
            correct_fill   = TR_PENDING
            correct_font_c = TR_PEND_FG

        try:
            edge_num = float(edge_val)
            edge_str = f"+{edge_num}%" if edge_num >= 0 else f"{edge_num}%"
        except (ValueError, TypeError):
            edge_str = str(edge_val) if edge_val else ""

        row_vals = [
            tr_row.get("Date", ""),
            tr_row.get("Game Time", ""),
            tr_row.get("Home Team", ""),
            tr_row.get("Away Team", ""),
            predicted,
            tr_row.get("Home Win %", ""),
            edge_str,
            tr_row.get("Confidence", ""),
            actual if actual and actual.lower() != "nan" else "",
            correct_val,
        ]

        LEFT_COLS = {3, 4, 5, 9}
        for col_idx, val in enumerate(row_vals, start=1):
            cell = wt.cell(row=row_idx, column=col_idx, value=sanitize_cell(val))
            cell.font      = Font(size=10, name="Calibri")
            cell.alignment = Alignment(
                horizontal="left" if col_idx in LEFT_COLS else "center",
                vertical="center"
            )
            cell.border = tr_border

            if col_idx == 10:
                cell.fill = PatternFill("solid", fgColor=correct_fill)
                cell.font = Font(size=10, name="Calibri", bold=True, color=correct_font_c)
            elif col_idx == 8:
                conf = str(val)
                if conf == "High":
                    cell.fill = PatternFill("solid", fgColor=TR_GREEN_BG)
                    cell.font = Font(size=10, name="Calibri", bold=True, color=TR_GREEN_FG)
                elif conf == "Medium":
                    cell.fill = PatternFill("solid", fgColor=TR_PENDING)
                    cell.font = Font(size=10, name="Calibri", bold=True, color=TR_PEND_FG)
                elif conf == "Low":
                    cell.fill = PatternFill("solid", fgColor=TR_RED_BG)
                    cell.font = Font(size=10, name="Calibri", bold=True, color=TR_RED_FG)
                else:
                    cell.fill = PatternFill("solid", fgColor=bg)
            elif col_idx == 7:
                try:
                    ev = float(edge_val)
                    if ev >= 5:
                        cell.fill = PatternFill("solid", fgColor=TR_GREEN_BG)
                        cell.font = Font(size=10, name="Calibri", bold=True, color=TR_GREEN_FG)
                    elif ev <= -5:
                        cell.fill = PatternFill("solid", fgColor=TR_RED_BG)
                        cell.font = Font(size=10, name="Calibri", bold=True, color=TR_RED_FG)
                    else:
                        cell.fill = PatternFill("solid", fgColor=bg)
                except (ValueError, TypeError):
                    cell.fill = PatternFill("solid", fgColor=bg)
            else:
                cell.fill = PatternFill("solid", fgColor=bg)

        wt.row_dimensions[row_idx].height = 18

    wt.freeze_panes = "A5"

    wb.save(output_path)
    print(f"Daily MLB report saved to {output_path}")
    _save_predictions_json(games_list)
    return output_path


def run_daily_predictions():
    games_today = []

    # Feature 1 — auto-fill yesterday's results before anything else
    print("Checking yesterday's results...")
    auto_fill_results()

    schedule = fetch_todays_schedule()
    if not schedule:
        print("No schedule data — using placeholder games.")
        schedule = [
            {"home_name": "Yankees", "home_id": 147,
             "away_name": "Red Sox", "away_id": 111,
             "venue": "Yankee Stadium",
             "home_pitcher_id": None, "home_pitcher_name": "TBD",
             "away_pitcher_id": None, "away_pitcher_name": "TBD"},
        ]

    print(f"Running predictions for {len(schedule)} game(s)...\n")

    # Fetch once up-front (all cached for the run)
    savant_batting = fetch_savant_team_batting()   # Savant team batting
    all_odds       = fetch_vegas_odds()            # Feature 2 — live Vegas odds

    for game in schedule:
        home_name = game["home_name"]
        away_name = game["away_name"]
        home_id   = game["home_id"]
        away_id   = game["away_id"]
        venue     = game["venue"]
        print(f"--- {away_name} at {home_name} ({venue}) ---")

        home_stats   = get_team_stats(home_id, home_name, savant_batting=savant_batting)
        away_stats   = get_team_stats(away_id, away_name, savant_batting=savant_batting)
        home_pitch   = get_pitcher_stats(game["home_pitcher_id"], game["home_pitcher_name"])
        away_pitch   = get_pitcher_stats(game["away_pitcher_id"], game["away_pitcher_name"])
        weather      = get_weather(venue, game_date_utc=game.get("game_date", ""))
        park_factor  = get_park_factor(venue)
        umpire       = get_umpire_factor("Default")
        travel_home  = get_travel(home_name)
        travel_away  = get_travel(away_name)

        # Feature 3 — live lineup strength (falls back to team avg if not yet posted)
        home_lineup = get_lineup_strength(home_id, home_name, home_stats["lineup_strength"])
        away_lineup = get_lineup_strength(away_id, away_name, away_stats["lineup_strength"])

        # Feature 4 — real bullpen fatigue (replaces placeholder 0.35)
        bullpen_home = get_bullpen_fatigue(home_id, home_name)
        bullpen_away = get_bullpen_fatigue(away_id, away_name)

        # Feature 2 — enhanced Vegas odds (vig-removed consensus or win% fallback)
        vegas = get_vegas_line(home_name, away_name, all_odds,
                               home_wpct=home_stats["wpct"],
                               away_wpct=away_stats["wpct"])

        wind_dir_out = 1 if weather["wind_start_dir"] == "out" else 0

        adv_inputs = (
            home_stats["wpct"],        away_stats["wpct"],
            home_stats["rpg"],         away_stats["rpg"],
            home_pitch["home_era"],    home_pitch["away_era"],
            away_pitch["home_era"],    away_pitch["away_era"],
            home_stats["bullpen_era"], away_stats["bullpen_era"],
            park_factor,
            umpire,
            travel_home, travel_away,
            bullpen_home, bullpen_away,
            True,
            weather["temp_start"],       weather["temp_mid"],
            weather["wind_start_speed"], weather["wind_start_dir"],
            weather["wind_mid_speed"],   weather["wind_mid_dir"],
            weather["rain_chance"],
            home_stats["hard_hit"],    away_stats["hard_hit"],
            home_stats["defense"],     away_stats["defense"],
            home_lineup,               away_lineup,
            home_stats["vs_hand_split"], away_stats["vs_hand_split"]
        )

        adv_prob = advanced_model(adv_inputs)

        ml_features = {
            "team_wpct":       home_stats["wpct"],
            "opp_wpct":        away_stats["wpct"],
            "team_rpg":        home_stats["rpg"],
            "opp_rpg":         away_stats["rpg"],
            "pitcher_era":     home_pitch["home_era"],
            "opp_pitcher_era": away_pitch["away_era"],
            "bullpen_era":     home_stats["bullpen_era"],
            "opp_bullpen_era": away_stats["bullpen_era"],
            "park_factor":     park_factor,
            "temp":            weather["temp_start"],
            "wind_speed":      weather["wind_start_speed"],
            "wind_dir_out":    wind_dir_out,
            "home":            1,
            "pitcher_recent_delta":     home_pitch.get("home_recent_delta", 0.0),
            "opp_pitcher_recent_delta": away_pitch.get("away_recent_delta", 0.0),
        }
        if ml_model and isinstance(ml_model, dict):
            feat_df  = pd.DataFrame([ml_features])
            feat_cols = ml_model.get("feature_cols")
            if feat_cols:
                for c in feat_cols:
                    if c not in feat_df.columns:
                        feat_df[c] = 0.0
                feat_df = feat_df[feat_cols]
            rf_prob  = ml_model["rf"].predict_proba(feat_df)[0, 1]
            xgb_prob = ml_model["xgb"].predict_proba(feat_df)[0, 1]
            ml_prob  = round((rf_prob + xgb_prob) / 2, 4)
        else:
            ml_prob = adv_prob

        vegas_prob    = vegas.get("home_prob_novig",
                           1 / (1 + 10 ** (vegas["home_moneyline"] / 100)))
        vegas_is_live = vegas.get("is_live", False)
        final_prob    = final_win_probability(adv_prob, ml_prob, vegas_prob,
                                              vegas_is_live=vegas_is_live)

        adv_pct   = round(adv_prob    * 100, 2)
        ml_pct    = round(ml_prob     * 100, 2)
        vegas_pct = round(vegas_prob  * 100, 2)

        spread = max(adv_pct, ml_pct, vegas_pct) - min(adv_pct, ml_pct, vegas_pct)
        if spread <= 8:
            confidence = "High"
        elif spread <= 18:
            confidence = "Medium"
        else:
            confidence = "Low"

        predicted_winner = home_name if final_prob >= 50 else away_name
        model_edge = round(final_prob - vegas_pct, 1)

        # Parse game time to Eastern for display
        game_time_et = ""
        raw_gd = game.get("game_date", "")
        if raw_gd:
            try:
                import zoneinfo
                dt_utc = datetime.datetime.strptime(raw_gd, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=datetime.timezone.utc)
                dt_et  = dt_utc.astimezone(zoneinfo.ZoneInfo("America/New_York"))
                game_time_et = dt_et.strftime("%-I:%M %p ET")
            except Exception:
                pass

        weather_str = (
            f"{weather['temp_start']}°F · "
            + ("dome" if weather['wind_start_speed'] == 0 and weather['wind_start_dir'] == 'neutral'
               else f"{weather['wind_start_speed']}mph {weather['wind_start_dir'].upper()}")
            + (f" · {weather['rain_chance']}% rain" if weather['rain_chance'] > 5 else "")
        )

        games_today.append({
            # Display columns
            "Game Time":               game_time_et,
            "Home Team":               home_name,
            "Away Team":               away_name,
            "Venue":                   venue,
            "Weather":                 weather_str,
            "Home Starting Pitcher":   game["home_pitcher_name"],
            "Away Starting Pitcher":   game["away_pitcher_name"],
            "Home Win %":              home_stats["wpct"],
            "Away Win %":              away_stats["wpct"],
            "Home Runs / Game":        home_stats["rpg"],
            "Away Runs / Game":        away_stats["rpg"],
            "Home Bullpen ERA":        home_stats["bullpen_era"],
            "Away Bullpen ERA":        away_stats["bullpen_era"],
            "Home SP ERA (at Home)":   home_pitch["home_era"],
            "Away SP ERA (on Road)":   away_pitch["away_era"],
            "Park Factor":             park_factor,
            "Adv Model %":             adv_pct,
            "ML Model %":              ml_pct,
            "Vegas Implied %":         vegas_pct,
            "Model Edge":              model_edge,
            "Home Win Probability":    final_prob,
            "Predicted Winner":        predicted_winner,
            "Confidence":              confidence,
            # ML training features stored in tracker for auto-recalibration
            "team_wpct":        home_stats["wpct"],
            "opp_wpct":         away_stats["wpct"],
            "team_rpg":         home_stats["rpg"],
            "opp_rpg":          away_stats["rpg"],
            "pitcher_era":      home_pitch["home_era"],
            "opp_pitcher_era":  away_pitch["away_era"],
            "bullpen_era":      home_stats["bullpen_era"],
            "opp_bullpen_era":  away_stats["bullpen_era"],
            "park_factor":      park_factor,
            "temp":             weather["temp_start"],
            "wind_speed":       weather["wind_start_speed"],
            "wind_dir_out":     wind_dir_out,
            "home":             1,
            "pitcher_recent_delta":     home_pitch.get("home_recent_delta", 0.0),
            "opp_pitcher_recent_delta": away_pitch.get("away_recent_delta", 0.0),
        })
        print(f"  => Home win probability: {final_prob}%  |  Predicted Winner: {predicted_winner}  [{confidence} confidence]\n")

    # Feature 6 — monthly auto-recalibration of the ML model
    retrain_model_if_needed()

    return save_daily_report(games_today)


# ============================================================
# EMAIL REPORT
# ============================================================

def email_report(report_path):
    to_email   = "radkins0206199@gmail.com"
    from_email = "radkins0206199@gmail.com"
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")

    if not app_password:
        print("ERROR: GMAIL_APP_PASSWORD environment variable not set.")
        return

    today_str = datetime.datetime.now(ET).strftime("%B %d, %Y")
    msg = EmailMessage()
    msg["Subject"] = f"MLB Prediction Report — {today_str}"
    msg["From"]    = from_email
    msg["To"]      = to_email
    msg.set_content(
        f"Your MLB prediction report for {today_str} is attached.\n\n"
        "Probabilities reflect live team stats, probable pitcher ERA, and Vegas lines."
    )

    with open(report_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"mlb_report_{today_et()}.xlsx"
        )

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(from_email, app_password)
        smtp.send_message(msg)

    print("Email sent successfully.")


# ============================================================
# RUN EVERYTHING
# ============================================================

if __name__ == "__main__":
    report_path = run_daily_predictions()
    email_report(report_path)
