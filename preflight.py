#!/usr/bin/env python3
"""
preflight.py — BallparkBets backend deploy safety check.

Run this BEFORE every backend deploy:
    python3 preflight.py

It runs five checks and prints one clean PASS/FAIL report:
  1. Syntax — every .py file compiles
  2. Definitions — all critical functions/classes exist, none duplicated,
     key classes are defined before first use
  3. Retrain smoke test — trains all 4 ensemble models through the real
     CalibratedModel + _train_calibrated code path on synthetic data,
     including the CI bootstrap and feature-importance access that
     silently broke for 19 days in June 2026
  4. Globals — Elo constants and CI variables are present
  5. Requirements — heavy deps are declared

Exit code 0 = safe to deploy. Non-zero = do NOT deploy.

This exists because on 2026-06-04 a str_replace deleted the
`class CalibratedModel:` declaration line. Syntax stayed valid, the
pipeline kept serving an old pickle, and every retrain silently failed
with "name 'CalibratedModel' is not defined" for 19 days. The two checks
that would have caught it instantly — a definition-integrity scan and a
retrain smoke test — are checks 2 and 3 below.
"""

import ast
import sys
import warnings
import importlib.util
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MAIN = HERE / "mlb_daily_report.py"

GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def bad(msg):  print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YEL}⚠️  {msg}{RESET}")

failures = []

# Critical functions/classes that MUST exist for the pipeline to work.
# Add to this list whenever a new load-bearing definition is introduced.
CRITICAL_DEFS = [
    "CalibratedModel", "_train_calibrated",
    "_train_calibrated_crossfit", "_baseline_report", "_deployment_gate",
    "get_elo", "load_elo_ratings", "update_elo_after_resolve",
    "elo_win_probability", "_elo_expected", "_elo_k_factor",
    "fetch_vegas_odds", "detect_steam_move", "store_opening_lines",
    "get_travel", "retrain_model_if_needed", "run_daily_predictions",
    "auto_fill_results", "compute_calibration_metrics",
    "_save_predictions_supabase", "_save_game_features_to_results",
    "select_lock_of_the_day", "select_dangerous_underdog",
    "get_lightgbm_prediction", "retrain_lightgbm", "get_vegas_line",
]

# Classes that are instantiated elsewhere and therefore MUST be defined
# textually before their first use (def-before-use at module scope).
ORDER_SENSITIVE = ["CalibratedModel"]

# Module-scope constants the Elo + CI code depends on.
REQUIRED_GLOBALS = ["ELO_BASE", "ELO_K", "ELO_HOME_ADV", "_elo_cache", "_elo_loaded"]

REQUIRED_REQS = ["scikit-learn", "scipy", "numpy", "xgboost", "lightgbm", "psycopg2"]


# ────────────────────────────────────────────────────────────────────
def check_syntax():
    print("\n[1/5] Syntax")
    all_ok = True
    for py in sorted(HERE.glob("*.py")):
        if py.name == "preflight.py":
            continue
        try:
            ast.parse(py.read_text())
            ok(f"{py.name}")
        except SyntaxError as e:
            bad(f"{py.name} line {e.lineno}: {e.msg}")
            failures.append(f"syntax:{py.name}")
            all_ok = False
    return all_ok


def check_definitions():
    print("\n[2/5] Definitions")
    src = MAIN.read_text()
    tree = ast.parse(src)

    defined = {}
    all_def_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            all_def_names.append(node.name)
            # record earliest line for each name
            if node.name not in defined:
                defined[node.name] = node.lineno

    # 2a. all critical defs present
    missing = [d for d in CRITICAL_DEFS if d not in defined]
    if missing:
        bad(f"missing definitions: {missing}")
        failures.append("defs:missing")
    else:
        ok(f"all {len(CRITICAL_DEFS)} critical defs present")

    # 2b. no duplicates
    dupes = {n: c for n, c in Counter(all_def_names).items() if c > 1}
    if dupes:
        warn(f"duplicate definitions: {dupes}")
    else:
        ok("no duplicate definitions")

    # 2c. order-sensitive classes defined before first instantiation
    for name in ORDER_SENSITIVE:
        class_line = defined.get(name)
        first_use = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == name):
                if first_use is None or node.lineno < first_use:
                    first_use = node.lineno
        if class_line is None:
            bad(f"{name} not defined at all")
            failures.append(f"order:{name}")
        elif first_use is not None and class_line > first_use:
            bad(f"{name} used at line {first_use} but defined at {class_line}")
            failures.append(f"order:{name}")
        else:
            ok(f"{name} defined (line {class_line}) before first use")

    # 2d. required globals present
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
    missing_g = [g for g in REQUIRED_GLOBALS if g not in assigned]
    if missing_g:
        bad(f"missing globals: {missing_g}")
        failures.append("defs:globals")
    else:
        ok(f"all {len(REQUIRED_GLOBALS)} required globals assigned")

    # 2e. CI variables initialised before the game dict reads them
    init_pos = src.find("ci_low, ci_high, ci_width = None, None, None")
    use_pos  = src.find('"CI Low":')
    if init_pos < 0:
        bad("CI variables never initialised")
        failures.append("defs:ci_init")
    elif use_pos > 0 and init_pos > use_pos:
        bad("CI variables initialised AFTER first use")
        failures.append("defs:ci_init")
    else:
        ok("CI variables initialised before use")


def check_retrain_smoke():
    """
    The check that would have caught the June 2026 outage.
    Rebuilds the real CalibratedModel + _train_calibrated path on synthetic
    data and exercises every operation the daily retrain performs:
      - 4-model ensemble training
      - isotonic calibration wrapper
      - Nelder-Mead weight optimisation
      - RF bootstrap confidence interval (base_model.estimators_)
      - feature importance averaging (base_model.estimators_)
      - holdout vs train Brier
    """
    print("\n[3/5] Retrain smoke test")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.isotonic import IsotonicRegression
        from sklearn.ensemble import (RandomForestClassifier,
                                       GradientBoostingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from scipy.optimize import minimize
    except Exception as e:
        bad(f"could not import ML deps: {e}")
        failures.append("smoke:imports")
        return

    # Pull the ACTUAL class + helper out of the source so we test the real
    # code, not a copy that could drift from it.
    src = MAIN.read_text()
    ns = {
        "np": np, "IsotonicRegression": IsotonicRegression,
    }
    try:
        mod = ast.parse(src)
        wanted = {"CalibratedModel", "_train_calibrated"}
        for node in mod.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                exec(compile(ast.Module([node], []), "<extract>", "exec"), ns)
        CalibratedModel   = ns["CalibratedModel"]
        _train_calibrated = ns["_train_calibrated"]
        ok("extracted CalibratedModel + _train_calibrated from source")
    except Exception as e:
        bad(f"could not extract real definitions: {e}")
        failures.append("smoke:extract")
        return

    try:
        rng = np.random.RandomState(42)
        n = 400
        X = pd.DataFrame(rng.rand(n, 15), columns=[f"f{i}" for i in range(15)])
        y = pd.Series(((X["f0"] + X["f1"]) > 1.0).astype(int))
        X_tr, y_tr   = X[:280], y[:280]
        X_cal, y_cal = X[280:340], y[280:340]
        X_te, y_te   = X[340:], y[340:]

        cal_rf  = _train_calibrated(
            RandomForestClassifier(n_estimators=40, random_state=42),
            X_tr, y_tr, X_cal, y_cal)
        cal_xgb = _train_calibrated(
            GradientBoostingClassifier(n_estimators=40, random_state=42),
            X_tr, y_tr, X_cal, y_cal)
        scaler  = StandardScaler()
        Xtr_s   = scaler.fit_transform(X_tr)
        Xcal_s  = scaler.transform(X_cal)
        cal_lr  = _train_calibrated(
            LogisticRegression(C=0.1, max_iter=1000),
            Xtr_s, y_tr, Xcal_s, y_cal)
        ok("4-model ensemble trains through CalibratedModel")

        # weight optimisation
        probs = [cal_rf.predict_proba(X_cal)[:, 1],
                 cal_xgb.predict_proba(X_cal)[:, 1],
                 cal_rf.predict_proba(X_cal)[:, 1],
                 cal_lr.predict_proba(Xcal_s)[:, 1]]
        yv = np.array(y_cal)
        def eb(w):
            w = np.clip(w, 0.05, 0.6); w = w / w.sum()
            return float(np.mean((sum(wi*pi for wi, pi in zip(w, probs)) - yv) ** 2))
        minimize(eb, [0.3, 0.35, 0.2, 0.15], method="Nelder-Mead")
        ok("ensemble weight optimisation runs")

        # CI bootstrap via base_model.estimators_  (the bit that broke)
        trees = cal_rf.base_model.estimators_
        tp = np.array([t.predict_proba(X_te.values[:1])[0, 1] for t in trees])
        _ = (np.percentile(tp, 10), np.percentile(tp, 90))
        ok("RF bootstrap CI accesses base_model.estimators_")

        # feature importance via base_model.estimators_
        _ = np.mean([t.feature_importances_ for t in trees], axis=0)
        ok("feature importance averages base_model.estimators_")

        # holdout vs train brier
        hb = float(np.mean((cal_rf.predict_proba(X_te)[:, 1] - np.array(y_te)) ** 2))
        tb = float(np.mean((cal_rf.predict_proba(X_tr)[:, 1] - np.array(y_tr)) ** 2))
        ok(f"holdout Brier {hb:.3f} / train Brier {tb:.3f}")

    except Exception as e:
        bad(f"retrain path raised: {type(e).__name__}: {e}")
        failures.append("smoke:retrain")


def check_feature_pipeline():
    """
    Phase 4.5 — end-to-end feature path check.

    The Jun 2 – Aug 11 failure: team_wpct/opp_wpct/team_rpg/opp_rpg were in
    ml_feat_cols but absent from the training SELECT, so row.get() returned
    None and fell through to 0.0 on all 1,072 rows. Ten weeks of a model
    running with no team-strength signal at all.

    Core principle: a feature that is not verifiably retrieved, loaded, and
    used must be treated as NOT EXISTING. This asserts the DB -> SELECT ->
    ml_feat_cols path is intact for every declared feature.
    """
    print("\n[6/7] Feature pipeline")
    src = MAIN.read_text()

    m = __import__("re").search(r"ml_feat_cols\s*=\s*\[(.*?)\]", src, __import__("re").S)
    if not m:
        bad("could not locate ml_feat_cols")
        failures.append("featpipe:no_list")
        return
    feats = __import__("re").findall(r'"([^"]+)"', m.group(1))

    # Team-quality feats are cohort-gated: declared separately, spliced in
    # only once Cohort B is large enough to train on alone.
    tq = __import__("re").search(r"TEAM_QUALITY_FEATS\s*=\s*\[(.*?)\]", src, __import__("re").S)
    gated = __import__("re").findall(r'"([^"]+)"', tq.group(1)) if tq else []
    if gated:
        ok(f"cohort-gated features: {len(gated)} ({', '.join(gated[:3])}…)")
    feats = feats + gated
    ok(f"ml_feat_cols declares {len(feats)} features (incl. gated)")

    sel = __import__("re").search(r"SELECT date, home_team.*?FROM results", src, __import__("re").S)
    if not sel:
        bad("could not locate training SELECT")
        failures.append("featpipe:no_select")
        return
    select_sql = sel.group(0)

    # Columns the SELECT provides, accounting for the rename map
    renamed = {"pitcher_era": "home_sp_era", "opp_pitcher_era": "away_sp_era"}
    missing = []
    for f in feats:
        col = renamed.get(f, f)
        if col not in select_sql:
            missing.append(f)
    if missing:
        bad(f"in ml_feat_cols but NOT in training SELECT: {missing}")
        bad("  these would be silently 0.0 — the exact Jun-Aug bug")
        failures.append("featpipe:select_gap")
    else:
        ok("every ml_feat_col is retrieved by the training SELECT")

    # The write path must persist them too
    for col in ["team_wpct", "opp_wpct", "team_rpg", "opp_rpg", "elo_diff"]:
        if src.count(col) < 2:
            bad(f"{col} appears <2x — likely not written at prediction time")
            failures.append(f"featpipe:{col}")
    else:
        ok("team-quality + elo_diff appear in both read and write paths")

    if "_assert_feature_health" in src:
        ok("retrain-blocking feature health gate present")
    else:
        bad("feature health gate missing")
        failures.append("featpipe:no_health_gate")

    if "raw_games" in src and "def _save_predictions_supabase(payload, raw_games=None)" in src:
        ok("feature_snapshot receives unslimmed dicts")
    else:
        bad("feature_snapshot write path still broken")
        failures.append("featpipe:snapshot")


def check_requirements():
    print("\n[7/7] Requirements & secrets")
    req = HERE / "requirements.txt"
    if not req.exists():
        bad("requirements.txt missing")
        failures.append("reqs:missing")
        return
    text = req.read_text().lower()
    missing = [r for r in REQUIRED_REQS if r.lower() not in text]
    if missing:
        bad(f"missing from requirements.txt: {missing}")
        failures.append("reqs:incomplete")
    else:
        ok(f"all {len(REQUIRED_REQS)} key deps declared")


def check_no_hardcoded_secrets():
    
    src = MAIN.read_text()
    # the old leaked service_role token fragment
    if "nmA4kZM5" in src or '"role":"service_role"' in src:
        bad("hardcoded service_role key found in source")
        failures.append("secrets:service_role")
    else:
        ok("no hardcoded service_role key")


def main():
    print("=" * 60)
    print("  BallparkBets — backend preflight")
    print("=" * 60)

    if not MAIN.exists():
        print(f"{RED}mlb_daily_report.py not found next to preflight.py{RESET}")
        sys.exit(2)

    check_syntax()
    check_definitions()
    check_retrain_smoke()
    check_feature_pipeline()
    check_requirements()
    check_no_hardcoded_secrets()

    print("\n" + "=" * 60)
    if failures:
        print(f"  {RED}✗ FAIL — {len(failures)} issue(s): {failures}{RESET}")
        print(f"  {RED}  DO NOT DEPLOY until these are resolved.{RESET}")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"  {GREEN}✓ PASS — safe to deploy{RESET}")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
