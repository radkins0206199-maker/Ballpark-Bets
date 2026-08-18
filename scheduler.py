import datetime
import time
import subprocess
import os
import pytz

ET = pytz.timezone("America/New_York")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def et_now():
    return datetime.datetime.now(ET)


def predictions_fresh_today():
    path = os.path.join(DATA_DIR, "daily_predictions.json")
    if not os.path.exists(path):
        return False
    try:
        import json
        with open(path) as f:
            d = json.load(f)
        return d.get("date") == et_now().strftime("%Y-%m-%d")
    except Exception:
        return False


def run_report():
    print(f"[Scheduler] Running daily report at {et_now().strftime('%Y-%m-%d %H:%M ET')}", flush=True)
    try:
        script = os.path.join(DATA_DIR, "mlb_daily_report.py")
        import sys
        result = subprocess.run(
            ["python3", "-u", script],  # -u = unbuffered output
            timeout=600,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        if result.returncode == 0:
            print("[Scheduler] Report completed successfully", flush=True)
        else:
            print(f"[Scheduler] Report failed with code {result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("[Scheduler] Report timed out after 10 minutes", flush=True)
    except Exception as e:
        print(f"[Scheduler] Report error: {e}", flush=True)


def next_9am_et():
    now = et_now()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    return target


def sleep_until(dt):
    now = et_now()
    secs = (dt - now).total_seconds()
    if secs > 0:
        print(f"[Scheduler] Sleeping until {dt.strftime('%Y-%m-%d %H:%M ET')}")
        time.sleep(secs)


# NOTE: closing-line capture is NOT scheduled here.
#
# Render's free tier spins the service down after ~15 min without HTTP
# traffic, which kills this process and any timer inside it. An hourly loop
# added here on 2026-08-14 never fired — closing lines were captured exactly
# once, on the day the endpoint was hit by hand.
#
# Capture is therefore driven externally by the Netlify scheduled function
# netlify/functions/capture-clv.js, which wakes Render and then calls
# /api/mlb/capture-closing-lines. Same pattern as resolve-yesterday.js.


def _run_loop():
    if not predictions_fresh_today():
        print("[Scheduler] No fresh predictions — running now")
        run_report()
    else:
        print("[Scheduler] Predictions already fresh, waiting for tomorrow")

    last_report_date = et_now().date()

    while True:
        sleep_until(next_9am_et())
        print("[Scheduler] 9 AM ET — generating today's picks")
        run_report()
        last_report_date = et_now().date()


def main():
    print("[Scheduler] BallparkBets scheduler started (ET timezone — 9 AM daily)")
    while True:
        try:
            _run_loop()
        except Exception as e:
            print(f"[Scheduler] FATAL: {e} — restarting in 5 minutes")
            time.sleep(300)


if __name__ == "__main__":
    main()
