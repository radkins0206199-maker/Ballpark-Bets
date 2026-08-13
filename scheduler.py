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


# Hours (ET) to attempt closing-line capture. MLB first pitches run from
# roughly 12:00 to 22:10 ET; capture_closing_lines() only writes for games
# within ~90 min of starting, so these hourly passes progressively cover the
# whole slate. Repeated calls are idempotent — a closing price is never
# overwritten once stored.
CLV_HOURS = list(range(12, 23))


def run_clv_capture():
    """Capture closing lines near first pitch. Failures never affect the report."""
    try:
        from mlb_daily_report import capture_closing_lines
        result = capture_closing_lines()
        got = result.get("captured", 0)
        if got:
            print(f"[Scheduler] CLV: captured {got} closing line(s)")
    except Exception as e:
        print(f"[Scheduler] CLV capture failed (non-fatal): {e}")


def _run_loop():
    if not predictions_fresh_today():
        print("[Scheduler] No fresh predictions — running now")
        run_report()
    else:
        print("[Scheduler] Predictions already fresh, waiting for tomorrow")

    last_report_date = et_now().date()

    while True:
        # Wake at the top of each hour rather than sleeping a full day, so
        # closing lines can be captured near first pitch. A line captured at
        # 9 AM is an opening line and would make CLV meaningless.
        now = et_now()
        nxt = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_until(nxt)

        now = et_now()

        # Daily report at 9 AM ET
        if now.hour == 9 and now.date() != last_report_date:
            print("[Scheduler] 9 AM ET — generating today's picks")
            run_report()
            last_report_date = now.date()

        # Closing-line capture through the afternoon and evening
        if now.hour in CLV_HOURS:
            run_clv_capture()


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
