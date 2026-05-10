"""
BallparkBets — Render entry point.
Runs the Flask API and the daily ML scheduler in the same process.
The scheduler runs in a background thread; Flask serves HTTP so Render
sees a live web service and doesn't shut it down.
"""
import threading
import os
import sys

# Make sure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from api_server import app
from scheduler import _run_loop


def run_scheduler():
    while True:
        try:
            _run_loop()
        except Exception as e:
            import time
            print(f"[Scheduler] crashed: {e} — restarting in 5 min")
            time.sleep(300)


if __name__ == "__main__":
    # Start scheduler in background
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    print("[Start] Scheduler thread running")

    # Start Flask (blocking — keeps Render happy)
    port = int(os.environ.get("PORT", 10000))
    print(f"[Start] API server on port {port}")
    app.run(host="0.0.0.0", port=port)
