#!/usr/bin/env python3
"""
migrate_to_supabase.py
Run this once to migrate results_tracker.csv into Supabase.
Usage: python3 migrate_to_supabase.py
"""
import csv, json, os, urllib.request

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://wkxpdmfabiepkfdxbdie.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

def sb_post(path, data):
    url = f'{SUPABASE_URL}/rest/v1{path}'
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('apikey', SUPABASE_KEY)
    # Authorization not needed for new sb_ keys
    req.add_header('Content-Type', 'application/json')
    req.add_header('Prefer', 'return=minimal')
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.status
    except urllib.error.HTTPError as e:
        print(f'  Error: {e.code} {e.read().decode()[:200]}')
        return e.code

def safe_float(v):
    try: return float(v) if v and str(v).strip() not in ('', 'nan', 'None') else None
    except: return None

def safe_bool(v):
    if isinstance(v, bool): return v
    if str(v).strip().lower() in ('true', '1', 'yes'): return True
    if str(v).strip().lower() in ('false', '0', 'no'): return False
    return None

# Read CSV
csv_path = os.path.join(os.path.dirname(__file__), 'results_tracker.csv')
if not os.path.exists(csv_path):
    print(f'❌ CSV not found at {csv_path}')
    exit(1)

rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rows.append(dict(row))

print(f'Found {len(rows)} rows in CSV')

# Map CSV columns to Supabase columns
mapped = []
for r in rows:
    mapped.append({
        'date':                    r.get('Date', '').strip(),
        'home_team':               r.get('Home Team', '').strip(),
        'away_team':               r.get('Away Team', '').strip(),
        'predicted_winner':        r.get('Predicted Winner', '').strip() or None,
        'actual_winner':           r.get('Actual Winner', '').strip() or None,
        'correct':                 r.get('Correct?', '').strip() or None,
        'confidence':              r.get('Confidence', '').strip() or None,
        'model_edge':              safe_float(r.get('Model Edge')),
        'home_win_prob':           safe_float(r.get('Home Win %')),
        'game_time':               r.get('Game Time', '').strip() or None,
        'home_sp_era':             safe_float(r.get('pitcher_era')),
        'away_sp_era':             safe_float(r.get('opp_pitcher_era')),
        'bullpen_era':             safe_float(r.get('bullpen_era')),
        'opp_bullpen_era':         safe_float(r.get('opp_bullpen_era')),
        'park_factor':             safe_float(r.get('park_factor')),
        'temp':                    safe_float(r.get('temp')),
        'wind_speed':              safe_float(r.get('wind_speed')),
        'wind_dir_out':            safe_bool(r.get('wind_dir_out')),
        'home':                    safe_bool(r.get('home')),
        'pitcher_recent_delta':    safe_float(r.get('pitcher_recent_delta')),
        'opp_pitcher_recent_delta':safe_float(r.get('opp_pitcher_recent_delta')),
    })
    # Filter out empty-date rows
    mapped = [m for m in mapped if m['date'] and m['home_team'] and m['away_team']]

print(f'Migrating {len(mapped)} valid rows...')

# Insert in batches of 20
batch_size = 20
success = 0
for i in range(0, len(mapped), batch_size):
    batch = mapped[i:i+batch_size]
    status = sb_post('/results', batch)
    if status in (200, 201):
        success += len(batch)
        print(f'  ✅ Batch {i//batch_size + 1}: {len(batch)} rows inserted')
    else:
        print(f'  ❌ Batch {i//batch_size + 1}: failed with status {status}')

print(f'\nDone. {success}/{len(mapped)} rows migrated to Supabase.')
