"""Download StatsBomb open data for one competition/season into ./data-style layout.

Usage: python download.py [competition_name] [season_name]
Default: "UEFA Euro" 2024. Writes matches.json, events/, lineups/, three-sixty/.
"""
import json, os, sys, concurrent.futures as cf
import requests

BASE = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data'
comp_name = sys.argv[1] if len(sys.argv) > 1 else 'UEFA Euro'
season_name = sys.argv[2] if len(sys.argv) > 2 else '2024'

comps = requests.get(f'{BASE}/competitions.json', timeout=60).json()
c = next(x for x in comps if x['competition_name'] == comp_name and x['season_name'] == season_name)
cid, sid = c['competition_id'], c['season_id']
print(f'{comp_name} {season_name}: competition_id={cid} season_id={sid} 360={c.get("match_available_360") is not None}')

matches = requests.get(f'{BASE}/matches/{cid}/{sid}.json', timeout=60).json()
json.dump(matches, open('matches.json', 'w'))
for d in ('events', 'lineups', 'three-sixty'):
    os.makedirs(d, exist_ok=True)

def fetch(args):
    folder, mid = args
    dst = f'{folder}/{mid}.json'
    if os.path.exists(dst):
        return
    r = requests.get(f'{BASE}/{folder}/{mid}.json', timeout=120)
    if r.status_code == 200:
        open(dst, 'wb').write(r.content)
    elif folder != 'three-sixty':
        r.raise_for_status()

jobs = [(f, m['match_id']) for m in matches for f in ('events', 'lineups', 'three-sixty')]
with cf.ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, jobs))
print(f'{len(matches)} matches downloaded')
