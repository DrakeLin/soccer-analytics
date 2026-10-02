# soccer-analytics

Pass- and positioning-level football analysis for a [Side Quest](https://thesidequest.com)-style story,
built on [StatsBomb Open Data](https://github.com/statsbomb/open-data) — currently **UEFA Euro 2024**
(51 matches, full event data + 360 freeze frames for every match).

## Layout

| Path | What |
|---|---|
| `download.py` | Pull a competition from StatsBomb open data (default: Euro 2024) into `events/`, `lineups/`, `three-sixty/`, `matches.json` |
| `flatten.py` | Flatten events + join 360 freeze-frame context (nearest defender, defenders goal-side, defensive line, block width/depth) → `events_flat.pkl` |
| `stats.py` | Minutes from lineups, tournament-fitted xT grid, player table (`out/players.csv`) and team table (`out/teams.csv`) |
| `charts.py` | Prototype charts → `out/*.svg|png` |
| `ideas.md` | Survey of common football analytics + 360-only stats + story angles |
| `out/` | Derived results (committed): CSVs, xT grid, charts |

Raw data (~535 MB) and the flattened pickle are gitignored; regenerate with:

```bash
pip install -r requirements.txt
python download.py            # or: python download.py "FIFA World Cup" 2022
python flatten.py
python stats.py
python charts.py
```

## Current prototype outputs

- `out/players.csv` — per player: passes, completion %, progressive & line-breaking passes, passes under pressure,
  xT from passes/carries, ball receipts by space to nearest defender, final-third receipts, shots/xG/goals, per-90s
- `out/teams.csv` — per team: possession, defensive-line height, block width/depth, PPDA, line-breaking passes for/against,
  final-third receipt-space rates for/against, xT, xG/xGA per match
- `out/kroos_lb` — Toni Kroos line-breaking passes map
- `out/kroos_360` — a single 360 freeze frame (Germany v Hungary 73')
- `out/xt_grid` — xT grid fitted on this tournament
- `out/spain_network` — Spain pass network, Euro 2024 final (until first sub)
- `out/defline_xga` — defensive-line height vs xG conceded per match
- `out/receipts_space` — final-third receipts per 90 vs share received with ≥5 yds of space

## Method caveats

- "Line-breaking" and "defensive line" are our own approximations from the 360 freeze frames (second-deepest
  visible outfield defender; pass advances ≥10% toward goal and ends beyond the line / between defenders),
  not StatsBomb's production metrics. 360 frames only include players inside the broadcast camera's visible area.
- xT grid is fitted on Euro 2024 only (16×12 cells, 6 iterations); values are tournament-specific.
- Per-90 leaderboards use a 180-minute floor; one tournament is a small sample — treat player-level rankings as indicative.
- Minutes come from the lineup position intervals (substitution-aware).

## Data attribution

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under its
[licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) — StatsBomb must be credited in any publication.

## Cover shadows (off-ball defending)

`python cover_shadow.py && python cover_shadow_charts.py` builds the blocked-passing-lane / "threat prevented" model
described in `docs/cover-shadows-euro2024.md` (method survey in `docs/defensive-positioning-metrics.md`).
Outputs: `out/cover_shadow_team.csv`, `_match.csv`, `_zone.csv`, `_team_depth.csv`, `_summary.json`, charts `out/cs_*.png`.
`python cover_shadow_players.py` (~11 min) tests whether blockers can be named; result: no (see the doc) — writes `out/cover_shadow_player_id_validation.json` and a small identified sample.
