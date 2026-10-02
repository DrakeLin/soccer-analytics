# Cover shadows at Euro 2024 — how much do defenders prevent just by standing there?

Code: `cover_shadow.py` (model) and `cover_shadow_charts.py` (figures). Outputs in `out/cover_shadow_*.csv|json`, `out/cs_*.png`.
Method follows the geometric half of Cascioli et al. 2025, *Quantifying Off-Ball Defensive Impact through Cover Shadows*
(see `defensive-positioning-metrics.md`); everything here is a **custom approximation on open data**, not an official StatsBomb/Hudl metric.

## What we measure

For every open-play, non-high **foot pass** started between the halfway line and the attacking box (16,714 passes with a 360 frame):

1. **Options** = every visible teammate of the passer (117,943 options, ~7 per pass).
2. **Man-markers** = defenders within 3 m of a point 1 m goal-side of any attacker (≈2.3 of the ≈9.5 visible defenders). They are *not* credited as lane-blockers.
3. **Blocked lane** (Gaussian cone): defender ≈ N(position, 1 m²), cone from passer to receiver widening to 0.2 × pass length; lane blocked if any defender has ≥ 50 % of his mass inside the cone cross-section.
   25 % of all lanes are blocked; 19 % are blocked by a lane-blocker (not a man-marker). 73 % of passes have at least one lane-blocker in the frame.
4. **Threat of the situation** = Σ over options of xT(receiver) × P(passer picks this option) × P(pass completes).
   P(pick) is a conditional logit over visible options; P(complete) a logistic regression on executed passes; both fit on Euro 2024 only (features: distance, forward gain, Gaussian lane mass, defenders in cone, nearest defender to receiver, goal-side defenders, location). xT is the 16×12 grid fitted in `stats.py`.
5. **Counterfactual**: delete every lane-blocker who blocks ≥ 1 lane, re-score. **Blocking score = threat without blockers − threat observed.**
   Also computed per individual blocker (delete only him) and attributed to the zone he stood in.
6. **Passes deterred** = Σ over blocked options of (P(pick) without blockers − P(pick) observed): the expected number of passes that would have been played into those lanes.

## Headline results

**Does standing in the lane actually stop passes?** Yes, overwhelmingly.

| | Open lane | Blocked lane |
|---|---|---|
| P(passer picks this forward option) | 11.6 % | 0.3 % |
| Completion when played anyway | 96.1 % | 65 % (n = 40) |

Only 40 of 11,311 matched passes went into a lane a defender was standing in. Across the tournament ≈ **950 passes were deterred by cover shadows, 5.7 per 100 open-play passes** in this zone — i.e. roughly one pass in eighteen never happened because a defender was in the way.

**How much danger does that remove?** Mean threat per snapshot is 0.0130 xT with the defenders present vs 0.0137 without the lane-blockers: cover shadows remove **≈ 4.9 % of the available passing threat** in this zone (Cascioli et al. report ~25 % for one Serie A example, but their threat model is a CNN over the whole pitch, ours is receiver-level and far coarser).

**Team ranking** (xT prevented per 100 opponent passes; `out/cover_shadow_team.csv`, `out/cs_team_rank.png`):

| Rank | Team | prevented /100 | % of threat removed | passes deterred /100 | xGA / match |
|---|---|---|---|---|---|
| 1 | Serbia | 0.098 | 7.3 | 5.7 | 0.89 |
| 2 | Croatia | 0.096 | 7.5 | 8.5 | 1.29 |
| 3 | Romania | 0.086 | 6.6 | 6.2 | 1.42 |
| 4 | Spain | 0.085 | 6.3 | 5.8 | 0.92 |
| 5 | Italy | 0.084 | 5.8 | 5.8 | 1.04 |
| 7 | Germany | 0.083 | 6.0 | 5.8 | 0.84 |
| 17 | England | 0.056 | 4.2 | 5.1 | 1.32 |
| 18 | Portugal | 0.055 | 3.8 | 4.4 | 2.10 |
| 24 | Netherlands | 0.043 | 3.5 | 5.0 | 0.98 |

Blocking *rate* barely varies (17–21 % of lanes), the *value* of the blocks does: Serbia and Croatia (two of the tournament's compact, deep mid-blocks) and the champions Spain are at the top; Portugal and the Netherlands at the bottom.

**Where it matters** (`out/cs_zone_heatmap.png`, `out/cover_shadow_zone.csv`): almost all the value comes from blockers standing 10–30 yd in front of their own goal, in the central channel — the classic screening midfielder/centre-back spot. Lane-blockers in their own attacking half have a slightly *negative* mean score (removing them does not create danger; they just push selection toward other low-value options).

## Validation / honesty

- **vs xG conceded**: team-level r = −0.26 (more threat prevented ↔ fewer xG conceded, right sign, 24 teams); match-level r = +0.15 (102 team-matches, wrong sign, essentially noise). This is a *process* metric, not a results predictor — same as the paper, which only validates deterrence (pass choice / completion), not goals.
- **Extra time** (567 snapshots): blocking score per snapshot rises from 0.00067 to 0.00088 and 77 % of frames have a blocker vs 73 % — consistent with tired teams dropping deeper, but a thin sample.
- Recipient matching: only 68 % of passes have the receiver inside the visible 360 area; selection/completion models are fit on those. Unmatched passes still get scored.

## Which players? (attempted, not defensible)

`cover_shadow_players.py` tries to put names on the anonymous blockers (`out/cover_shadow_player_id_validation.json`).

1. **Direct id + short tracks.** A defender is named when the defending team has an on-ball event by one player within ±2 s and 2.5 yd of the frame position; ids are then carried along nearest-neighbour tracks between consecutive frames of the same possession (≤6 s gap, ≤6 yd move). Precision is 90 % where a looser direct id (±8 s, 4 yd) can check it (16,756 cases), but coverage is 6.8 % of the 1.29 M visible outfield defenders and only **661 of 23,734 lane-blockers (2.8 %)** — 39 players reach five identified blocks. That is a sample (`out/cover_shadow_blockers_identified.csv`, `out/cover_shadow_players_sample.csv`), not a ranking: it is biased toward players who touch the ball soon after standing in a lane (holding midfielders dominate — Freuler, Kiteishvili, Kochorashvili, Gnezda Čerin, Rice, Rodri).
2. **Hungarian matching to mean event positions.** Every visible defender is assigned to one of the eleven on-pitch players by distance to the player's match-level mean defensive-event location (team centroid aligned to the frame). Scored on 3,741 route-1 ids with no on-ball event nearby, it names the right player **37.5 %** of the time and even the right role group (CB / FB / DM / CM / wide / FW) only 46 %. Variants (all events, nominal formation slots, mixes, with/without centroid shift) scored 11–32 %. Compact defensive blocks put several players within a few yards of each other and sparse on-ball events are a poor proxy for where someone stands off the ball, so this is unusable for a per-player stat.

Conclusion: with open 360 data, "who is good at blocking lanes" is answerable at **team and zone** level only. A named leaderboard needs tracking data (or StatsBomb's non-open 360 with player ids / DefR).

## Caveats (read before quoting)

- Freeze frames are **anonymous** — see "Which players?" below: we could not recover identities reliably, so there is no player leaderboard. Aggregates are by team / match / zone.
- No velocities: the paper's physics-based Lane Control model (time-to-intercept) cannot be reproduced; we use the Gaussian-cone geometric baseline only.
- Only visible players count; options outside the camera frame are ignored, so threat is underestimated for switches of play.
- The completion model has a 95.9 % base rate (ground passes in midfield rarely fail), so counterfactual gains come mostly through pass *selection*, not completion.
- xT is fitted on 51 matches; P(pick)/P(complete) are simple linear models with a handful of features, not SoccerMap CNNs.
- Sample: 51 matches, 3–7 per team. Team ranks are indicative; the top-vs-bottom gap (0.098 vs 0.043 per 100 passes) is large relative to within-team match variance but nothing here is significance-tested.

Data: StatsBomb Open Data (Euro 2024, events + 360). Please credit StatsBomb if you publish any of this.
