# Does a "how good is a defender's positioning / how many passes do they prevent just by being there" stat exist?

Short answer: **yes, but only since ~2025, and none of it is a mainstream public stat.** Three families:

## 1. Cover shadows / blocked passing lanes — the closest match (and built on the same data we have)

**Cascioli, Wang et al. (KU Leuven + Belgian FA), "Quantifying Off-Ball Defensive Impact through Cover Shadows", Hudl Performance Insights 2025**
<https://lirias.kuleuven.be/retrieve/799f6a67-e48b-43c5-92a8-0f722d32951e>

- Data: **StatsBomb 360 freeze frames** (Serie A 2024/25 + Belgium–Portugal, Euro 2020) — exactly what Euro 2024 open data gives us.
- Scope: passes in the attacking half, foot passes, open play, outside the box, ball carrier has ≥1 forward option.
- Splits defenders into *man-markers* (within a 3 m radius centred 1 m goal-side of an attacker) and *lane-blockers* (everyone else); only lane-blockers get credit.
- **RQ1 "is this lane blocked?"** — baselines: line corridor (2 m wide), cone (width = 0.2 × pass length), Gaussian cone (defender = 2-D Gaussian, σ=1 m, blocked if ≥50 % mass in cone cross-section). Their own model, *Lane Control*, samples 30 points along centre/left/right lines of the cone and computes time-to-intercept for ball vs defenders vs attackers (pitch-control style). Geometric cone ≈ what we can do with frozen positions (360 has no velocities).
- **RQ2 "how much threat did the shadow remove?"** — threat of a snapshot = Σ over attackers of xT(location) × P(pass selected) × P(pass completes) (SoccerMap models, Voronoi cells per receiver). Delete a lane-blocking defender from the frame, re-score → **blocking score = threat(without) − threat(with)**. Example: 0.041 → 0.031 = 25 % of danger removed by shadows.
- Player outputs: **block rate** (share of forward passing lanes you were blocking) and **total xT prevented**. In Belgium–Portugal, CMs (Witsel, Tielemans, Palhinha, Moutinho) had highest block rates; full-backs (Meunier, T. Hazard) blocked rarely but prevented the most threat; CBs and forwards were low on both.
- Team result: Serie A 24/25 blocking score per match ranked **relegation-zone teams top, Bologna/Como bottom** — i.e. it measures style as much as quality.
- RQ3: Hungarian-algorithm repositioning of defenders into passing cones; threat drops in 75 % of 63k snapshots.
- Caveat they flag: 360 frames lack player IDs → per-player values need tracking data to identify who is who (they synced RBFA tracking for one match). In the open data we also have no IDs on freeze-frame players.

## 2. Vendor products (paid; not in open data)

- **Hudl Statsbomb Defensive Responsibility (DefR), June 2026** <https://www.hudl.com/blog/defensive-responsibility-defr-statsbomb> — for every opposition carry/pass/shot predicts which defender(s) *should* respond given location, attack momentum and defensive shape → "expected defensive actions", "defensive actions above expectation", and **Responsibility-Weighted OBV Conceded** (attacking value allowed through your zone). Event-based, aimed at "centre-backs are hard to evaluate". Fields live in `defensive_responsibility` on paid v10 events; absent from the open Euro 2024 files (checked).
- **SkillCorner Passing Options** — tracking-based receiver model; counts how often a player is a viable option through/around each defensive line. Attack-side mirror of what we want.
- **Opta Vision off-ball runs** — tracking; not pass-denial per se.

## 3. Research models on full tracking data (not reproducible on 360)

- **DEFCON** (Kim et al., SSAC 2026, Ajax tracking) — GNN component models; defensive value = reduction of opponent EPV, incl. "preventing" credit for making forward passes unlikely. Code: <https://github.com/hyunsungkim-ds/defcon>
- **Pass Suppression Value** (thesis, SkillCorner PL 24/25) — graph-attention pass-selection + threat models; mask defender→receiver attention to get a counterfactual, suppression weighted by xT; correlates with CB transfer value.
- **"Blame is easier than praise"** (Bischofberger, arXiv 2606.19931) — distributes event xT changes among defenders via defensive pressure areas and role-conditioned baselines.
- **Everett et al. 2025** — GAT influence on reception probabilities; MDP repositioning.
- **EF-OBSO** (StatsBomb conf. 2023) — counterfactual team-defence positioning on 360 data; team-level, OBSO-based.
- Precursors: Stöckl et al. 2021 "Making offensive play predictable" (Stats Perform, expected disruption), Fernández & Bornn 2018 pitch control / space generation.

## What we can build on Euro 2024 (open 360 data)

Reproduce the **geometric half of Cascioli et al.** — no velocities, no player IDs, but 51 matches × ~900 passes with frames:

1. For every open-play foot pass in the attacking half with the ball carrier visible: enumerate visible teammates as candidate receivers; mark each lane blocked/open with the Gaussian-cone test.
2. Threat of each option = our tournament xT at the receiver × a simple completion model (pass length, angle, nearest-defender distance) × a softmax selection model fitted on the passes actually played.
3. Counterfactual: remove each lane-blocking defender → **threat prevented**; sum per team per match and per frame-position (we can't name the defender, but we *can* name the defending team and the pitch zone/role slot).
4. Validation we can do: do teams with higher blocking score concede less xG/line-breaks? Does the pass actually played avoid the blocked lanes (i.e. are passers deterred)? Head-to-head: how often is the *most dangerous* visible receiver shadowed?

Story angles: "the passes that never happened" — which Euro 2024 defences erased the most danger just by standing in the right place; Spain's pressing shape vs England's low block; how shadows decay in extra time (fatigue hypothesis from the paper).
