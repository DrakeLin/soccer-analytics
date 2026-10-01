# Pass & positioning stats we can build from Euro 2024 (StatsBomb events + 360)

## Standard stats analysts publish (event data)
1. **xG / xG-chain / xGBuildup** — shot quality; who is involved in possessions that end in shots.
2. **Expected Threat (xT)** — value of moving the ball from zone A to zone B; ranks progressive passers/carriers.
3. **VAEP / OBV-style possession value** — ML model valuing every action by change in P(score) − P(concede).
4. **Progressive passes/carries, passes into final third / box, zone-14 entries, switches, through balls, cut-backs.**
5. **Pass networks** — avg positions + pass volumes between players; centralisation, "who is the hub".
6. **PPDA, high turnovers, pressures in final third, counterpress recoveries within 5s** — pressing intensity.
7. **Field tilt, possession share by zone, direct-speed (yards gained/sec in possession), passes per sequence** — style metrics.
8. **Shot maps, pass sonars, heatmaps, carry maps.**

## 360-only stats (need player positions) — rarer, more interesting
9. **Line-breaking passes** — completed passes that go ≥10% closer to goal and pass between/behind a pair of defenders (StatsBomb's own definition). Who breaks lines, who receives behind lines.
10. **Receipts in space** — distance to nearest defender when receiving (0–2, 2–5, 5–10, 10+ yds). "Who finds space?" / "Who is marked tightest?"
11. **Defensive line height & compactness** — x of the 2nd-deepest defender at each opposition pass; team width/depth block shape. "Who defends highest? Who is most compact?"
12. **Defenders bypassed per pass / carry** — number of opponents goalside before vs after; "players removed from the game".
13. **Pressure context on the passer** — nearest-opponent distance at pass release vs completion %, by player ("who is calm under pressure").
14. **Passing-lane availability** — number of open teammates (no defender within cone) at each pass; decision quality (took the best option?).
15. **Space creation off-ball** — teammates' isolation at receipt relative to prior frame (who drags markers).
16. **Shot freeze-frame metrics** — defenders in shot cone, GK positioning, shooting under pressure vs xG.
17. **Rest-defence shape** — positions of non-attacking players at the moment of shots/turnovers; counter-attack vulnerability.
18. **Dribble isolation** — 1v1 situations: attacker vs nearest defender distance, success rate.

## Side-Quest-style story angles
- "Spain won Euro 2024 by finding space, not by passing more" — receipts-in-space vs possession.
- "Who actually breaks lines?" — line-breaking passes per 90 vs reputation (Kroos, Rodri, Wirtz…).
- "The highest line in Europe" — defensive-line heights vs goals conceded on counters.
- "Marked men" — which forwards face the tightest marking, and does it work?
- Interactive: pick a team → pass network + block shape; pick a player → receipt-space profile.
