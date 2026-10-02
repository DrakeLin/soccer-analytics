"""Can lane-blocking in anonymous 360 frames be attributed to named defenders?

Open 360 freeze frames carry no player ids. Two routes are tested:
  1. Direct identification: a defending-team on-ball event by player P within +-2 s and 2.5 yd of the
     frame position identifies that defender as P. Ids are then propagated along nearest-neighbour
     tracks between consecutive frames of the same possession (gap <= 6 s, move <= 6 yd, Hungarian
     one-to-one). Precision is checked against a looser direct id (+-8 s, 4 yd) where both exist.
  2. Hungarian matching of every visible defender to the on-pitch players, by distance to each
     player's match-level mean defensive-event location (team centroid aligned to the frame).
     Accuracy is scored on defenders identified by route 1 that had no on-ball event nearby.
Writes out/cover_shadow_blockers_identified.csv (route-1 named blockers), out/cover_shadow_players_sample.csv
(players with >= 5 identified blocks; a sample, not a leaderboard) and out/cover_shadow_player_id_validation.json.
"""
import json, glob, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

df = pd.read_pickle('events_flat.pkl')
Mt = pd.read_pickle('out/cover_shadow_frames.pkl')
I = pd.read_pickle('out/cover_shadow_blockers.pkl')
STRICT, LOOSE = (2, 2.5), (8, 4.0)
GAP_S, MOVE_YD = 6, 6.0
DEF = ['Pressure', 'Duel', 'Interception', 'Block', 'Clearance', 'Ball Recovery', 'Foul Committed']

def tsec(period, clock):
    m, s = clock.split(':'); return period * 10000 + int(m) * 60 + int(s)

onpitch, names, minutes = [], {}, {}
for p in glob.glob('lineups/*.json'):
    mid = int(p.split('/')[-1][:-5])
    for team in json.load(open(p)):
        for pl in team['lineup']:
            names[pl['player_id']] = pl['player_nickname'] or pl['player_name']
            for pos in pl['positions']:
                t1 = tsec(pos['to_period'], pos['to']) if pos['to'] else 9 * 10000
                onpitch.append((mid, team['team_name'], pl['player_id'], tsec(pos['from_period'], pos['from']), t1, pos['position']))
                a = int(pos['from'].split(':')[0]) + int(pos['from'].split(':')[1]) / 60
                b = int(pos['to'].split(':')[0]) + int(pos['to'].split(':')[1]) / 60 if pos['to'] else df[df.match_id == mid].minute.max() + 1
                minutes[pl['player_id']] = minutes.get(pl['player_id'], 0) + max(b - a, 0)
OP = pd.DataFrame(onpitch, columns=['match_id', 'team', 'player_id', 't0', 't1', 'position'])
pos_mode = OP.groupby('player_id').position.agg(lambda s: s.mode().iloc[0])

ev = df[df.x.notna() & df.player_id.notna()].copy()
ev['t'] = ev.period * 10000 + ev.minute * 60 + ev.second
ev['fx'] = 120 - ev.x; ev['fy'] = 80 - ev.y          # flip into the attacking team's frame
evg = {k: g for k, g in ev.groupby(['match_id', 'team'])}
ref_all = ev.groupby(['match_id', 'player_id'])[['fx', 'fy']].mean()
ref_def = ev[ev.type.isin(DEF)].groupby(['match_id', 'player_id'])[['fx', 'fy']].mean()

allf = {mid: {f['event_uuid']: f['freeze_frame'] for f in json.load(open(f'three-sixty/{mid}.json'))} for mid in Mt.match_id.unique()}
E = df[df.id.isin(set().union(*[set(v) for v in allf.values()]))][['id', 'match_id', 'team', 'home', 'away', 'period', 'minute', 'second', 'possession']].copy()
E['t'] = E.period * 10000 + E.minute * 60 + E.second
E['defending'] = np.where(E.team == E.home, E.away, E.home)
sec = df.set_index('id').second
Mt['t'] = Mt.period * 10000 + Mt.minute * 60 + Mt.id.map(sec)

def direct(mid, team, t, D, T, R):
    g = evg.get((mid, team)); g = g[(g.t - t).abs() <= T]
    out = []
    for x, y in D:
        d = np.hypot(g.fx - x, g.fy - y); ids = set(g.player_id[d <= R]); out.append(ids.pop() if len(ids) == 1 else None)
    return out

# ---- route 1: direct ids + propagation along short tracks ----
st = dict(defenders=0, strict=0, loose=0, propagated=0, agree=0, disagree=0)
ided, loose_id = {}, {}
for (mid, dteam), g in E.groupby(['match_id', 'defending']):
    seq = []
    for r in g.sort_values('t').itertuples():
        D = np.array([q['location'] for q in allf[mid][r.id] if not q['teammate'] and not q['keeper']]).reshape(-1, 2)
        if len(D) == 0: continue
        s = direct(mid, dteam, r.t, D, *STRICT)
        seq.append(dict(id=r.id, t=r.t, poss=r.possession, D=D, strict=s, loose=direct(mid, dteam, r.t, D, *LOOSE), prop=list(s)))
    def link(a, b):
        if b['poss'] != a['poss'] or b['t'] - a['t'] > GAP_S: return
        C = np.hypot(a['D'][:, None, 0] - b['D'][None, :, 0], a['D'][:, None, 1] - b['D'][None, :, 1])
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > MOVE_YD: continue
            if a['prop'][i] is not None and b['prop'][j] is None: b['prop'][j] = a['prop'][i]
            elif b['prop'][j] is not None and a['prop'][i] is None: a['prop'][i] = b['prop'][j]
    for _ in range(3):
        for k in range(len(seq) - 1): link(seq[k], seq[k + 1])
        for k in range(len(seq) - 1, 0, -1): link(seq[k - 1], seq[k])
    for s in seq:
        st['defenders'] += len(s['D'])
        for i in range(len(s['D'])):
            st['strict'] += s['strict'][i] is not None; st['loose'] += s['loose'][i] is not None; st['propagated'] += s['prop'][i] is not None
            if s['strict'][i] is None and s['loose'][i] is not None and s['prop'][i] is not None:
                st['agree' if s['prop'][i] == s['loose'][i] else 'disagree'] += 1
            if s['prop'][i] is not None: ided[(s['id'], i)] = s['prop'][i]
            if s['loose'][i] is not None: loose_id[(s['id'], i)] = s['loose'][i]

# ---- route 2: Hungarian on mean defensive-event location, scored on propagated ids w/o nearby on-ball event ----
hung = dict(n=0, player=0, group=0)
def GRP(p):
    if 'Center Back' in p: return 'CB'
    if 'Back' in p: return 'FB/WB'
    if 'Defensive Midfield' in p: return 'DM'
    if p in ('Right Midfield', 'Left Midfield', 'Right Wing', 'Left Wing'): return 'Wide'
    return 'CM/AM' if 'Midfield' in p else 'FW'
for m in Mt.itertuples():
    D = np.array([q['location'] for q in allf[m.match_id][m.id] if not q['teammate'] and not q['keeper']]).reshape(-1, 2)
    gt = [ided.get((m.id, i)) if (m.id, i) not in loose_id else None for i in range(len(D))]
    if not any(g is not None for g in gt): continue
    on = OP[(OP.match_id == m.match_id) & (OP.team == m.defending) & (OP.t0 <= m.t) & (OP.t1 > m.t) & (OP.position != 'Goalkeeper')].drop_duplicates('player_id')
    pids, slot = on.player_id.values, dict(zip(on.player_id, on.position))
    R = np.array([(ref_def if (m.match_id, p) in ref_def.index else ref_all).loc[(m.match_id, p)].values if (m.match_id, p) in ref_all.index else [np.nan, np.nan] for p in pids])
    ok = ~np.isnan(R[:, 0])
    if not ok.any(): continue
    P, pp = R[ok], pids[ok]; sh = D.mean(0) - P.mean(0)
    C = np.hypot(D[:, None, 0] - (P[None, :, 0] + sh[0]), D[:, None, 1] - (P[None, :, 1] + sh[1]))
    for a, b in zip(*linear_sum_assignment(C)):
        if gt[a] is not None and gt[a] in slot:
            hung['n'] += 1; hung['player'] += pp[b] == gt[a]; hung['group'] += GRP(slot[pp[b]]) == GRP(slot[gt[a]])

# ---- join identified defenders to lane-blockers ----
rows = []
for r in I.itertuples():
    D = [q['location'] for q in allf[r.match_id][r.id] if not q['teammate'] and not q['keeper']]
    k = next((i for i, (x, y) in enumerate(D) if abs(x - r.dx) < 1e-9 and abs(y - r.dy) < 1e-9), None)
    pid = ided.get((r.id, k)) if k is not None else None
    if pid is not None:
        rows.append(dict(id=r.id, match_id=r.match_id, defending=r.defending, player_id=pid, player=names[pid], position=pos_mode.get(pid),
                         dx=r.dx, dy=r.dy, lanes=r.lanes, score=r.score, depth_band=r.depth_band, lane=r.lane, source='direct' if (r.id, k) in loose_id else 'propagated'))
B = pd.DataFrame(rows)
B.round(5).to_csv('out/cover_shadow_blockers_identified.csv', index=False)
pl = B.groupby('player_id').agg(player=('player', 'first'), team=('defending', 'first'), position=('position', 'first'),
                                blocks=('score', 'size'), lanes=('lanes', 'sum'), prevented=('score', 'sum'))
pl['minutes'] = pd.Series(minutes); pl['prevented_per_block'] = pl.prevented / pl.blocks
pl = pl[pl.blocks >= 5].sort_values('blocks', ascending=False)
pl.round(4).to_csv('out/cover_shadow_players_sample.csv')

val = dict(frames_with_360=len(E), visible_outfield_defenders=st['defenders'],
           direct_strict=st['strict'], direct_loose=st['loose'], identified_after_propagation=st['propagated'],
           identified_share=round(st['propagated'] / st['defenders'], 4),
           propagation_precision_vs_loose_direct=round(st['agree'] / max(1, st['agree'] + st['disagree']), 3), propagation_checked=st['agree'] + st['disagree'],
           blockers_total=len(I), blockers_identified=len(B), blockers_identified_share=round(len(B) / len(I), 4),
           hungarian_scored_on=hung['n'], hungarian_player_accuracy=round(hung['player'] / max(1, hung['n']), 3), hungarian_role_group_accuracy=round(hung['group'] / max(1, hung['n']), 3),
           players_with_5plus_identified_blocks=int(len(pl)), params=dict(strict=STRICT, loose=LOOSE, gap_s=GAP_S, move_yd=MOVE_YD))
json.dump(val, open('out/cover_shadow_player_id_validation.json', 'w'), indent=1)
print(json.dumps(val, indent=1)); pd.set_option('display.width', 200); print(pl.head(20).round(3).to_string())
print(B.groupby('position').size().sort_values(ascending=False).head(10))
