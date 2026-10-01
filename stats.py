import json, glob, os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
os.makedirs('out', exist_ok=True)
df = pd.read_pickle('events_flat.pkl')
matches = json.load(open('matches.json'))
# ---------- minutes played ----------
def mmss(s):
    a, b = s.split(':'); return int(a) + int(b) / 60
mins = []
for p in glob.glob('lineups/*.json'):
    mid = int(os.path.basename(p)[:-5])
    endt = df[df.match_id == mid].minute.max() + 1
    for t in json.load(open(p)):
        for pl in t['lineup']:
            tot = 0
            for pos in pl['positions']:
                fr = mmss(pos['from']); to = mmss(pos['to']) if pos['to'] else endt
                tot += max(0, to - fr)
            if tot: mins.append((mid, t['team_name'], pl['player_id'], pl['player_name'], pl['player_nickname'] or pl['player_name'], tot))
mins = pd.DataFrame(mins, columns=['match_id', 'team', 'player_id', 'player', 'nick', 'mins'])
pm = mins.groupby(['player_id', 'player', 'nick', 'team'], as_index=False).mins.sum()
pm['apps'] = mins.groupby('player_id').size().reindex(pm.player_id).values
# ---------- xT grid fitted on this tournament ----------
NX, NY = 16, 12
def cell(x, y):
    return (np.clip((x / 120 * NX).astype(int), 0, NX - 1), np.clip((y / 80 * NY).astype(int), 0, NY - 1))
op = df[df.play_pattern.isin(['Regular Play', 'From Counter', 'From Keeper', 'From Throw In', 'From Goal Kick', 'From Free Kick', 'From Corner', 'From Kick Off'])]
mv = df[df.type.isin(['Pass', 'Carry']) & df.end_x.notna() & (df.pass_type.isna() | (df.pass_type == 'Recovery'))].copy()
mv = mv[(mv.type == 'Carry') | (mv.outcome == 'Complete')]
sh = df[df.type == 'Shot']
cx, cy = cell(mv.x.values, mv.y.values); ex, ey = cell(mv.end_x.values, mv.end_y.values)
sx, sy = cell(sh.x.values, sh.y.values)
move_cnt = np.zeros((NX, NY)); shot_cnt = np.zeros((NX, NY)); goal_cnt = np.zeros((NX, NY))
np.add.at(move_cnt, (cx, cy), 1); np.add.at(shot_cnt, (sx, sy), 1); np.add.at(goal_cnt, (sx, sy), (sh.outcome == 'Goal').values.astype(float))
xg_sum = np.zeros((NX, NY)); np.add.at(xg_sum, (sx, sy), sh.xg.values)
T = np.zeros((NX, NY, NX, NY)); np.add.at(T, (cx, cy, ex, ey), 1)
tot = move_cnt + shot_cnt
p_shot = np.divide(shot_cnt, tot, out=np.zeros_like(tot), where=tot > 0); p_move = 1 - p_shot
g = np.divide(xg_sum, shot_cnt, out=np.zeros_like(tot), where=shot_cnt > 0)  # avg xG per shot from cell
Tn = T / np.maximum(move_cnt[:, :, None, None], 1)
xT = np.zeros((NX, NY))
for _ in range(6):
    xT = p_shot * g + p_move * np.einsum('ijkl,kl->ij', Tn, xT)
np.save('out/xT_grid.npy', xT)
mv['xT_gain'] = xT[ex, ey] - xT[cx, cy]
mv['xT_gain_pos'] = mv.xT_gain.clip(lower=0)
# ---------- player stats ----------
P = df[(df.type == 'Pass')].copy()
P['open'] = P.pass_type.isna() | (P.pass_type == 'Recovery')
P['complete'] = P.outcome == 'Complete'
P['prog'] = P.complete & ((120 - P.end_x) <= 0.75 * (120 - P.x)) & (P.end_x > P.x + 5) & (P.x > 40)
P['lb'] = P.line_breaking == True
P['ff_ok'] = P.ff_opp >= 5
R = df[df.type == 'Ball Receipt*'].copy(); R['ff_ok'] = R.ff_opp >= 5
R = R[R.ff_ok]
R['tight'] = R.nearest_opp <= 2; R['space5'] = R.nearest_opp >= 5; R['space10'] = R.nearest_opp >= 10
R['final3'] = R.x >= 80
agg = P.groupby('player_id').agg(team=('team', 'first'), passes=('id', 'size'), comp=('complete', 'mean'),
    open_passes=('open', 'sum'), prog=('prog', 'sum'), lb=('lb', 'sum'),
    up_passes=('under_pressure', 'sum'), up_comp=('complete', lambda s: s[P.loc[s.index, 'under_pressure']].mean() if P.loc[s.index, 'under_pressure'].any() else np.nan),
    rel_space=('nearest_opp', 'mean'))
lbc = P[P.ff_ok & P.open & P.complete].groupby('player_id').lb.mean().rename('lb_share')
xt = mv.groupby('player_id').agg(xT=('xT_gain_pos', 'sum'), xT_pass=('xT_gain_pos', lambda s: s[mv.loc[s.index, 'type'] == 'Pass'].sum()), xT_carry=('xT_gain_pos', lambda s: s[mv.loc[s.index, 'type'] == 'Carry'].sum()))
rc = R.groupby('player_id').agg(receipts=('id', 'size'), tight_share=('tight', 'mean'), space5_share=('space5', 'mean'), space10_share=('space10', 'mean'),
    recv_space=('nearest_opp', 'median'), f3_receipts=('final3', 'sum'),
    f3_space5=('nearest_opp', lambda s: (s[R.loc[s.index, 'final3']] >= 5).mean() if R.loc[s.index, 'final3'].any() else np.nan))
shots = sh.groupby('player_id').agg(shots=('id', 'size'), xg=('xg', 'sum'), goals=('outcome', lambda s: (s == 'Goal').sum()))
S = pm.set_index('player_id').join([agg.drop(columns='team'), lbc, xt, rc, shots])
S = S[S.mins >= 180].copy()
for c in ['passes', 'prog', 'lb', 'xT', 'xT_pass', 'xT_carry', 'receipts', 'f3_receipts', 'shots', 'xg']:
    S[c + '_90'] = S[c] / S.mins * 90
S = S.reset_index().sort_values('lb_90', ascending=False)
S.to_csv('out/players.csv', index=False)
# ---------- team stats ----------
rows = []
teams = sorted(df.team.unique())
for t in teams:
    own = df[df.team == t]; oppev = df[(df.team != t) & (df.match_id.isin(own.match_id.unique()))]
    opp_pass = oppev[(oppev.type == 'Pass') & (oppev.ff_opp >= 7) & oppev.pass_type.isna() & (oppev.x < 90)]  # opp passing in build-up/midfield, our block visible
    defl = 120 - opp_pass.def_line_x  # distance of our 2nd-deepest defender from our own goal line
    width = opp_pass.opp_width; depth = opp_pass.opp_depth
    own_pass = P[P.team == t]; opp_P = P[P.team != t][P.match_id.isin(own.match_id.unique())]
    poss = own_pass.shape[0] / (own_pass.shape[0] + opp_P.shape[0])
    # PPDA: opponent passes in their own 60% of pitch / our defensive actions there
    opp_build = opp_P[opp_P.x < 72]
    defacts = own[own.type.isin(['Pressure', 'Duel', 'Interception', 'Ball Recovery', 'Block']) & (own.x > 48)]
    ppda = opp_build.shape[0] / max(1, defacts.shape[0])
    Rt = R[R.team == t]; Ro = R[(R.team != t) & R.match_id.isin(own.match_id.unique())]
    xgf = sh[sh.team == t].xg.sum(); xga = sh[(sh.team != t) & sh.match_id.isin(own.match_id.unique())].xg.sum()
    n = own.match_id.nunique()
    rows.append(dict(team=t, matches=n, possession=poss, def_line_height=defl.median(), block_width=width.median(), block_depth=depth.median(), ppda=ppda,
        lb_90=own_pass.lb.sum() / n, lb_share=own_pass[own_pass.ff_ok & own_pass.open & own_pass.complete].lb.mean(),
        lb_allowed_90=opp_P.lb.sum() / n,
        recv_space_f3=Rt[Rt.final3].space5.mean(), recv_space_allowed_f3=Ro[Ro.final3].space5.mean(),
        xT_90=mv[mv.team == t].xT_gain_pos.sum() / n, xg_90=xgf / n, xga_90=xga / n,
        goals=sum(m['home_score'] if m['home_team']['home_team_name'] == t else m['away_score'] for m in matches if t in (m['home_team']['home_team_name'], m['away_team']['away_team_name']))))
Tm = pd.DataFrame(rows).sort_values('def_line_height', ascending=False)
Tm.to_csv('out/teams.csv', index=False)
print(Tm.round(2).to_string())
print(S[['nick', 'team', 'mins', 'lb', 'lb_90', 'lb_share', 'xT_90', 'prog_90', 'recv_space', 'space5_share', 'up_comp']].head(20).round(2).to_string())
