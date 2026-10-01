"""Cover-shadow / blocked-passing-lane model on Euro 2024 StatsBomb 360 frames.

Approximates the geometric half of Cascioli et al. (2025), "Quantifying Off-Ball
Defensive Impact through Cover Shadows":
  * eligible snapshots: open-play foot passes (not High), origin between halfway
    line and attacking box (x in [60,102)), actor visible in the 360 frame
  * options  = visible teammates of the passer
  * defenders within 3 m of a point 1 m goal-side of any attacker = man-markers;
    everyone else = potential lane-blocker
  * Gaussian-cone lane test: defender ~ N(pos, 1m^2 I), cone width 0.2*pass length,
    lane blocked if any defender's mass inside the cone cross-section >= 0.5
  * threat(frame) = sum_r xT(r) * P(select r) * P(complete r)
      P(select)   : conditional-logit over options (fit on Euro 2024)
      P(complete) : logistic regression on executed passes (fit on Euro 2024)
  * counterfactual: delete lane-blocking defenders that block >=1 lane, re-score
      blocking score = threat_cf - threat_obs  (>0 => cover shadows helped)
No velocities, no player identities (open 360 frames are anonymous), so results
are aggregated by defending team / match / pitch zone, not by named defender.
Writes out/cover_shadow_*.csv|json and out/cover_shadow_options.pkl.
"""
import json, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

YD = 1 / 0.9144           # yards per metre (pitch is 120x80 yards)
SIG, K, TAU = 1.0 * YD, 0.2, 0.5
MM_R, MM_OFF = 3.0 * YD, 1.0 * YD
MATCH_R = 6.0             # yards: actual pass end must be this close to a visible teammate

df = pd.read_pickle('events_flat.pkl')
xT = np.load('out/xT_grid.npy'); NX, NY = xT.shape

def xt_at(x, y):
    ix = np.clip((np.asarray(x) / 120 * NX).astype(int), 0, NX - 1)
    iy = np.clip((np.asarray(y) / 80 * NY).astype(int), 0, NY - 1)
    return xT[ix, iy]

P = df[(df.type == 'Pass') & df.ff_n.notna() & df.pass_type.isna() & (df.height != 'High Pass')
       & df.body_part.isin(['Right Foot', 'Left Foot']) & (df.x >= 60) & (df.x < 102)].copy()
P['defending'] = np.where(P.team == P.home, P.away, P.home)
P['complete'] = P.outcome == 'Complete'

frames = {}
for mid in P.match_id.unique():
    frames.update({f['event_uuid']: f for f in json.load(open(f'three-sixty/{mid}.json'))})

def lane_mass(a, r, D):
    """Gaussian mass of each defender inside the cone cross-section at its projection on lane a->r."""
    v = r - a; L = np.hypot(*v)
    if L < 1e-6 or len(D) == 0:
        return np.zeros(len(D))
    u = v / L; n = np.array([-u[1], u[0]])
    rel = D - a; t = rel @ u; dp = rel @ n
    h = K * t / 2
    F = norm.cdf((h - dp) / SIG) - norm.cdf((-h - dp) / SIG)
    F[(t <= 0) | (t >= L)] = 0
    return F

def man_markers(A, D):
    """bool per defender: within MM_R of a point MM_OFF goal-side (higher x) of any attacker."""
    if len(D) == 0: return np.zeros(0, bool)
    tgt = A + np.array([MM_OFF, 0])
    d = np.hypot(D[:, None, 0] - tgt[None, :, 0], D[:, None, 1] - tgt[None, :, 1])
    return d.min(1) <= MM_R

FEATS = ['dist', 'dx', 'fmax', 'ncone', 'rnear', 'rgoalside', 'rx', 'ry_c', 'xt']

def option_feats(a, R, D):
    """feature matrix (n_options, len(FEATS)) + per-option lane mass per defender (n_opt, n_def)."""
    n = len(R); M = np.zeros((n, len(D)))
    for i in range(n):
        M[i] = lane_mass(a, R[i], D)
    dist = np.hypot(*(R - a).T)
    if len(D):
        dd = np.hypot(R[:, None, 0] - D[None, :, 0], R[:, None, 1] - D[None, :, 1])
        rnear = dd.min(1); rgs = (D[None, :, 0] > R[:, None, 0]).sum(1)
    else:
        rnear = np.full(n, 30.0); rgs = np.zeros(n)
    X = np.column_stack([dist / 20, (R[:, 0] - a[0]) / 20, M.max(1) if len(D) else np.zeros(n),
                         (M >= 0.1).sum(1) if len(D) else np.zeros(n), np.minimum(rnear, 30) / 10,
                         rgs / 5, R[:, 0] / 120, np.abs(R[:, 1] - 40) / 40, xt_at(R[:, 0], R[:, 1]) * 10])
    return X, M

rows = []   # one row per (frame, option)
meta = []   # one row per frame
for r in P.itertuples():
    f = frames.get(r.id)
    if f is None: continue
    ff = f['freeze_frame']
    act = [q for q in ff if q['actor']]
    if not act: continue
    a = np.array(act[0]['location'])
    R = np.array([q['location'] for q in ff if q['teammate'] and not q['actor']])
    D = np.array([q['location'] for q in ff if not q['teammate']]).reshape(-1, 2)
    if len(R) == 0: continue
    A = np.vstack([a[None], R])
    mm = man_markers(A, D)
    X, M = option_feats(a, R, D)
    blocked = (M >= TAU).any(1) if len(D) else np.zeros(len(R), bool)
    blocked_lb = (M[:, ~mm] >= TAU).any(1) if (~mm).any() else np.zeros(len(R), bool)
    # actual recipient = nearest visible teammate to pass end point
    end = np.array([r.end_x, r.end_y])
    de = np.hypot(*(R - end).T); j = int(de.argmin()); matched = de[j] <= MATCH_R
    nblk = int(((M >= TAU)[:, ~mm]).any(0).sum()) if (~mm).any() else 0
    meta.append(dict(id=r.id, match_id=r.match_id, period=r.period, minute=r.minute, team=r.team, defending=r.defending,
                     stage=r.stage, ax=a[0], ay=a[1], n_opt=len(R), n_def=len(D), n_mm=int(mm.sum()),
                     n_lanes_blocked=int(blocked.sum()), n_blockers=nblk, matched=matched, recip=j if matched else -1,
                     complete=r.complete, under_pressure=bool(r.under_pressure)))
    for i in range(len(R)):
        rows.append((r.id, i, *X[i], blocked[i], blocked_lb[i], i == j and matched))
    # stash arrays for counterfactual pass
    meta[-1]['arr_a'] = a; meta[-1]['arr_R'] = R; meta[-1]['arr_D'] = D; meta[-1]['arr_mm'] = mm; meta[-1]['arr_M'] = M

O = pd.DataFrame(rows, columns=['id', 'opt', *FEATS, 'blocked', 'blocked_lb', 'chosen'])
Mt = pd.DataFrame(meta)
print(f'frames {len(Mt)}  options {len(O)}  matched recipient {Mt.matched.mean():.3f}  '
      f'lanes blocked {O.blocked.mean():.3f} (by lane-blockers {O.blocked_lb.mean():.3f})  '
      f'man-markers/frame {Mt.n_mm.mean():.2f} of {Mt.n_def.mean():.2f} defenders')

# ---------- pass selection: conditional logit over visible options ----------
SEL = ['dist', 'dx', 'fmax', 'ncone', 'rnear', 'rgoalside', 'rx', 'ry_c', 'xt']
ids = Mt.loc[Mt.matched, 'id'].values
Osel = O[O.id.isin(ids)]
grp = Osel.groupby('id', sort=False)
maxo = grp.size().max(); nf = len(ids)
Xs = np.zeros((nf, maxo, len(SEL))); mask = np.zeros((nf, maxo), bool); ysel = np.zeros(nf, int)
pos = {i: k for k, i in enumerate(ids)}
for gid, g in grp:
    k = pos[gid]; m = len(g)
    Xs[k, :m] = g[SEL].values; mask[k, :m] = True; ysel[k] = int(np.flatnonzero(g.chosen.values)[0])
mu, sd = Xs[mask].mean(0), Xs[mask].std(0) + 1e-9
Xs = (Xs - mu) / sd

def nll(w):
    s = Xs @ w; s[~mask] = -1e9
    s -= s.max(1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return -lp[np.arange(nf), ysel].mean() + 1e-3 * (w @ w)

w = minimize(nll, np.zeros(len(SEL)), method='L-BFGS-B').x
print('selection coef', dict(zip(SEL, w.round(2))), ' NLL', round(nll(w), 3),
      ' chance NLL', round(float(np.log(mask.sum(1)).mean()), 3))

def p_select(X):
    s = ((X - mu) / sd) @ w; s -= s.max(); e = np.exp(s); return e / e.sum()

# ---------- pass success: logistic on executed passes ----------
SUC = ['dist', 'dx', 'fmax', 'ncone', 'rnear', 'rgoalside', 'rx', 'ry_c']
chosen = Osel[Osel.chosen].merge(Mt[['id', 'complete', 'under_pressure']], on='id')
Xc = np.column_stack([chosen[SUC].values, chosen.under_pressure.values.astype(float)])
clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, chosen.complete.values)
print('success coef', dict(zip(SUC + ['pressure'], clf.coef_[0].round(2))),
      ' base rate', round(chosen.complete.mean(), 3), ' acc', round(clf.score(Xc, chosen.complete.values), 3))

def p_success(X, up):
    return clf.predict_proba(np.column_stack([X[:, :len(SUC)], np.full(len(X), float(up))]))[:, 1]

def frame_threat(a, R, D, up):
    """returns (total threat, P(select) per option)."""
    X, _ = option_feats(a, R, D)
    ps = p_select(X)
    return float((xt_at(R[:, 0], R[:, 1]) * ps * p_success(X, up)).sum()), ps

# ---------- counterfactuals ----------
T_obs, T_cf, deterred, indiv = [], [], [], []
for m in Mt.itertuples():
    a, R, D, mm, M = m.arr_a, m.arr_R, m.arr_D, m.arr_mm, m.arr_M
    t0, ps0 = frame_threat(a, R, D, m.under_pressure)
    blockers = np.flatnonzero((~mm) & (M >= TAU).any(0)) if len(D) else np.array([], int)
    if len(blockers):
        t1, ps1 = frame_threat(a, R, np.delete(D, blockers, 0), m.under_pressure)
        blk = (M[:, blockers] >= TAU).any(1)
        det = float((ps1 - ps0)[blk].sum())   # expected passes into those lanes that were deterred
        for j in blockers:
            tj, _ = frame_threat(a, R, np.delete(D, j, 0), m.under_pressure)
            indiv.append(dict(id=m.id, match_id=m.match_id, defending=m.defending, period=m.period,
                              dx=D[j, 0], dy=D[j, 1], lanes=int((M[:, j] >= TAU).sum()), score=tj - t0))
    else:
        t1, det = t0, 0.0
    T_obs.append(t0); T_cf.append(t1); deterred.append(det)
Mt['threat_obs'] = T_obs; Mt['threat_cf'] = T_cf; Mt['blocking_score'] = Mt.threat_cf - Mt.threat_obs
Mt['passes_deterred'] = deterred
I = pd.DataFrame(indiv)
# defender zone, in the DEFENDING team's frame: depth = distance from own goal line
I['depth_yd'] = 120 - I.dx
I['depth_band'] = pd.cut(I.depth_yd, [0, 18, 30, 45, 120], labels=['box', 'edge of box', 'mid block', 'high'])
I['lane'] = pd.cut(I.dy, [0, 26.67, 53.33, 80], labels=['right (def view)', 'centre', 'left (def view)'])

Mt.drop(columns=[c for c in Mt.columns if c.startswith('arr_')]).to_pickle('out/cover_shadow_frames.pkl')
O.to_pickle('out/cover_shadow_options.pkl'); I.to_pickle('out/cover_shadow_blockers.pkl')

# ---------- aggregates ----------
lanes = O.merge(Mt[['id', 'defending', 'match_id']], on='id')
team = Mt.groupby('defending').agg(matches=('match_id', 'nunique'), frames=('id', 'size'),
                                   frames_with_block=('n_blockers', lambda s: (s > 0).mean()),
                                   threat_obs=('threat_obs', 'sum'), threat_cf=('threat_cf', 'sum'),
                                   prevented=('blocking_score', 'sum'), passes_deterred=('passes_deterred', 'sum'))
team['deterred_per100'] = team.passes_deterred / team.frames * 100
team['lanes'] = lanes.groupby('defending').size()
team['blocking_rate'] = lanes.groupby('defending').blocked_lb.mean()
team['prevented_per100'] = team.prevented / team.frames * 100
team['pct_reduction'] = team.prevented / team.threat_cf * 100
shots = df[df.type == 'Shot'].copy(); shots['conceding'] = np.where(shots.team == shots.home, shots.away, shots.home)
team['xga_per_match'] = shots.groupby('conceding').xg.sum() / team.matches
team['goals_against'] = shots[shots.outcome == 'Goal'].groupby('conceding').size().reindex(team.index).fillna(0).astype(int)
team = team.sort_values('prevented_per100', ascending=False).round(4)
team.to_csv('out/cover_shadow_team.csv')

match = Mt.groupby(['match_id', 'defending']).agg(frames=('id', 'size'), prevented=('blocking_score', 'sum'),
                                                   threat_cf=('threat_cf', 'sum')).reset_index()
match['blocking_rate'] = lanes.groupby(['match_id', 'defending']).blocked_lb.mean().values
mx = shots.groupby(['match_id', 'conceding']).xg.sum().rename('xga').reset_index().rename(columns={'conceding': 'defending'})
match = match.merge(mx, how='left').fillna({'xga': 0})
match['prevented_per100'] = match.prevented / match.frames * 100
match.round(4).to_csv('out/cover_shadow_match.csv', index=False)

zone = I.groupby(['depth_band', 'lane'], observed=True).agg(blockers=('score', 'size'), lanes=('lanes', 'sum'),
                                                             prevented=('score', 'sum'), mean_score=('score', 'mean')).reset_index()
zone.round(5).to_csv('out/cover_shadow_zone.csv', index=False)
team_zone = I.groupby(['defending', 'depth_band'], observed=True).score.sum().unstack().fillna(0)
team_zone.round(4).to_csv('out/cover_shadow_team_depth.csv')

# deterrence / validity checks
chk = O.merge(Mt.loc[Mt.matched, ['id', 'complete']], on='id')
sel_rate = chk.groupby('blocked_lb').chosen.mean()
n_blocked_played = int((chk.chosen & chk.blocked_lb).sum())
comp_rate = chk[chk.chosen].groupby('blocked_lb').complete.mean()
fwd = chk[chk.dx > 0.25]  # forward options (>5 yd)
sel_fwd = fwd.groupby('blocked_lb').chosen.mean()
corr = match[['prevented_per100', 'blocking_rate', 'xga']].corr().loc['xga']
et = Mt.groupby(Mt.period >= 3).agg(frames=('id', 'size'), prevented=('blocking_score', 'mean'),
                                    frames_with_block=('n_blockers', lambda s: (s > 0).mean()))
summary = dict(
    frames=int(len(Mt)), options=int(len(O)), matched_share=float(Mt.matched.mean()),
    lanes_blocked_share=float(O.blocked.mean()), lanes_blocked_by_lane_blockers=float(O.blocked_lb.mean()),
    man_markers_per_frame=float(Mt.n_mm.mean()), defenders_per_frame=float(Mt.n_def.mean()),
    frames_with_any_blocker=float((Mt.n_blockers > 0).mean()),
    selection_rate_open=float(sel_rate.get(False, np.nan)), selection_rate_blocked=float(sel_rate.get(True, np.nan)),
    selection_rate_fwd_open=float(sel_fwd.get(False, np.nan)), selection_rate_fwd_blocked=float(sel_fwd.get(True, np.nan)),
    completion_open=float(comp_rate.get(False, np.nan)), completion_blocked=float(comp_rate.get(True, np.nan)),
    n_passes_into_blocked_lanes=n_blocked_played, n_passes_matched=int(chk.chosen.sum()),
    passes_deterred_total=float(Mt.passes_deterred.sum()), passes_deterred_per100=float(Mt.passes_deterred.mean() * 100),
    mean_threat_obs=float(Mt.threat_obs.mean()), mean_threat_cf=float(Mt.threat_cf.mean()),
    mean_blocking_score=float(Mt.blocking_score.mean()), pct_reduction=float(Mt.blocking_score.sum() / Mt.threat_cf.sum() * 100),
    corr_xga_prevented_per100=float(corr['prevented_per100']), corr_xga_blocking_rate=float(corr['blocking_rate']),
    extra_time=et.to_dict(), selection_coef=dict(zip(SEL, map(float, w))), success_coef=dict(zip(SUC + ['pressure'], map(float, clf.coef_[0]))))
json.dump(summary, open('out/cover_shadow_summary.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
print(team[['matches', 'frames', 'blocking_rate', 'deterred_per100', 'prevented_per100', 'pct_reduction', 'xga_per_match']].to_string())
print(zone.to_string())
