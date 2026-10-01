"""Charts for the cover-shadow analysis (run after cover_shadow.py)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch

team = pd.read_csv('out/cover_shadow_team.csv', index_col=0)
match = pd.read_csv('out/cover_shadow_match.csv')
zone = pd.read_csv('out/cover_shadow_zone.csv')
Mt = pd.read_pickle('out/cover_shadow_frames.pkl')
O = pd.read_pickle('out/cover_shadow_options.pkl')
S = json.load(open('out/cover_shadow_summary.json'))

# 1. team ranking: threat prevented per 100 opponent passes
t = team.sort_values('prevented_per100')
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(t.index, t.prevented_per100, color='#2c7fb8')
for i, (v, br) in enumerate(zip(t.prevented_per100, t.blocking_rate)):
    ax.text(v + 0.002, i, f'{br*100:.0f}% lanes blocked', va='center', fontsize=7, color='#555')
ax.set_xlabel('xT prevented by cover shadows per 100 opponent open-play passes (own half → box)')
ax.set_title('Euro 2024: who took away the most danger just by standing in passing lanes?')
fig.tight_layout(); fig.savefig('out/cs_team_rank.png', dpi=150); plt.close(fig)

# 2. blocking rate vs xGA per match (team), threat prevented vs xGA (match)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
ax.scatter(team.prevented_per100, team.xga_per_match, s=team.matches * 15, color='#2c7fb8', alpha=.8)
for n, r in team.iterrows():
    ax.annotate(n, (r.prevented_per100, r.xga_per_match), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('xT prevented per 100 opponent passes'); ax.set_ylabel('xG conceded per match')
ax.set_title(f'Team level (r = {team[["prevented_per100", "xga_per_match"]].corr().iloc[0, 1]:.2f}; size = matches)')
ax = axs[1]
ax.scatter(match.prevented_per100, match.xga, color='#7fcdbb', alpha=.7)
ax.set_xlabel('xT prevented per 100 opponent passes'); ax.set_ylabel('xG conceded in match')
ax.set_title(f'Match level, n={len(match)} team-matches (r = {S["corr_xga_prevented_per100"]:.2f})')
fig.tight_layout(); fig.savefig('out/cs_vs_xga.png', dpi=150); plt.close(fig)

# 3. deterrence: selection + completion rate, open vs blocked lanes
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Open lane', 'Blocked lane']
sel = [S['selection_rate_fwd_open'] * 100, S['selection_rate_fwd_blocked'] * 100]
comp = [S['completion_open'] * 100, S['completion_blocked'] * 100]
x = np.arange(2); wdt = .35
ax.bar(x - wdt / 2, sel, wdt, label='P(passer picks this forward option)', color='#2c7fb8')
ax.bar(x + wdt / 2, comp, wdt, label='Completion % when played anyway', color='#f03b20')
for i in range(2):
    ax.text(x[i] - wdt / 2, sel[i] + 1, f'{sel[i]:.1f}%', ha='center', fontsize=9)
    ax.text(x[i] + wdt / 2, comp[i] + 1, f'{comp[i]:.1f}%', ha='center', fontsize=9)
ax.set_xticks(x, labels); ax.set_ylabel('%'); ax.legend(fontsize=8, loc='center right'); ax.set_ylim(0, 110)
ax.set_title('Standing in the lane works: passers avoid it, and passes into it fail more')
fig.tight_layout(); fig.savefig('out/cs_deterrence.png', dpi=150); plt.close(fig)

# 4. where lane-blocking happens (defending team's view, own goal at left)
I = pd.read_pickle('out/cover_shadow_blockers.pkl')
pitch = Pitch(pitch_type='statsbomb', line_color='#444')
fig, ax = pitch.draw(figsize=(9, 6))
# flip to defending-team orientation: own goal at x=0
bx, by = 120 - I.dx, 80 - I.dy
stat = pitch.bin_statistic(bx, by, values=I.score, statistic='sum', bins=(12, 8))
pcm = pitch.heatmap(stat, ax=ax, cmap='Reds', edgecolors='#ffffff', linewidth=.3)
fig.colorbar(pcm, ax=ax, shrink=.7, label='total xT prevented by lane-blockers standing here')
ax.set_title('Where cover shadows earn their keep (defending team attacks →, own goal on the left)', fontsize=11)
fig.savefig('out/cs_zone_heatmap.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 5. example frame: biggest single blocking score with >= 3 options
ex = Mt[(Mt.n_opt >= 4) & (Mt.n_blockers >= 1)].sort_values('blocking_score', ascending=False).iloc[0]
frames = {f['event_uuid']: f for f in json.load(open(f'three-sixty/{ex.match_id}.json'))}
ff = frames[ex.id]['freeze_frame']
a = np.array([q['location'] for q in ff if q['actor']][0])
R = np.array([q['location'] for q in ff if q['teammate'] and not q['actor']])
D = np.array([q['location'] for q in ff if not q['teammate']])
opts = O[O.id == ex.id].sort_values('opt')
fig, ax = pitch.draw(figsize=(9, 6))
for i, (r, o) in enumerate(zip(R, opts.itertuples())):
    L = np.hypot(*(r - a)); u = (r - a) / L; n = np.array([-u[1], u[0]]); h = 0.2 * L / 2
    poly = np.array([a, r + n * h, r - n * h])
    ax.fill(poly[:, 0], poly[:, 1], color='#f03b20' if o.blocked else '#31a354', alpha=.18)
    ax.plot([a[0], r[0]], [a[1], r[1]], color='#f03b20' if o.blocked else '#31a354', lw=1.4, ls='--' if o.blocked else '-')
pitch.scatter(R[:, 0], R[:, 1], ax=ax, color='#1f78b4', s=120, zorder=5, label='teammates (options)')
pitch.scatter(D[:, 0], D[:, 1], ax=ax, color='#e31a1c', s=120, zorder=5, marker='s', label='defenders')
pitch.scatter([a[0]], [a[1]], ax=ax, color='black', s=160, zorder=6, marker='*', label='passer')
ax.legend(loc='lower left', fontsize=8)
ax.set_title(f'{ex.team} pass vs {ex.defending}, min {ex.minute}: red lanes blocked by cover shadows\n'
             f'threat with defenders {ex.threat_obs:.3f} xT → without lane-blockers {ex.threat_cf:.3f} '
             f'(blocking score {ex.blocking_score:+.3f})', fontsize=10)
fig.savefig('out/cs_example_frame.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('charts written')
