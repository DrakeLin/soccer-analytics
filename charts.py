import json, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'sans-serif', 'font.size': 9})
df = pd.read_pickle('events_flat.pkl'); matches = json.load(open('matches.json'))
S = pd.read_csv('out/players.csv'); Tm = pd.read_csv('out/teams.csv')
P = df[df.type == 'Pass']
def save(fig, name):
    fig.savefig(f'out/{name}.svg', bbox_inches='tight', transparent=True); fig.savefig(f'out/{name}.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# 1. Kroos line-breaking passes map
k = P[(P.player == 'Toni Kroos')]
pitch = Pitch(pitch_type='statsbomb', line_color='#8a8f98', linewidth=1)
fig, ax = pitch.draw(figsize=(7, 4.6))
ok = k[(k.line_breaking == True)]
oth = k[(k.line_breaking == False) & (k.outcome == 'Complete') & k.pass_type.isna()]
pitch.lines(oth.x, oth.y, oth.end_x, oth.end_y, ax=ax, lw=0.6, color='#9aa3ad', alpha=0.25, comet=False)
pitch.lines(ok.x, ok.y, ok.end_x, ok.end_y, ax=ax, lw=1.2, color='#d9480f', alpha=0.85, comet=False)
pitch.scatter(ok.end_x, ok.end_y, ax=ax, s=10, color='#d9480f', zorder=3)
ax.set_title(f'Toni Kroos — {len(ok)} line-breaking passes (orange) out of {len(oth)+len(ok)} open-play completions, Euro 2024', fontsize=9, loc='left')
save(fig, 'kroos_lb')

# 2. One 360 freeze frame example: Kroos' longest line-breaking pass
ev_id = ok.sort_values('defenders_beaten', ascending=False).iloc[0]
ff = {f['event_uuid']: f for f in json.load(open(f'three-sixty/{ev_id.match_id}.json'))}[ev_id.id]
fig, ax = pitch.draw(figsize=(7, 4.6))
fr = pd.DataFrame([dict(x=q['location'][0], y=q['location'][1], tm=q['teammate'], actor=q['actor'], gk=q['keeper']) for q in ff['freeze_frame']])
va = np.array(ff['visible_area']).reshape(-1, 2)
ax.fill(va[:, 0], va[:, 1], color='#8a8f98', alpha=0.08, zorder=0)
pitch.scatter(fr[~fr.tm].x, fr[~fr.tm].y, ax=ax, s=90, color='#1c7ed6', edgecolors='white', zorder=3, label='Opponents')
pitch.scatter(fr[fr.tm & ~fr.actor].x, fr[fr.tm & ~fr.actor].y, ax=ax, s=90, color='#e8590c', edgecolors='white', zorder=3, label='Teammates')
pitch.scatter([ev_id.x], [ev_id.y], ax=ax, s=140, marker='*', color='#e8590c', edgecolors='black', zorder=4, label='Kroos')
pitch.lines(ev_id.x, ev_id.y, ev_id.end_x, ev_id.end_y, ax=ax, lw=2, color='#e8590c', comet=True, zorder=2)
ax.axvline(ev_id.def_line_x, color='#1c7ed6', ls='--', lw=1, alpha=0.6)
m = next(mm for mm in matches if mm['match_id'] == ev_id.match_id)
ax.set_title(f"360 freeze frame: {m['home_team']['home_team_name']} v {m['away_team']['away_team_name']}, {int(ev_id.minute)}' — Kroos beats {int(ev_id.defenders_beaten)} defenders (dashed = defensive line); shaded = camera-visible area", fontsize=8.5, loc='left')
ax.legend(loc='lower left', fontsize=7, frameon=False)
save(fig, 'kroos_360')

# 3. xT grid
xT = np.load('out/xT_grid.npy')
fig, ax = pitch.draw(figsize=(7, 4.6))
pitch.heatmap(dict(statistic=xT.T, x_grid=np.linspace(0, 120, xT.shape[0] + 1), y_grid=np.linspace(0, 80, xT.shape[1] + 1), cx=None, cy=None), ax=ax, cmap='Oranges', edgecolors='white', lw=0.3, alpha=0.9)
for i in range(xT.shape[0]):
    for j in range(xT.shape[1]):
        if xT[i, j] >= 0.03: ax.text((i + .5) * 120 / xT.shape[0], (j + .5) * 80 / xT.shape[1], f'{xT[i,j]:.2f}', ha='center', va='center', fontsize=5.5, color='black')
ax.set_title('Expected threat (xT) grid fitted on Euro 2024 passes, carries and shots — P(goal) from possessing the ball in each cell', fontsize=8.5, loc='left')
save(fig, 'xt_grid')

# 4. Spain pass network, final
fin = [mm for mm in matches if mm['competition_stage']['name'] == 'Final'][0]
sp = P[(P.match_id == fin['match_id']) & (P.team == 'Spain') & (P.outcome == 'Complete')]
subs = df[(df.match_id == fin['match_id']) & (df.team == 'Spain')]
first_sub = sp.minute.max()
lu = [t for t in json.load(open(f"lineups/{fin['match_id']}.json")) if t['team_name'] == 'Spain'][0]
starters = {pl['player_name']: pl['player_nickname'] or pl['player_name'] for pl in lu['lineup'] if pl['positions'] and pl['positions'][0]['start_reason'] == 'Starting XI'}
# minute of first Spain substitution
sub_ev = sorted(mm for mm in subs[subs.type == 'Pass'].minute.unique())
sp2 = sp[sp.player.isin(starters) & sp.recipient.isin(starters)]
import itertools
first_sub_min = 100
for t in json.load(open(f"lineups/{fin['match_id']}.json")):
    if t['team_name'] == 'Spain':
        for pl in t['lineup']:
            for pos in pl['positions']:
                if (pos['end_reason'] or '').startswith('Substitution - Off') and pos['to']:
                    a, b = pos['to'].split(':'); first_sub_min = min(first_sub_min, int(a))
sp2 = sp2[sp2.minute < first_sub_min]
loc = pd.concat([sp2[['player', 'x', 'y']], sp2[['recipient', 'end_x', 'end_y']].rename(columns={'recipient': 'player', 'end_x': 'x', 'end_y': 'y'})]).groupby('player').agg(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'))
pairs = sp2.groupby(['player', 'recipient']).size().reset_index(name='n'); pairs['key'] = pairs.apply(lambda r: tuple(sorted([r.player, r.recipient])), axis=1)
pairs = pairs.groupby('key').n.sum().reset_index(); pairs = pairs[pairs.n >= 3]
fig, ax = pitch.draw(figsize=(7, 4.6))
for _, r in pairs.iterrows():
    a, b = loc.loc[r.key[0]], loc.loc[r.key[1]]
    pitch.lines(a.x, a.y, b.x, b.y, ax=ax, lw=r.n * 0.35, color='#c2255c', alpha=0.45, zorder=1)
pitch.scatter(loc.x, loc.y, s=loc.n * 3.5, ax=ax, color='#c2255c', edgecolors='white', zorder=3)
for name, r in loc.iterrows(): ax.annotate(starters.get(name, name).split()[-1], (r.x, r.y), ha='center', va='center', fontsize=6.5, zorder=4, color='white', weight='bold')
ax.set_title(f"Spain pass network v England, Euro 2024 final (until first sub, {first_sub_min}') — node size = passes involved, line width = passes between pair (≥3)", fontsize=8.5, loc='left')
save(fig, 'spain_network')

# 5. Team def line vs xGA scatter (hand-drawn in matplotlib)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(Tm.def_line_height, Tm.xga_90, s=Tm.matches * 18, color='#1c7ed6', alpha=0.7, edgecolors='white')
for _, r in Tm.iterrows(): ax.annotate(r.team, (r.def_line_height, r.xga_90), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('Defensive line height (yards from own goal line, median at opposition pass)'); ax.set_ylabel('xG conceded per match')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.set_title('Who defended highest, and did it cost them? (bubble size = matches played)', fontsize=9, loc='left')
save(fig, 'defline_xga')

# 6. Receipts in space: forwards/attackers in final third — tight marking share
A = S[(S.f3_receipts >= 40)].copy()
A['tight_f3'] = 1 - A.f3_space5
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(A.f3_receipts_90, A.f3_space5, s=28, color='#2f9e44', alpha=0.75, edgecolors='white')
for _, r in A.sort_values('f3_receipts_90', ascending=False).head(28).iterrows(): ax.annotate(r.nick.split()[-1], (r.f3_receipts_90, r.f3_space5), fontsize=6.5, xytext=(3, 2), textcoords='offset points')
ax.set_xlabel('Final-third ball receipts per 90'); ax.set_ylabel('Share received with nearest defender ≥5 yds away')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.set_title('Who finds space in the final third? (players with ≥40 final-third receipts, ≥180 min)', fontsize=9, loc='left')
save(fig, 'receipts_space')
print(A.sort_values('f3_space5').head(10)[['nick','team','f3_receipts','f3_space5']].round(2).to_string())
print(A.sort_values('f3_space5', ascending=False).head(10)[['nick','team','f3_receipts','f3_space5']].round(2).to_string())
print('first sub', first_sub_min, 'pairs', len(pairs))
