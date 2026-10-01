import json, glob, os, math
import pandas as pd, numpy as np

matches = {m['match_id']: m for m in json.load(open('matches.json'))}
rows, shots, ff_rows = [], [], []
for p in sorted(glob.glob('events/*.json')):
    mid = int(os.path.basename(p)[:-5]); m = matches[mid]
    ev = json.load(open(p))
    ff = {f['event_uuid']: f for f in json.load(open(f'three-sixty/{mid}.json'))} if os.path.exists(f'three-sixty/{mid}.json') else {}
    for e in ev:
        t = e['type']['name']
        if t not in ('Pass', 'Carry', 'Shot', 'Pressure', 'Ball Receipt*', 'Dribble', 'Interception', 'Ball Recovery', 'Clearance', 'Duel', 'Block'):
            continue
        loc = e.get('location') or [np.nan, np.nan]
        r = dict(match_id=mid, stage=m['competition_stage']['name'], home=m['home_team']['home_team_name'], away=m['away_team']['away_team_name'],
                 id=e['id'], idx=e['index'], period=e['period'], minute=e['minute'], second=e['second'], type=t,
                 team=e['team']['name'], player=(e.get('player') or {}).get('name'), player_id=(e.get('player') or {}).get('id'),
                 position=(e.get('position') or {}).get('name'), possession=e['possession'], play_pattern=e['play_pattern']['name'],
                 x=loc[0], y=loc[1], under_pressure=e.get('under_pressure', False), duration=e.get('duration', np.nan))
        if t == 'Pass':
            ps = e['pass']; end = ps.get('end_location', [np.nan, np.nan])
            r.update(end_x=end[0], end_y=end[1], recipient=(ps.get('recipient') or {}).get('name'), recipient_id=(ps.get('recipient') or {}).get('id'),
                     length=ps.get('length'), angle=ps.get('angle'), height=(ps.get('height') or {}).get('name'),
                     body_part=(ps.get('body_part') or {}).get('name'), pass_type=(ps.get('type') or {}).get('name'),
                     outcome=(ps.get('outcome') or {}).get('name', 'Complete'), cross=ps.get('cross', False), switch=ps.get('switch', False),
                     through_ball=ps.get('through_ball', False), cut_back=ps.get('cut_back', False), shot_assist=ps.get('shot_assist', False), goal_assist=ps.get('goal_assist', False))
        elif t == 'Carry':
            end = e['carry']['end_location']; r.update(end_x=end[0], end_y=end[1])
        elif t == 'Shot':
            s = e['shot']; end = s.get('end_location', [np.nan, np.nan])
            r.update(end_x=end[0], end_y=end[1], xg=s['statsbomb_xg'], outcome=s['outcome']['name'], shot_type=s['type']['name'],
                     body_part=(s.get('body_part') or {}).get('name'), technique=(s.get('technique') or {}).get('name'), first_time=s.get('first_time', False))
        # 360 context
        f = ff.get(e['id'])
        if f:
            fr = f['freeze_frame']
            opp = [q['location'] for q in fr if not q['teammate']]
            tm = [q['location'] for q in fr if q['teammate'] and not q['actor']]
            r['ff_n'] = len(fr); r['ff_opp'] = len(opp); r['ff_tm'] = len(tm)
            if opp:
                ox = np.array(opp)
                d = np.hypot(ox[:, 0] - loc[0], ox[:, 1] - loc[1]); r['nearest_opp'] = d.min()
                r['opp_within5'] = int((d <= 5).sum())
                r['opp_goalside'] = int((ox[:, 0] > loc[0]).sum())  # opponents between ball and goal (attacking towards x=120)
                # defensive line = x of 2nd-deepest visible outfield opponent (exclude keeper)
                outf = np.array([q['location'] for q in fr if not q['teammate'] and not q['keeper']])
                if len(outf) >= 2:
                    xs = np.sort(outf[:, 0])[::-1]; r['def_line_x'] = xs[1]
                    r['opp_width'] = outf[:, 1].max() - outf[:, 1].min()
                    r['opp_depth'] = outf[:, 0].max() - outf[:, 0].min()
                if t == 'Pass' and not np.isnan(r.get('end_x', np.nan)):
                    ex_, ey_ = r['end_x'], r['end_y']
                    de = np.hypot(ox[:, 0] - ex_, ox[:, 1] - ey_); r['recv_nearest_opp'] = de.min()
                    r['recv_opp_goalside'] = int((ox[:, 0] > ex_).sum())
                    # line-breaking: completed, >=10% closer to goal, and passes behind >=1 outfield defender who was goalside of start (beats a defender) 
                    # with defenders on both sides of the pass line in y within 6 yards -> approximates "intersects a pair of defenders" OR passes the deepest def line
                    if len(outf) >= 2 and r['outcome'] == 'Complete':
                        d0 = 120 - loc[0]; d1 = 120 - ex_
                        closer = d1 <= 0.9 * d0
                        beaten = outf[(outf[:, 0] > loc[0] + 0.5) & (outf[:, 0] < ex_ - 0.5)]
                        # pairs: any two beaten defenders with the pass line (in y, interpolated) between them and within 12 yards of each other
                        pair = False
                        if len(beaten) >= 2 and ex_ != loc[0]:
                            for i in range(len(beaten)):
                                for j in range(i + 1, len(beaten)):
                                    a, b = beaten[i], beaten[j]
                                    if abs(a[0] - b[0]) <= 8 and abs(a[1] - b[1]) <= 15:
                                        xm = (a[0] + b[0]) / 2
                                        ym = loc[1] + (ey_ - loc[1]) * (xm - loc[0]) / (ex_ - loc[0])
                                        if min(a[1], b[1]) < ym < max(a[1], b[1]): pair = True
                        behind_line = ex_ > r['def_line_x'] + 1 and loc[0] < r['def_line_x']
                        r['line_breaking'] = bool(closer and (pair or behind_line))
                        r['defenders_beaten'] = len(beaten) if closer else 0
        rows.append(r)
df = pd.DataFrame(rows)
df.to_pickle("events_flat.pkl")
print(df.shape); print(df.type.value_counts()); print(df[df.type=='Pass'][['line_breaking','nearest_opp','def_line_x']].describe())
