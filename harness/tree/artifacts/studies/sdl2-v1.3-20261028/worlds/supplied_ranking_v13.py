"""SDL-2 v1.3 (worlds/supplied_ranking_v13.py, from v1.2's supplied_model_ranking.py): the shipped C7.3 model's own P(all 3 bodies meet 10 kOe / 11 MGOe) for every declared route, both paths.

This is what an agent gets by deferring to the released model before any evidence; it is world-independent (conditioned
on the released P7 record only). Used for the defer-to-model shortcut fixture, for stratification and as the BO
baseline's prior mean. The route maps to the tool's keys exactly as the world maps a request (c7_campaign.c7_route).
CPU only, $0. Usage: python -B supplied_model_ranking.py SHARD NSHARDS
"""
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'env/models-agent/supplied/C7.3'   # v1.3: the shipped package (world rows removed)
sys.path.insert(0, str(P))
import ndfeb_physics_tool as tool  # noqa: E402
from ndfeb_sps.models import c7_campaign as c7, sdl2_v12 as v  # noqa: E402

MODE = {'on_heating': 'heating', 'on_cooling': 'cooling', 'on_dwell': 'dwell'}
KEY = {0: 'none', 1: '900', 2: '900+520'}


def tool_route(r):
    steps = v.ANNEAL_PROGRAMS[r['anneal']]
    out = {'T_peak_C': r['temperature_c'], 'dwell_min': r['hold_time_s'] / 60.0, 'pressure_mode': MODE[r['pressure_schedule']],
           'anneal': KEY[len(steps)], 'degas': 0.0 if r['preparation'] == 'none' else 1.0,
           'feedstock_d50_um': r['feedstock_d50_um'], 'heating_rate_C_per_min': r['heating_rate_c_min'],
           'P_MPa': r['pressure_mpa'], 'current_path': r['current_path']}
    if r['pressure_full_at_c'] is not None:
        out['load_ramp_min'] = (r['temperature_c'] - r['pressure_full_at_c']) / c7.COOLING_RATE_C_MIN
    if steps:
        out['anneal_hold_h'] = steps[0][2] / 3600.0
    return out


if __name__ == '__main__':
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    pm = tool.load()
    spec = {'Hci_kOe_min': v.REQUIREMENT['hci_kOe'], 'BHmax_MGOe_min': v.REQUIREMENT['bhmax_MGOe']}
    path = Path(__file__).resolve().parent / 'ranking' / f'supplied_ranking_v13_{shard}.jsonl'
    done = {json.loads(l)['id'] for l in path.open()} if path.exists() else set()
    out = path.open('a')
    for i, r in enumerate(v.declared_routes()):
        if i % n != shard or v.route_id(r) in done:
            continue
        jt = pm.joint_target(tool_route(r), spec=spec)
        row = {'id': v.route_id(r), 'route': r}
        for p, key in (('link', 'coercivity_link'), ('c5', 'c5')):
            q = jt['paths'][key]
            row[p] = {'p_all': q['p_all_3_bodies_x_1_preparation'], 'p_one': q['p_one_body'], 'medians': q['medians']}
        out.write(json.dumps(row) + '\n'); out.flush()
