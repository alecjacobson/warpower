import json
from pathlib import Path
import numpy as np

w = Path('.work/vtm')
commands = json.loads((w / 'seed-pairs-commands.json').read_text())
assert len(commands) == 18 and all(r['exit_code'] == 0 and r['quota_clean'] for r in commands)


def canonical(t):
    t = np.sort(t, axis=1)
    return t[np.lexsort(t.T[::-1])]


def oriented(t):
    return sorted(min(tuple(f), tuple(np.roll(f, 1)), tuple(np.roll(f, 2))) for f in t)


rows = []
for name in dict.fromkeys(r['name'] for r in commands):
    data, meshes = [], []
    for phase in ['control', 'prepared']:
        stem = w / f'seed-pairs-{phase}-{name}'
        data.append(json.loads((stem.with_suffix('.json') if name.startswith('torus') else stem / 'result.json').read_text()))
        meshes.append(np.load(stem.with_suffix('.npz') if name.startswith('torus') else stem / 'mesh.npz'))
    a, b = meshes
    row = dict(name=name, points_bitwise_equal=bool(np.array_equal(a['points'].view(np.uint32), b['points'].view(np.uint32))), max_coordinate_change=float(np.max(np.abs(a['points'].astype(float)-b['points']))))
    for key in ['tetrahedra', 'full_tetrahedra']:
        row[key] = bool(np.array_equal(canonical(a[key]), canonical(b[key])))
    row['oriented_boundary'] = oriented(a['boundary']) == oriented(b['boundary'])
    row['roles'] = bool(np.array_equal(a['roles'], b['roles']))
    assert all(row[k] for k in ['tetrahedra', 'full_tetrahedra', 'oriented_boundary', 'roles']), row
    work = ['rounds', 'optimizer_steps', 'quality_moves'] if name.startswith('torus') else ['rounds', 'steps', 'moves']
    assert all(data[0][k] == data[1][k] for k in work)
    row['same_iteration_work'] = True
    for phase, d in zip(['control', 'prepared'], data):
        row[phase+'_ms'] = d.get('median_gpu_milliseconds', d.get('median_additional_replay_ms'))
    row['speedup'] = row['control_ms'] / row['prepared_ms']
    rows.append(row)
    print(json.dumps(row), flush=True)
(w / 'seed-pairs-comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
