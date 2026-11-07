"""Check observer meshes and its three repeated diagnostic histories."""
from pathlib import Path
import json, hashlib
import numpy as np

w = Path('.work/vtm')
helper = (w / 'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper, 'private-comparison-helper', 'exec'))
base = w / 'shared-wall-final/torus4096-after.npz'
old = np.load(base)
b = json.loads(base.with_suffix('.json').read_text())
rows = {}
for variant in ['v1', 'v2']:
    p = w / f'current-selection-observer-4096-{variant}.npz'
    cur = np.load(p)
    comparison = compare(old, cur)
    assert all(comparison[k] for k in ['points_bitwise_equal', 'roles_equal', 'full_tets_equal', 'material_tets_equal', 'oriented_boundary_equal'])
    d = json.loads((w / f'current-selection-observer-4096-{variant}-result.json').read_text())
    work = {k: b[k] == d[k] for k in ['counts', 'rounds', 'optimizer_steps', 'quality_moves', 'refinement_rejections', 'quality_rejections', 'reconstruction_routing']}
    assert all(work.values()) and d['valid'] == 1 and d['error'] == 0
    for key in ['input_points', 'input_faces']:
        np.testing.assert_array_equal(old[key], cur[key])
    rows[variant] = dict(mesh_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), comparison=comparison, work_equal=work)
o = json.loads((w / 'current-selection-observer-4096-v2.json').read_text())
a = np.array(o['instances'][0]['records'])
assert len(a) == 432
histories = []
for i in range(3):
    h = a[i * 144:(i + 1) * 144].copy()
    assert h[0, 2] == 0
    h[0, 1] = 0  # Previous-kind diagnostics are irrelevant after snapshot invalidation.
    histories.append(h)
    assert int(h[:, 2].sum()) == 11 and int((h[:, 0] == 1).sum()) == 15 and int((h[:, 0] == 2).sum()) == 129
    assert (h[h[:, 2] == 1, 3] == 1).all()
assert np.array_equal(histories[0], histories[1]) and np.array_equal(histories[0], histories[2])
report = dict(runtime='cc0a020f5f7c586370191d0246eb116c92cd0ac8', main='80d6ded503ac799048c3a82d17003951aa0f8aa6', baseline_mesh_sha256=hashlib.sha256(base.read_bytes()).hexdigest(), pairs=rows, evaluations_per_replay=144, select_evaluations_per_replay=15, check_evaluations_per_replay=129, identical_check_to_select_per_replay=11, all_repeated_outputs_bitwise_equal=True, three_replay_histories_equal_except_invalid_initial_previous_kind=True, scope='Observation only; every evaluation still executes. Both meshes match the independently validated baseline in positions/topology/work. v1 completed meshing but reporting failed on an expired weak reference; v2 stores static capacity during setup. An initial verifier compared the irrelevant previous-kind field across fresh resets; all effective observations match. No speed measurement.')
(w / 'current-selection-observer-independent-v2.json').write_text(json.dumps(report, indent=2) + '\n')
print(report)
