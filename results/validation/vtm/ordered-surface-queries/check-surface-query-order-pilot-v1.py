"""Independently validate ordering pilot meshes and report exact trajectory parity without hiding differences."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

w = Path('.work/vtm')
helper = w/'check-surface-bound-reuse-final-v2.py'
exec(compile(helper.read_text().split("out=w/'surface-bound-reuse-final'")[0], str(helper), 'exec'))
out = w/'surface-query-order-pilot-v1'
dest = w/'surface-query-order-pilot-v1-independent.json'
assert not dest.exists()
commands = json.loads((out/'commands.json').read_text())
assert len(commands) == 10 and all(r['exit_code'] == 0 and r['quota_clean'] for r in commands)
report = dict(scope=__doc__, cases={}, complete=False, limits='Independent geometry validation and explicit parity measurements only. Any failed strict parity field requires investigation; this report alone does not authorize production adoption or establish native capture/lifetime safety.')


def paths(stem):
    return (stem/'result.json', stem/'mesh.npz', stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'), stem.with_suffix('.npz'), None)


def independent(path, vtu):
    result = validate(path, vtu)
    if vtu is None:
        with np.load(path) as m, tempfile.TemporaryDirectory(prefix='ordering-boundary-', dir='/opt/warpower-generated-cache') as tmp:
            p = Path(tmp)/'boundary.off'
            with p.open('w') as f:
                f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n')
                np.savetxt(f, m['points'], fmt='%.17g')
                np.savetxt(f, np.column_stack((np.full(len(m['boundary']), 3), m['boundary'])), fmt='%d')
            ref = subprocess.run([str(w/'surface-reference'), str(p)], text=True, capture_output=True, check=True)
            assert int(ref.stdout.splitlines()[0].split()[-1]) == 0, ref.stdout
            result.update(boundary_self_intersection_pairs=0, boundary_reference_stdout=ref.stdout)
    return result


def work(d):
    return {k: d[k] for k in ['counts', 'rounds', 'topology', 'quality_rejections', 'refinement_rejections']} | dict(steps=d.get('steps', d.get('optimizer_steps')), moves=d.get('moves', d.get('quality_moves')))


for name in ['torus16384-fixed32', 'thin-hand-seed7', 'armadillo-graded4096', 'armadillo-graded8192-recovery', 'armadillo-uniform8192-recovery']:
    all_paths = [paths(out/(name+'-'+phase)) for phase in ['before', 'after']]
    a, b = [json.loads(p[0].read_text()) for p in all_paths]
    params = lambda d: {k: v for k, v in d['parameters'].items() if k not in ['output', 'save_mesh', 'dot']}
    assert params(a) == params(b)
    for d in [a, b]:
        quota(d)
        assert d['valid'] == 1 and d['error'] == 0
    row = dict(parameters_equal=True, work_equal={k: work(a)[k] == work(b)[k] for k in work(a)}, independent={}, timing={})
    for phase, data, (_, mesh, vtu) in zip(['before', 'after'], [a, b], all_paths):
        row['independent'][phase] = independent(mesh, vtu)
        row['timing'][phase+'_ms'] = data.get('median_gpu_milliseconds', data.get('median_additional_replay_ms'))
        row['timing'][phase+'_setup_seconds'] = data['setup_seconds']
        if 'persistent_workspace_pool_bytes' in data:
            row['timing'][phase+'_pool_bytes'] = data['persistent_workspace_pool_bytes']
        else:
            observed = json.loads((out/(name+'-'+phase+'-recovery.json')).read_text())
            row['timing'][phase+'_pool_bytes'] = observed['persistent_workspace_pool_bytes']
    row['timing']['speedup'] = row['timing']['before_ms']/row['timing']['after_ms']
    row['timing']['additional_pool_bytes'] = row['timing']['after_pool_bytes']-row['timing']['before_pool_bytes']
    with np.load(all_paths[0][1]) as x, np.load(all_paths[1][1]) as y:
        d = compare(x, y)
        d['input_arrays_equal'] = {k: x[k].tobytes() == y[k].tobytes() for k in ['input_points', 'input_faces']}
        d['changed_scalar_coordinates'] = int(np.count_nonzero(x['points'].view(np.uint32) != y['points'].view(np.uint32)))
        if 'radius_ratio' in x and d['material_tets_equal']:
            def quality(z):
                cells = np.sort(z['tetrahedra'], axis=1)
                return z['radius_ratio'][np.lexsort(cells.T[::-1])]
            qa, qb = quality(x), quality(y)
            d['maximum_stored_quality_difference'] = float(np.max(np.abs(qa-qb)))
            d['minimum_quality_before'] = float(qa.min())
            d['minimum_quality_after'] = float(qb.min())
        row['comparison'] = d
    row['strict_trajectory_parity'] = all(row['work_equal'].values()) and all(d['input_arrays_equal'].values()) and all(d[k] for k in ['points_bitwise_equal', 'roles_equal', 'full_tets_equal', 'material_tets_equal', 'oriented_boundary_equal'])
    report['cases'][name] = row
    dest.write_text(json.dumps(report, indent=2)+'\n')
    print(name, row['timing'], 'strict_parity', row['strict_trajectory_parity'], flush=True)
report['complete'] = True
report['strict_trajectory_parity'] = all(c['strict_trajectory_parity'] for c in report['cases'].values())
dest.write_text(json.dumps(report, indent=2)+'\n')
