"""Independent own-frame geometry and complete-solve schedule comparisons."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

w = Path('.work/vtm')
source = (w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source, str(w/'check-surface-bound-reuse-final-v2.py'), 'exec'))
out = w/'fixed-site-current-larger-v1'
dest = w/'fixed-site-current-larger-v1-independent.json'
assert not dest.exists()
commands = json.loads((out/'commands.json').read_text())
assert len(commands) == 4
report = dict(scope=__doc__, cases={}, comparisons=[])
input_digest = None
for command in commands:
    name = command['name']
    stem = out/name
    row = dict(command=command)
    report['cases'][name] = row
    if command.get('valid') != 1:
        row['excluded_reason'] = 'Invalid or failed solve; retained without quality claim'
        dest.write_text(json.dumps(report,indent=2)+'\n')
        continue
    assert command['exit_code'] == 0
    d = json.loads(stem.with_suffix('.json').read_text())
    m = np.load(stem.with_suffix('.npz'))
    digest = hashlib.sha256(m['input_points'].tobytes()+m['input_faces'].tobytes()).hexdigest()
    if input_digest is None:
        input_digest = digest
    assert digest == input_digest
    row.update(median_ms=d['median_gpu_milliseconds'], initial_ms=d['first_replay_gpu_milliseconds'], setup_seconds=d['setup_seconds'], independent=validate(stem.with_suffix('.npz')), input_sha256=digest, role_counts={str(int(k)):int(v) for k,v in zip(*np.unique(m['roles'],return_counts=True))}, work={k:d[k] for k in ['counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections','topology']})
    with tempfile.TemporaryDirectory(prefix='fixed-site-boundary-', dir='/opt/warpower-generated-cache') as temp:
        p = Path(temp)/'boundary.off'
        with p.open('w') as f:
            f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n')
            np.savetxt(f,m['points'],fmt='%.17g')
            np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
        result = subprocess.run([str(w/'surface-reference'),str(p)],text=True,capture_output=True,check=True)
        assert int(result.stdout.splitlines()[0].split()[-1]) == 0, (name,result.stdout)
        row['independent']['boundary_self_intersection_pairs'] = 0
        row['boundary_reference_stdout'] = result.stdout
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['median_ms'],row['independent']['quality']['radius_ratio_percentiles'],flush=True)
for n in [8192,16384]:
    a = report['cases'][f'torus{n}-progressive']
    for variant in ['fixed32']:
        b = report['cases'][f'torus{n}-{variant}']
        if 'independent' not in a or 'independent' not in b:
            continue
        qa,qb = [r['independent']['quality'] for r in [a,b]]
        flags = {f'radius_ratio_p{p}':qb['radius_ratio_percentiles'][p]>=qa['radius_ratio_percentiles'][p] for p in ['0','1','5','50','95']}
        flags.update(boundary_rms=qb['two_sided_area_weighted_rms']<=qa['two_sided_area_weighted_rms'],absolute_volume_error=abs(qb['relative_volume_error'])<=abs(qa['relative_volume_error']),minimum_dihedral=qb['dihedral_degrees_min']>=qa['dihedral_degrees_min'],maximum_dihedral=qb['dihedral_degrees_max']<=qa['dihedral_degrees_max'])
        report['comparisons'].append(dict(n=n,variant=variant,speedup=a['median_ms']/b['median_ms'],nondegrading_metrics=flags,all_checked_metrics_nondegrading=all(flags.values())))
report['complete'] = True
dest.write_text(json.dumps(report,indent=2)+'\n')
print(report['comparisons'],flush=True)
