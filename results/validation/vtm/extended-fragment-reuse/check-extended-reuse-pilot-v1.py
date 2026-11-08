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
out = w/'extended-reuse-pilot-v1'
dest = w/'extended-reuse-pilot-v1-independent.json'
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
controls = json.loads((w/'fixed-site-current-larger-v1-independent.json').read_text())
for name, row in report['cases'].items():
    before=controls['cases'][name]
    if 'independent' not in row or 'independent' not in before:
        continue
    with np.load(w/'fixed-site-current-larger-v1'/(name+'.npz')) as a, np.load(out/(name+'.npz')) as b:
        comparison=compare(a,b)
    row['control_comparison']=dict(comparison=comparison,work_equal={k:before['work'][k]==row['work'][k] for k in row['work']},speedup=before['median_ms']/row['median_ms'])
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['control_comparison'],flush=True)
for name,row in report['cases'].items():
    if 'control_comparison' not in row:
        continue
    c=row['control_comparison']
    assert all(c['work_equal'].values()),name
    assert all(c['comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),name
    assert c['comparison']['maximum_coordinate_difference']<=1e-8,name
report['complete']=True
dest.write_text(json.dumps(report,indent=2)+'\n')
