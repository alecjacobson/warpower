"""Fresh complete-solve control/prototype pairs; retain all raw outputs."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
w = root / '.work/vtm'
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert source == 'a84e274425fcc4f0a85e00bbc1e2b61648888894'
rows = []
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.')
cases = {'torus1000', 'torus4096', 'graded-torus', 'thin-hand', 'cavity', 'disconnected', 'counterbored-plate', 'armadillo-uniform', 'armadillo-graded'}
for old in json.loads((w / 'boundary-timing-commands.json').read_text()):
    if old['phase'] != 'after' or old['name'] not in cases:
        continue
    name = old['name']
    for phase in ['control', 'prepared']:
        stem = w / f'seed-pairs-{phase}-{name}'
        assert not stem.with_suffix('.log').exists(), stem
        cmd = [s.replace(f'boundary-after-{name}', f'seed-pairs-{phase}-{name}') for s in old['command']]
        current_env = dict(env)
        if phase == 'prepared':
            module = cmd[2]
            cmd[1:3] = [str(w / 'run-seed-prepared.py')]
            if module == 'examples.vtm_gallery':
                current_env['WARPOWER_PREPARED_GALLERY'] = '1'
        start = time.time()
        with stem.with_suffix('.log').open('w') as log:
            result = subprocess.run(cmd, cwd=root, env=current_env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=name, phase=phase, command=cmd, source=source, seconds=time.time()-start, exit_code=result.returncode, environment={k: current_env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'PYTHONPATH', 'WARPOWER_PREPARED_GALLERY'] if k in current_env})
        rows.append(row)
        (w / 'seed-pairs-commands.json').write_text(json.dumps(rows, indent=2)+'\n')
        assert result.returncode == 0, row
        data = json.loads((stem.with_suffix('.json') if name.startswith('torus') else stem / 'result.json').read_text())
        row['quota_clean'] = all(v['throttled_periods'] == 0 for k in ['cpu_quota_delta', 'initial_replays_cpu_quota_delta', 'additional_replays_cpu_quota_delta', 'first_replay_cpu_quota_delta'] if k in data for v in data[k].values())
        row['ms'] = data.get('median_gpu_milliseconds', data.get('median_additional_replay_ms'))
        (w / 'seed-pairs-commands.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(name, phase, row['ms'], 'quota_clean', row['quota_clean'], flush=True)
        assert row['quota_clean'], row
        assert data['valid'] == 1 and data['error'] == 0, data
