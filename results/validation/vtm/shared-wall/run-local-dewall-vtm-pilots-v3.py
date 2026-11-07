"""Complete fresh-process torus solves, same main commit and solver settings."""
import json, os, signal, subprocess, time
from pathlib import Path
root = Path.cwd()
out = root / '.work/vtm/local-dewall-vtm-v3'
assert not out.exists()
out.mkdir()
rows = []
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.:.work/vtm', PYTHONDONTWRITEBYTECODE='1', WARP_CACHE_PATH='/opt/warpower-generated-cache')
for n in [1000, 4096]:
    for mode in ['control', 'candidate']:
        stem = out / f'torus-{n}-{mode}'
        cmd = [str(root / '.venv/bin/python')]
        cmd += ['-m', 'benchmarks.mesher'] if mode == 'control' else ['.work/vtm/run-local-dewall-mesher-v3.py']
        cmd += ['--n', str(n), '--repeats', '5' if n == 1000 else '3', '--output', str(stem.with_suffix('.json')), '--save-mesh', str(stem.with_suffix('.npz'))]
        start = time.time()
        with stem.with_suffix('.log').open('w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=10)
                code = 124
        row = dict(command=cmd, source=source, exit_code=code, wall_seconds=time.time()-start)
        if stem.with_suffix('.json').exists():
            row['result'] = json.loads(stem.with_suffix('.json').read_text())
        rows.append(row)
        (out / 'commands.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k != 'result'}), flush=True)
        assert code == 0
