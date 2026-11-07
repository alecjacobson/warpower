"""Serial fresh-process reconstruction pilots; controls and candidate distinct."""
import json, os, signal, subprocess, time
from pathlib import Path
root = Path.cwd()
out = root / '.work/vtm/local-dewall-reconstruction-v2'
assert not out.exists()
out.mkdir()
rows = []
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.:.work/vtm', PYTHONDONTWRITEBYTECODE='1', WARP_CACHE_PATH='/opt/warpower-generated-cache')
for case, n in [('random', 1000), ('random', 4096), ('grid', 64), ('thin', 4096), ('cluster', 4096), ('random', 32768)]:
    for mode in ['control', 'candidate']:
        stem = out / f'{case}-{n}-{mode}'
        cmd = [str(root / '.venv/bin/python'), '.work/vtm/probe-local-dewall-full-v3.py', '--case', case, '--n', str(n), '--mode', mode, '--output', str(stem.with_suffix('.json'))]
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
        print(json.dumps(row), flush=True)
        assert code == 0
