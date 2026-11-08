"""Existing-API fresh-solve schedule pilot; no concurrent CPU geometry validation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path.cwd()
w = root / '.work/vtm'
out = w / 'fixed-site-current-pilot-v1'
assert not out.exists()
out.mkdir()
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert source == 'b44ab9e7332551a7a8a1c9bf9697f7bf8d7c109c'
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
rows = []
for n in [1000, 4096]:
    for label, initial, growth, polish in [('progressive', 256, 5, 20), ('fixed20', n, 0, 20), ('fixed32', n, 0, 32)]:
        name = f'torus{n}-{label}'
        stem = out / name
        cmd = [str(root / '.venv/bin/python'), '-m', 'benchmarks.mesher', '--case', 'torus', '--n', str(n), '--initial', str(initial), '--growth-iterations', str(growth), '--polish-iterations', str(polish), '--selective-iterations', '10', '--quality-rounds', '4', '--repeats', '3', '--seed', '2026', '--subdivisions', '4', '--save-mesh', str(stem.with_suffix('.npz')), '--output', str(stem.with_suffix('.json'))]
        start = time.time()
        with stem.with_suffix('.log').open('w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
                code = 124
        row = dict(name=name, command=cmd, source=source, exit_code=code, timeout_seconds=300, wall_seconds=time.time()-start)
        if stem.with_suffix('.json').exists():
            d = json.loads(stem.with_suffix('.json').read_text())
            row.update(valid=d['valid'], error=d['error'], counts=d['counts'], median_ms=d['median_gpu_milliseconds'], quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[k].values()), result_sha256=hashlib.sha256(stem.with_suffix('.json').read_bytes()).hexdigest())
        rows.append(row)
        (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(row, flush=True)
        assert code != 0 or row.get('valid') == 1
