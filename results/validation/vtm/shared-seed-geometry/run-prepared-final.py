"""Validation for integrated geometry sharing, serial GPU/CPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd(); w = root / '.work/vtm'; python = root / '.venv/bin/python'
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert not subprocess.check_output(['git', 'status', '--short'], text=True)
baseline = w / 'extraction-geometry'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=baseline, text=True).strip() == 'a84e274425fcc4f0a85e00bbc1e2b61648888894'
assert not subprocess.check_output(['git', 'status', '--short'], cwd=baseline, text=True)
rows = []


def run(name, cmd, cwd=root):
    cmd = list(map(str, cmd)); start = time.time()
    assert not (w / f'prepared-{name}.log').exists()
    with (w / f'prepared-{name}.log').open('w') as log:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = dict(name=name, command=cmd, cwd=str(cwd), seconds=time.time()-start, exit_code=result.returncode, source=source if cwd == root else 'a84e274425fcc4f0a85e00bbc1e2b61648888894')
    rows.append(row)
    (w / 'prepared-final-commands.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(name, result.returncode, row['seconds'], flush=True)
    assert result.returncode == 0, row


def quota(path):
    d = json.loads(path.read_text())
    for k in ['cpu_quota_delta', 'initial_replays_cpu_quota_delta', 'additional_replays_cpu_quota_delta', 'first_replay_cpu_quota_delta']:
        if k in d:
            assert all(v['throttled_periods'] == 0 for v in d[k].values()), (path, k, d[k])


old_rows = json.loads((w / 'boundary-timing-commands.json').read_text())
for old in old_rows:
    name = old['name']
    if old['phase'] != 'after':
        continue
    standalone = name.startswith('delaunay') or name.startswith('diagram')
    for phase in (['before', 'final'] if standalone else ['final']):
        stem = f'prepared-{phase}-{name}'
        cmd = [s.replace(f'boundary-after-{name}', stem) for s in old['command']]
        run(f'{phase}-{name}', cmd, baseline if phase == 'before' else root)
        result_path = w / (stem + '.json') if standalone or name.startswith('torus') else w / stem / 'result.json'
        if not standalone:
            quota(result_path)

run('full-suite', [python, '-m', 'pytest', '-q'])
mem = ['/usr/local/cuda-12.9/bin/compute-sanitizer', '--tool', 'memcheck', '--padding', '32', '--error-exitcode', '99']
tests = ['-q', 'tests/test_facet_seeds.py', 'tests/test_reuse_delaunay.py']
run('memcheck', mem + [python, '-m', 'benchmarks.memcheck', '--stack-bytes', '131072', '--'] + tests)
run('lifetime', mem + ['--check-api-memory-access', 'yes', '--track-stream-ordered-races', 'all', python, '-m', 'benchmarks.memcheck', '--stack-bytes', '131072', '--disable-mempool', '--'] + tests)
stem = w / 'prepared-full-native'
run('full-native', ['nsys', 'profile', '--trace=cuda', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true', '-o', stem, python, '-m', 'benchmarks.mesher', '--n', '1000', '--repeats', '3', '--profile', '--dot', stem.with_suffix('.dot'), '--output', stem.with_suffix('.json')])
run('full-native-export', ['nsys', 'export', '--type', 'sqlite', '--force-overwrite=true', '--output', stem.with_suffix('.sqlite'), stem.with_suffix('.nsys-rep')])
run('full-native-audit', [python, '-m', 'benchmarks.graph_capture', '--analyze', stem.with_suffix('.sqlite'), '--dot', stem.with_suffix('.dot'), '--scope', 'Three complete fresh torus1000 solves with current-input seed geometry prepared once per cell; one final synchronization, setup and readback excluded.', '--output', w / 'prepared-full-native-audit.json'])
run('mesher-loop', [python, '-m', 'benchmarks.mesher_loop', '--output', w / 'prepared-mesher-loop.json'])
run('independent', [python, w / 'check-prepared-results.py'])
