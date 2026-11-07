"""Broader paired complete solves after choosing a private query variant."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = w / 'prepared-query-broad'
mode = os.environ['WARPOWER_PREPARED_BROAD_MODE']
assert mode in ['point', 'analytic', 'strict', 'strict-stop']
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert source.startswith('6469ab6') and not subprocess.check_output(['git', 'status', '--short'], text=True)
assert not out.exists()
out.mkdir()
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.:.work/vtm')
assert env.get('WARP_CACHE_PATH') == '/opt/warpower-generated-cache'
old = json.loads((w / 'analytic-final/commands.json').read_text())
rows = []
for name in ['graded-torus', 'thin-hand', 'cavity', 'disconnected', 'armadillo-uniform', 'armadillo-graded']:
    ref = next(r for r in old if r['name'] == name and r['phase'] == 'after')
    for phase in ['control', 'candidate']:
        stem = out / (name + '-' + phase)
        cmd = ref['command'][:]
        cmd[cmd.index('--output') + 1] = str(stem)
        if phase == 'candidate':
            assert cmd[1] == '-m'
            cmd = [cmd[0], str(w / 'run-prepared-broad-entry.py'), *cmd[2:]]
        start = time.time()
        with stem.with_suffix('.log').open('w') as log:
            p = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                code = 124
        row = dict(name=name, phase=phase, mode=mode, command=cmd, source=source, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=300)
        path = stem / 'result.json'
        if path.exists():
            d = json.loads(path.read_text())
            row.update(result=str(path), valid=d['valid'], error=d['error'], median_ms=d['median_additional_replay_ms'])
            row['quota_clean'] = all(v['throttled_periods'] == 0 for k in ['cpu_quota_delta', 'initial_replays_cpu_quota_delta', 'additional_replays_cpu_quota_delta', 'first_replay_cpu_quota_delta'] if k in d for v in d[k].values())
        rows.append(row)
        (out / 'commands.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(row, flush=True)
        assert code == 0 and row.get('valid') == 1 and row.get('error') == 0 and row.get('quota_clean'), row
        # The scanned input is a byte-identical exported copy of the retained
        # persistent source. Share that storage only after verifying both hashes.
        duplicate = stem / 'input.off'
        canonical = w / 'scanned-armadillo-input/armadillo.off'
        if name.startswith('armadillo') and duplicate.exists():
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            assert sha(duplicate) == sha(canonical)
            row['input_deduplicated_against'] = str(canonical)
            row['input_sha256'] = sha(canonical)
            duplicate.unlink()
            os.link(canonical, duplicate)
            (out / 'commands.json').write_text(json.dumps(rows, indent=2) + '\n')
