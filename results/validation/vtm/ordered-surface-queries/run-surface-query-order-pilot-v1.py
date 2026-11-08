"""Size-gated ordering complete-solve pairs after contract checks; no production adoption."""
import gzip
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = w / 'surface-query-order-pilot-v1'
assert not out.exists()
assert not subprocess.check_output(['git','status','--porcelain'], text=True)
runtime = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
contract = json.loads((w/'surface-query-order-contracts-v3.json').read_text())
assert contract['exit_code'] == 0 and contract['passed']
for path, digest in contract['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
assert os.environ['WARP_CACHE_PATH'] == '/opt/warpower-generated-cache'
controls = json.loads((w/'quality-query-storage-final-v1/commands.json').read_text())
out.mkdir()
rows, archives = [], []
for name in ['torus16384-fixed32','thin-hand-seed7','armadillo-graded4096','armadillo-graded8192-recovery','armadillo-uniform8192-recovery']:
    control = next(c for c in controls if c['name'] == name and c['phase'] == 'after')
    for phase in ['before','after']:
        label = name+'-'+phase
        cmd = control['command'][:]
        if name.startswith('torus'):
            cmd = [cmd[0], str(w/'run-surface-query-order-full-v1.py'), *cmd[3:]]
            cmd[cmd.index('--output')+1] = str(out/(label+'.json'))
            cmd[cmd.index('--save-mesh')+1] = str(out/(label+'.npz'))
        else:
            cmd[1] = str(w/'run-surface-query-order-full-v1.py')
            cmd[cmd.index('--output')+1] = str(out/label)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.'+os.pathsep+str(w), WARPOWER_QUERY_ORDER_PHASE=phase, WARPOWER_QUERY_ORDER_KIND='benchmark' if name.startswith('torus') else 'gallery', WARPOWER_QUERY_ORDER_REPORT=str(out/(label+'-order.json')), WARPOWER_TOPOLOGY_DIAGNOSTICS=str(out/(label+'-recovery.json')))
        start = time.time()
        with (out/(label+'.log')).open('w') as stream:
            p = subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                p.wait()
                code = 124
        row = dict(name=name,phase=phase,source=runtime,command=cmd,exit_code=code,timeout_seconds=600,wall_seconds=time.time()-start)
        result = Path(cmd[cmd.index('--output')+1])
        if result.is_dir():
            result = result/'result.json'
        if result.exists():
            d = json.loads(result.read_text())
            quota_keys = ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta','initial_cpu_quota_delta']
            row.update(result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],median_ms=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms')),quota_clean=all(v['throttled_periods']==0 for k in quota_keys if k in d for v in d[k].values()))
        rows.append(row)
        (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(row,flush=True)
        assert code == 0 and row['valid'] == 1 and row['error'] == 0 and row['quota_clean'], row
        if 'armadillo' in name:
            duplicate = result.parent/'input.off'
            canonical = w/'scanned-armadillo-input/armadillo.off'
            assert hashlib.sha256(duplicate.read_bytes()).digest() == hashlib.sha256(canonical.read_bytes()).digest()
            duplicate.unlink()
            os.link(canonical,duplicate)
        if result.name == 'result.json':
            for export in ['mesh.vtu','boundary.off']:
                raw = result.parent/export
                data = raw.read_bytes()
                packed = raw.with_suffix(raw.suffix+'.gz')
                assert not packed.exists()
                packed.write_bytes(gzip.compress(data,mtime=0))
                assert gzip.decompress(packed.read_bytes()) == data
                archives.append(dict(original=str(raw),path=str(packed),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
                (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
                raw.unlink()
assert len(rows) == 10
