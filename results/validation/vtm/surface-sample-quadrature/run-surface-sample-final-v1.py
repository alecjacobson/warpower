"""Production quadrature complete-solve matrix and independent standalone controls."""
import gzip
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd()
w=root/'.work/vtm'
out=w/'surface-sample-final-v1'
baseline=w/'surface-sample-baseline'
assert not out.exists()
tests=json.loads((w/'surface-sample-production-tests-v2.json').read_text())
assert len(tests)==2 and all(r['exit_code']==0 and r['passed'] for r in tests)
source=json.loads((w/'surface-sample-production-source-v2.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['runtime']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()==source['baseline']
for cwd in [root,baseline]:assert not subprocess.check_output(['git','status','--porcelain'],cwd=cwd)
assert os.environ['WARP_CACHE_PATH']=='/opt/warpower-generated-cache'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'+os.pathsep+str(w))
out.mkdir()
rows=[]
archives=[]
lookup={}
for name in ['surface-query-order-final-v1']:
    for r in json.loads((w/name/'commands.json').read_text()):
        if r['phase']=='after':lookup[r['name']]=r['command']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(name,phase,cmd,cwd):
    label=name+'-'+phase
    runenv=dict(env,WARPOWER_QUERY_ORDER_PHASE=phase,WARPOWER_QUERY_ORDER_REPORT=str(out/(label+'-order.json')),WARPOWER_TOPOLOGY_DIAGNOSTICS=str(out/(label+'-observer.json')))
    start=time.time()
    with (out/(label+'.log')).open('w') as stream:
        p=subprocess.Popen(cmd,cwd=cwd,env=runenv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,phase=phase,command=cmd,cwd=str(cwd),source=source['baseline' if phase=='before' else 'runtime'],exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
    path=Path(cmd[cmd.index('--output')+1])
    if path.is_dir():path=path/'result.json'
    if path.exists():
        d=json.loads(path.read_text())
        row.update(result=str(path),result_sha256=sha(path),valid=d.get('valid'),error=d.get('error'),median_ms=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms',d.get('median_ms'))),quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta','initial_cpu_quota_delta'] if k in d for v in d[k].values()))
        if 'full' in d:row.update(median_ms=d['full']['median_ms'],quota_clean=all(v['throttled_periods']==0 for v in d['full']['cpu'].values()))
        if 'median_seconds' in d:row.update(median_ms=1000*d['median_seconds'],quota_clean=not d['cpu_throttled'])
    rows.append(row)
    (out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(row,flush=True)
    assert code==0 and row.get('quota_clean') and row.get('valid') in [None,1] and row.get('error') in [None,0],row
    if 'armadillo' in name:
        duplicate=path.parent/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
        assert sha(duplicate)==sha(canonical)
        duplicate.unlink();os.link(canonical,duplicate)
    if path.name=='result.json':
        for export in ['mesh.vtu','boundary.off']:
            raw=path.parent/export
            data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+'.gz')
            assert not packed.exists()
            packed.write_bytes(gzip.compress(data,mtime=0))
            assert gzip.decompress(packed.read_bytes())==data
            archives.append(dict(original=str(raw),path=str(packed),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=sha(packed)))
            (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
            raw.unlink()

for name in ['torus16384-fixed32','torus1000','torus8192-fixed32','torus8192-progressive']:
    template=lookup[name if name in lookup else 'torus16384-fixed32']
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=template[:]
        if name.startswith('torus8192'):
            cmd[cmd.index('--n')+1]='8192'
            cmd[cmd.index('--initial')+1]='8192' if name.endswith('fixed32') else '256'
            cmd[cmd.index('--growth-iterations')+1]='0' if name.endswith('fixed32') else '5'
            cmd[cmd.index('--polish-iterations')+1]='32' if name.endswith('fixed32') else '20'
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'))
        cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
        run(name,phase,cmd,cwd)
for name in ['thin-hand-seed7','cavity','disconnected','plate','armadillo-graded4096','armadillo-graded8192-recovery','armadillo-uniform8192-recovery']:
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=lookup[name][:]
        cmd[1]=str(w/'run-tangential-production-gallery-v1.py')
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase))
        run(name,phase,cmd,cwd)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=lookup[name][:]
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'))
        run(name,phase,cmd,cwd)
assert len(rows)==32
