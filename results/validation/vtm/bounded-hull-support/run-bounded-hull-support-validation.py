"""Serial adoption checks; no overlapping GPU or clean-timing CPU jobs."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
assert env.get('PYTHONDONTWRITEBYTECODE')=='1'
rows=[];archives=[]
start=time.time()
full_pid=1055587  # Replaced with the observed full-suite PID before launch.
while True:
    try:os.kill(full_pid,0)
    except ProcessLookupError:break
    assert time.time()-start<900,'Full-suite process has not finished'
    time.sleep(3)
assert '831 passed' in (w/'bounded-hull-support-full-suite.log').read_text()

def run(name,script,timeout):
    path=w/f'bounded-hull-support-{name}.log'
    assert not path.exists()
    command=[str(python),str(w/script)];start=time.time()
    with path.open('w') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    row=dict(name=name,command=command,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout)
    rows.append(row);(w/'bounded-hull-support-validation-workflow.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(row,flush=True);assert code==0,row

def archive(path):
    data=path.read_bytes();dest=path.with_suffix(path.suffix+'.gz')
    assert not dest.exists();dest.write_bytes(gzip.compress(data,mtime=0))
    assert gzip.decompress(dest.read_bytes())==data
    row=dict(original=str(path.relative_to(root)),path=str(dest.relative_to(root)),
             raw_bytes=len(data),bytes=dest.stat().st_size,
             raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    archives.append(row)
    (w/'bounded-hull-support-lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n')
    path.unlink()  # Verified exact persistent compressed copy exists.

run('final','run-bounded-hull-support-final.py',1200)
run('final-independent','check-bounded-hull-support-final.py',600)
for path in sorted((w/'bounded-hull-support-final').rglob('*')):
    if path.suffix=='.vtu' or path.name=='boundary.off':archive(path)
run('memory','run-bounded-hull-support-memory.py',2400)
run('native','run-bounded-hull-support-native.py',900)
for path in sorted(w.glob('bounded-hull-support-*-native.nsys-rep')):archive(path)
print('All serial adoption jobs completed; inspect records before publication.',flush=True)
