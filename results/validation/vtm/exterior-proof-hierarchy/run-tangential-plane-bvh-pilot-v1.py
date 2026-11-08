"""Full fresh graded/uniform8K experiments and a previously valid control."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'tangential-plane-bvh-pilot-v1';assert not out.exists();out.mkdir()
env=dict(os.environ,PYTHONPATH='.:.work/vtm',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
rows=[]
for name,case,n in [('graded8192','graded-surface',8192),('uniform8192','surface',8192),('graded4096','graded-surface',4096)]:
    cmd=[str(root/'.venv/bin/python'),str(w/'run-tangential-plane-bvh-gallery-v1.py'),'--case',case,'--input',str(w/'scanned-armadillo-input/armadillo.off'),'--n',str(n),'--initial','256','--subdivisions','1','--crease-subdivisions','1','--feature-angle','120','--corner-angle','45','--repeats','0','--output',str(out/name)]
    runenv=dict(env,WARPOWER_TOPOLOGY_DIAGNOSTICS=str(out/(name+'-recovery.json')));start=time.time()
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=runenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600)
    result=out/name/'result.json'
    if result.exists():
        d=json.loads(result.read_text());row.update(valid=d['valid'],error=d['error'],result=str(result))
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert code in [0,1] and result.exists(),row
    duplicate=out/name/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
    assert hashlib.sha256(duplicate.read_bytes()).digest()==hashlib.sha256(canonical.read_bytes()).digest();duplicate.unlink();os.link(canonical,duplicate)
    for name_ in ['mesh.vtu','boundary.off']:
        raw=out/name/name_
        if raw.exists():
            data=raw.read_bytes();packed=raw.with_suffix(raw.suffix+'.gz');packed.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(packed.read_bytes())==data;raw.unlink()
assert len(rows)==3
