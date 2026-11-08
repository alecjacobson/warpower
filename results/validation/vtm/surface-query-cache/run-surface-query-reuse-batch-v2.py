"""Sequential complete observations; instrumented timings are not performance claims."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'surface-query-reuse-observer-v2';assert not out.exists();out.mkdir()
python=root/'.venv/bin/python';commands=[];archives=[]
cases=[('torus1000','torus',1000,[]),('torus4096','torus',4096,[]),('hand7','thin-hand',1000,['--seed','7']),('plate','counterbored-plate',1000,[]),('graded-armadillo','graded-surface',4096,['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120'])]
for name,case,n,extra in cases:
    dest=out/name;cmd=[str(python),str(w/'run-surface-query-reuse-observer-v2.py'),'--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','1',*extra,'--output',str(dest)]
    start=time.time()
    with (out/(name+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,timeout_seconds=300,wall_seconds=time.time()-start)
    commands.append(row);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(row,flush=True)
    assert code==0,row
    for filename in ['mesh.vtu','boundary.off']:
        p=dest/filename;raw=p.read_bytes();packed=p.with_suffix(p.suffix+'.gz');assert not packed.exists()
        packed.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(packed.read_bytes())==raw
        archives.append(dict(path=str(packed),sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
        (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()
    if name=='graded-armadillo':
        p=dest/'input.off';canonical=w/'scanned-armadillo-input/armadillo.off'
        assert p.read_bytes()==canonical.read_bytes();p.unlink();os.link(canonical,p)
    result=json.loads((dest/'result.json').read_text());assert result['valid']==1 and result['error']==0
    observed=json.loads((dest/'query-observer.json').read_text())
    for instance in observed['instances']:
        s=instance['statistics'];assert all(s[i]==0 for i in [7,8,14,15,16,17]),instance
