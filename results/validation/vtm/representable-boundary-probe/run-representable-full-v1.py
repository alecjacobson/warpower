"""Sequential fresh full pilots, including the coarse failing hand configuration."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'representable-full-v1';assert not out.exists();out.mkdir()
rows=[];archives=[]
for name,case,extra in [('torus','torus',[]),('dense-hand','thin-hand',['--subdivisions','20','--crease-subdivisions','20']),('coarse-hand','thin-hand',['--subdivisions','4','--crease-subdivisions','4']),('plate','counterbored-plate',[])]:
    for mode in ['baseline','candidate']:
        stem=out/(name+'-'+mode)
        cmd=[str(root/'.venv/bin/python'),str(w/'representable_full_pilot_v1.py'),'--pilot-mode',mode,'--case',case,'--n','1000','--repeats','1',*extra,'--output',str(stem)]
        start=time.time()
        with stem.with_suffix('.log').open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=124
        row=dict(case=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=240)
        if (stem/'result.json').exists():
            d=json.loads((stem/'result.json').read_text());row.update({k:d.get(k) for k in ['valid','error','counts','quality','median_additional_replay_ms']})
        if (stem/'pilot.json').exists():row['corrections']=json.loads((stem/'pilot.json').read_text())['cores']
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
        for p in [stem/'mesh.vtu',stem/'boundary.off']:
            if p.exists():
                raw=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');assert not dest.exists();dest.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(dest.read_bytes())==raw
                archives.append(dict(original=str(p),archive=str(dest),raw_sha256=hashlib.sha256(raw).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
                (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()
        print({k:v for k,v in row.items() if k not in ['quality','command']},flush=True)
        assert code in [0,1] and (stem/'result.json').exists() and (stem/'pilot.json').exists(),row
