"""Clean sequential full solves; no observers or simultaneous heavy CPU validation."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'boundary-global-reduce-pilot-v1';assert not out.exists();out.mkdir()
python=root/'.venv/bin/python';commands=[];archives=[]
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source=='971e6f2e5aa641dce95c1b4ad2124e7454f11a6a'
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
for name,case,n,extra in [('torus4096','torus',4096,[]),('hand7','thin-hand',1000,['--seed','7'])]:
    for phase in ['before','after']:
        dest=out/(name+'-'+phase)
        cmd=[str(python),str(w/'run-boundary-global-reduce-gallery-v1.py'),*(['--candidate'] if phase=='after' else []),'--case',case,'--n',str(n),'--initial','256','--quality-rounds','4','--repeats','3',*extra,'--output',str(dest)]
        start=time.time()
        with (out/(name+'-'+phase+'.log')).open('w') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=300)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
        row=dict(name=name,phase=phase,source=source,command=cmd,exit_code=code,timeout_seconds=300,wall_seconds=time.time()-start)
        path=dest/'result.json'
        if path.exists():
            d=json.loads(path.read_text());row.update(valid=d['valid'],error=d['error'],median_ms=d['median_additional_replay_ms'],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            row['quota_clean']=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'] if k in d for v in d[k].values())
        commands.append(row);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(row,flush=True)
        assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
        for filename in ['mesh.vtu','boundary.off']:
            p=dest/filename;raw=p.read_bytes();packed=p.with_suffix(p.suffix+'.gz');assert not packed.exists()
            packed.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(packed.read_bytes())==raw
            archives.append(dict(path=str(packed),sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))
            (out/'lossless-archives.json').write_text(json.dumps(archives,indent=2)+'\n');p.unlink()
