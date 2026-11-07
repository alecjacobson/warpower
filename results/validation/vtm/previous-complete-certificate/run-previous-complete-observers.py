import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';rows=[]
for n in [1000,4096]:
    stem=w/f'previous-complete-observer-{n}'
    env=dict(os.environ,WARPOWER_WHOLE_OBSERVER=str(stem.with_suffix('.json')))
    assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
    cmd=[str(root/'.venv/bin/python'),str(w/'observe_previous_complete.py'),'--n',str(n),'--repeats','1','--output',str(w/f'previous-complete-observer-run-{n}.json'),'--save-mesh',str(stem.with_suffix('.npz'))]
    assert not stem.with_suffix('.json').exists();start=time.time()
    with stem.with_suffix('.log').open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    rows.append(dict(n=n,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300))
    (w/'previous-complete-observer-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
    assert code==0
