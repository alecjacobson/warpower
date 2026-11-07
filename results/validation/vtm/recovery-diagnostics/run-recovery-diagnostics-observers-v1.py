import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'recovery-diagnostics-observers-v1';assert not out.exists();out.mkdir();rows=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache')
for name in ['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    stem=w/'current-selection-final'/(name+'-after')
    baseline=stem/'result.json' if stem.is_dir() else stem.with_suffix('.json');mesh=stem/'mesh.npz' if stem.is_dir() else stem.with_suffix('.npz')
    cmd=[str(root/'.venv/bin/python'),str(w/'recovery_diagnostics_observer_v1.py'),'--baseline',str(baseline),'--mesh',str(mesh),'--output',str(out/(name+'.json'))]
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start);rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
