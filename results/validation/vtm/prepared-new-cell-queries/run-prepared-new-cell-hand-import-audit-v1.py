"""Two dependency audit runs; no clean timing comparison is inferred."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-new-cell-hand-import-audit-v1';assert not out.exists();out.mkdir();rows=[]
for phase,cwd in [('before',w/'prepared-new-cell-baseline'),('after',root)]:
 cmd=[str(root/'.venv/bin/python'),str(w/'audit-prepared-new-cell-hand-imports-v1.py'),'--case','fitted-hand','--repeats','1','--output',str(out/(phase+'.json'))];start=time.time()
 with (out/(phase+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONPATH='.'),stdout=f,stderr=subprocess.STDOUT,timeout=120)
 rows.append(dict(phase=phase,command=cmd,cwd=str(cwd),exit_code=p.returncode,wall_seconds=time.time()-start));(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0
r=[json.loads((out/(phase+'.imports.json')).read_text()) for phase in ['before','after']];assert r[0]['modules']==r[1]['modules'];assert not any(r[0]['changed_modules_loaded'].values()) and not any(r[1]['changed_modules_loaded'].values());report=dict(scope='Executed project dependency sets are byte-identical; neither modified runtime module is loaded. This proves code-path independence, not zero timing variability.',project_modules=len(r[0]['modules']),modules_identical=True,changed_modules_not_loaded=True,complete=True);(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
