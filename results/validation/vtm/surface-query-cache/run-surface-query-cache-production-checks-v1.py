"""Sequential production oracle and complete suite after focused integration passes."""
import json,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';out=w/'surface-query-cache-production-checks-v1.json';assert not out.exists()
assert 'passed' in (w/'surface-query-cache-production-focused-v1.log').read_text() and 'failed' not in (w/'surface-query-cache-production-focused-v1.log').read_text()
rows=[]
for name,command in [('retained',[str(python),str(w/'check-surface-query-cache-retained-production-v1.py')]),('full',[str(python),str(w/'check-surface-query-cache-full-production-v1.py')]),('suite',[str(python),'-m','pytest','-q'])]:
 log=w/('surface-query-cache-full-suite-v1.log' if name=='suite' else 'surface-query-cache-'+name+'-production-check-v1.log');assert not log.exists();start=time.time()
 with log.open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 rows.append(dict(name=name,command=command,exit_code=r.returncode,wall_seconds=time.time()-start));out.write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True);assert r.returncode==0
