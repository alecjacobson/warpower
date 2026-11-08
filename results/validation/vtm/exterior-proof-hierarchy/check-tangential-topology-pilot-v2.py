"""Independent successful complete private solves; all three pilot outputs succeed."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');helper=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(helper,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
folder=w/'tangential-topology-pilot-v2';dest=w/'tangential-topology-pilot-v2-independent.json';assert not dest.exists()
commands=json.loads((folder/'commands.json').read_text());assert len(commands)==3 and all(x['exit_code'] in [0,1] for x in commands)
report=dict(scope=__doc__,cases={})
for name in ['graded8192','uniform8192','graded4096']:
 d=json.loads((folder/name/'result.json').read_text());r=json.loads((folder/(name+'-recovery.json')).read_text())
 row=dict(result_sha256=hashlib.sha256((folder/name/'result.json').read_bytes()).hexdigest(),valid=d['valid'],error=d['error'],recovery=r)
 if d['valid']:
  assert d['error']==0
  row['independent']=validate(folder/name/'mesh.npz',folder/name/'mesh.vtu.gz')
  if name=='graded8192':assert r['chosen']==[6598] and r['accepted']==[1] and r['attempts']==[1] and d['rounds'][2]==4
  if name=='uniform8192':assert r['chosen']==[3406] and r['accepted']==[1] and r['attempts']==[22] and d['rounds'][2]==4
  if name=='graded4096':assert r['invoked']==r['accepted']==r['attempts']==[0]
 else:
  raise AssertionError((name,d['error']))
 report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,d['valid'],row.get('independent',{}).get('full_hull',{}).get('valid'),flush=True)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
