"""Independent exact global tests on every successful private flip output."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
root=Path('.work/vtm/local-flips-v2');dest=root/'independent.json';assert not dest.exists();meta=json.loads((root/'results.json').read_text());report=dict(scope=__doc__,cases={})
scratch=Path('/opt/warpower-generated-cache/validation-derived')
for name,row in meta['cases'].items():
 if not row['accepted']:continue
 path=root/(name+'.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==row['output_sha256']
 with np.load(path) as d,tempfile.TemporaryDirectory(dir=scratch) as td:
  td=Path(td);p=d['points'];t=d['tetrahedra'];np.savetxt(td/'p',p,fmt='%.17g',header=str(len(p)),comments='');np.savetxt(td/'t',t,fmt='%d',header=str(len(t)),comments='')
  proc=subprocess.run(['.work/vtm/check-delaunay',str(td/'p'),str(td/'t')],text=True,capture_output=True);record=dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,input_sha256=row['output_sha256']);report['cases'][name]=record;dest.write_text(json.dumps(report,indent=2)+'\n');assert proc.returncode==0,(name,record)
 print(name,flush=True)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
