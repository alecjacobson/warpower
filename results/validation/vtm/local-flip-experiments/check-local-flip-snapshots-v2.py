"""Offline independent checks of retained previous and reconstructed full hulls."""
import hashlib,json,subprocess,tempfile,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);dest=root/'snapshot-validation-graded.json';assert not dest.exists()
scratch=Path('/opt/warpower-generated-cache/validation-derived');scratch.mkdir(parents=True,exist_ok=True)
report=dict(scope=__doc__,cases={})
for p in sorted(root.glob('graded-armadillo/previous-*-snapshot-*.npz')):
 row=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 with np.load(p) as d,tempfile.TemporaryDirectory(dir=scratch) as td:
  td=Path(td)
  for phase,pk,tk in [('previous','previous_points','previous_tets'),('reconstructed','points','tetrahedra')]:
   points=d[pk];tets=d[tk];assert len(tets)>0
   np.savetxt(td/'points.txt',points,fmt='%.17g',header=str(len(points)),comments='')
   np.savetxt(td/'tets.txt',tets,fmt='%d',header=str(len(tets)),comments='')
   proc=subprocess.run(['.work/vtm/check-delaunay',str(td/'points.txt'),str(td/'tets.txt')],text=True,capture_output=True)
   row[phase]=dict(exit_code=proc.returncode,stderr=proc.stderr,stdout=proc.stdout)
   report['cases'][str(p)]=row;dest.write_text(json.dumps(report,indent=2)+'\n')
   assert proc.returncode==0,(p,phase,row[phase])
 print(p,flush=True)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
