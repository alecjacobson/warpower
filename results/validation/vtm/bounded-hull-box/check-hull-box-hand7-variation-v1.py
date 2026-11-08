"""Audit the strict final-check failure and test saved connectivity across point frames."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from bounded_hull_box_variation_v1 import audit_pair
w=Path('.work/vtm');source=w/'bounded-hull-box-final-independent-v1.json';report=json.loads(source.read_text());name='hand-seed7';out=w/'bounded-hull-box-hand7-variation-v1.json';assert not out.exists()
r=audit_pair(w/'bounded-hull-box-final',name,report['cases'][name],report['equivalence'][name]['work_equal'])
r['failed_strict_report_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();r['cross_frame_checks']={}
with np.load(w/'bounded-hull-box-final/hand-seed7-before/mesh.npz') as a,np.load(w/'bounded-hull-box-final/hand-seed7-after/mesh.npz') as b:
 with tempfile.TemporaryDirectory(prefix='cross-frame-hull-',dir='/opt/warpower-generated-cache/validation-derived') as tmp:
  pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
  for label,points,tets in [('before_points_after_cells',a['points'],b['full_tetrahedra']),('after_points_before_cells',b['points'],a['full_tetrahedra'])]:
   np.savetxt(pp,points,fmt='%.17g',header=str(len(points)),comments='');np.savetxt(tp,tets,fmt='%d',header=str(len(tets)),comments='')
   p=subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,timeout=60)
   r['cross_frame_checks'][label]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
r['scope']='Both own-frame full hulls independently validate; this audit retains changed exterior cells, exact volumes, per-material-cell quality and cross-frame checks. No claim of bitwise reproducibility or a traced causal kernel.'
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
