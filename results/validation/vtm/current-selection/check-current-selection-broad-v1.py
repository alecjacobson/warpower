"""Independent complete meshes for the current-selection cache's broader pilots."""
import json,hashlib
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
helper=(w/'check-shared-wall-final.py').read_text().split("out=w/'shared-wall-final';commands=")[0]
exec(compile(helper,'shared-wall-independent-helper','exec'))
out=w/'current-selection-broad-v1';commands=json.loads((out/'commands.json').read_text());assert len(commands)==12 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
rows={}
for name in ['graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
 before,after=[out/(name+'-'+phase) for phase in ['control','candidate']]
 a,b=[json.loads((p/'result.json').read_text()) for p in [before,after]]
 for d in [a,b]:quota(d);assert d['error']==0 and d['valid']==1
 ap,bp=before/'mesh.npz',after/'mesh.npz';x,y=np.load(ap),np.load(bp);comparison=compare(x,y)
 assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,comparison)
 for key in ['input_points','input_faces']:np.testing.assert_array_equal(x[key],y[key])
 parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output']};assert parameters(a)==parameters(b)
 work={k:a[k]==b[k] for k in ['counts','rounds','steps','moves','refinement_rejections','quality_rejections']};assert all(work.values()),(name,work)
 changed=np.argwhere(x['points'].view(np.uint32)!=y['points'].view(np.uint32))
 rows[name]=dict(before_ms=a['median_additional_replay_ms'],after_ms=b['median_additional_replay_ms'],comparison=comparison,changed_coordinate_values=[[int(i),int(j),float(x['points'][i,j]),float(y['points'][i,j])] for i,j in changed],work_equal=work,parameters_equal=True,all_first_and_timed_quotas_clean=True,before_mesh_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),after_mesh_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),independent=validate(bp,after/'mesh.vtu'))
 (w/'current-selection-broad-v1-independent.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,rows[name]['before_ms'],rows[name]['after_ms'],comparison,flush=True)
