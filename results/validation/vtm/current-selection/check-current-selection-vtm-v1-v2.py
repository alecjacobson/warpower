"""Independent paired complete-solve checks for the private current-evaluation cache."""
import json,hashlib
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
helper=(w/'check-shared-wall-final.py').read_text().split("out=w/'shared-wall-final';commands=")[0]
exec(compile(helper,'shared-wall-independent-helper','exec'))
out=w/'current-selection-vtm-v1';commands=json.loads((out/'commands.json').read_text());assert len(commands)==4 and all(r['exit_code']==0 for r in commands)
rows={}
for n in [1000,4096]:
 a,b=[json.loads((out/f'torus-{n}-{phase}.json').read_text()) for phase in ['control','candidate']]
 for d in [a,b]:quota(d);assert d['error']==0 and d['valid']==1
 ap,bp=[out/f'torus-{n}-{phase}.npz' for phase in ['control','candidate']]
 x,y=np.load(ap),np.load(bp);comparison=compare(x,y);assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 for key in ['input_points','input_faces']:np.testing.assert_array_equal(x[key],y[key])
 parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['save_mesh','output','dot']};assert parameters(a)==parameters(b)
 keys=['counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections']
 if 'reconstruction_routing' in a or 'reconstruction_routing' in b:keys.append('reconstruction_routing')
 work={k:a[k]==b[k] for k in keys};assert all(work.values())
 rows[str(n)]=dict(before_ms=a['median_gpu_milliseconds'],after_ms=b['median_gpu_milliseconds'],before_pool_bytes=a['persistent_workspace_pool_bytes'],after_pool_bytes=b['persistent_workspace_pool_bytes'],comparison=comparison,work_equal=work,parameters_equal=True,all_timing_windows_quota_clean=True,before_mesh_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),after_mesh_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),independent=validate(bp))
 (w/'current-selection-vtm-v1-independent-v2.json').write_text(json.dumps(rows,indent=2)+'\n');print(n,rows[str(n)]['before_ms'],rows[str(n)]['after_ms'],comparison,flush=True)
