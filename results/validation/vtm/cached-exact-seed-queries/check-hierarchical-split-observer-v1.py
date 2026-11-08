"""Verify diagnostic counters preserve the already independently checked clean mesh."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
s=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(s,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
ref=w/'hierarchical-split-pilot-v1/torus16384-fixed32-after';obs=w/'hierarchical-split-observer-v1'
a,b=[json.loads(p.with_suffix('.json').read_text()) for p in [ref,obs]]
keys=['valid','error','status','counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections','last_errors','last_bad_extraction','topology']
work={k:a[k]==b[k] for k in keys};assert all(work.values()) and b['valid']==1 and b['error']==0
with np.load(ref.with_suffix('.npz')) as x,np.load(obs.with_suffix('.npz')) as y:
 equality=compare(x,y);assert all(equality[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
counts=json.loads((w/'hierarchical-split-observer-v1-counts.json').read_text());assert len(counts['rows'])==1
calls,instances,cells,lanes=counts['rows'][0]['statistics'];assert calls==3*227 and cells==lanes==0
r=dict(scope=__doc__,work_equal=work,geometry=equality,executions=3,calls=calls,cell_instances=instances,fallback_cells=cells,fallback_lanes=lanes,interpretation='Three full graph executions: first measured execution, second warm execution, then one additional measured execution. Observer timings are diagnostic, not performance evidence. Zero fallback is specific to this trajectory; exact ties remain supported.',sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ref.with_suffix('.npz'),obs.with_suffix('.npz'),w/'hierarchical-split-observer-v1-counts.json']})
(w/'hierarchical-split-observer-v1-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
