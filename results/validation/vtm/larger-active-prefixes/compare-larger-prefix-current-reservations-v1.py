"""Compare validated post-change solves across reservation sizes; no new timing."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');out=w/'larger-prefix-current-reservation-comparison-v1.json';assert not out.exists()
validation=w/'larger-prefix-final-v1-independent.json';v=json.loads(validation.read_text());assert v['complete']
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
report=dict(scope=__doc__,source='670bd3b1a801e5397f55837d3caf685c20286656',independent_validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),cases={})
for small,large in [('torus4096','reserved16384-target4096'),('torus4096','reserved8192-target4096'),('torus8192-progressive','reserved16384-target8192')]:
 paths=[w/'larger-prefix-final-v1'/(n+'-after.json') for n in [small,large]];a,b=[json.loads(p.read_text()) for p in paths]
 for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
 parameters=lambda d:{k:z for k,z in d['parameters'].items() if k not in ['output','save_mesh','dot','capacity']}
 assert parameters(a)==parameters(b)
 keys=['counts','rounds','topology','optimizer_steps','quality_moves','quality_rejections','refinement_rejections'];work={k:a[k]==b[k] for k in keys};assert all(work.values())
 with np.load(paths[0].with_suffix('.npz')) as x,np.load(paths[1].with_suffix('.npz')) as y:
  comparison=compare(x,y);assert all(comparison[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
  assert all(np.array_equal(x[k],y[k]) for k in ['input_points','input_faces'])
 am,bm=a['median_gpu_milliseconds'],b['median_gpu_milliseconds']
 report['cases'][large]=dict(dedicated=small,reserved=large,dedicated_ms=am,reserved_ms=bm,reserved_over_dedicated=bm/am,comparison=comparison,work_equal=work,parameters_equal_except_reservation=True,pool_bytes=[a['persistent_workspace_pool_bytes'],b['persistent_workspace_pool_bytes']],result_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],mesh_sha256=[hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest() for p in paths])
 print(large,am,bm,bm/am,flush=True)
report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')
