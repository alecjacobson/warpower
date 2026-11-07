import json,hashlib
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
helper=(w/'check-current-selection-final.py').read_text().split("out=w/'current-selection-final';commands=")[0]
exec(compile(helper,'current-selection-validation-helper','exec'))
out=w/'recovery-classification-pilot-v1';cmd=json.loads((out/'commands.json').read_text());assert len(cmd)==2 and all(r['exit_code']==0 and r['quota_clean'] for r in cmd)
a,b=[json.loads((out/phase/'result.json').read_text()) for phase in ['control','candidate']]
for d in [a,b]:quota(d);assert d['error']==0 and d['valid']==1
params=lambda d:{k:v for k,v in d['parameters'].items() if k!='output'};assert params(a)==params(b)
x,y=[np.load(out/phase/'mesh.npz') for phase in ['control','candidate']];comparison=compare(x,y)
assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']);assert comparison['points_bitwise_equal']
work={k:a[k]==b[k] for k in ['counts','rounds','steps','moves','refinement_rejections','quality_rejections']};assert all(work.values())
for k in ['input_points','input_faces']:np.testing.assert_array_equal(x[k],y[k])
result=dict(before_ms=a['median_additional_replay_ms'],after_ms=b['median_additional_replay_ms'],comparison=comparison,work_equal=work,parameters_equal=True,all_first_and_timed_quotas_clean=True,independent=validate(out/'candidate/mesh.npz',out/'candidate/mesh.vtu.gz'))
(w/'recovery-classification-pilot-v1-independent.json').write_text(json.dumps(result,indent=2)+'\n');print(result['before_ms'],result['after_ms'],comparison,flush=True)
