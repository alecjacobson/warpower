"""Verify diagnostic output before interpreting invocation counters."""
from pathlib import Path
import hashlib,json
import numpy as np
w=Path('.work/vtm')
x=np.load(w/'float32-combined-pilot/float32-4096-candidate.npz');y=np.load(w/'float32-counter-4096.npz')
def canon(t):
 t=np.sort(t,axis=1);return t[np.lexsort(t.T[::-1])]
checks=dict(points_bitwise_equal=bool(np.array_equal(x['points'].view(np.uint32),y['points'].view(np.uint32))),roles_equal=bool(np.array_equal(x['roles'],y['roles'])),full_tets_equal=bool(np.array_equal(canon(x['full_tetrahedra']),canon(y['full_tetrahedra']))),material_tets_equal=bool(np.array_equal(canon(x['tetrahedra']),canon(y['tetrahedra']))))
assert all(checks.values()),checks
x,y=[json.loads(p.read_text()) for p in [w/'float32-combined-pilot/float32-4096-candidate.json',w/'float32-counter-run-4096.json']]
keys=['counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections'];work={k:x[k]==y[k] for k in keys};assert all(work.values())
counts=json.loads((w/'float32-counter-4096.json').read_text());totals={kind:{k:sum(row['counts'][k] for row in counts[kind]) for k in counts[kind][0]['counts']} for kind in ['face','seed']}
assert totals['face']['invalid_incomplete_threads']==0
report=dict(scope='Counter-only diagnostic, no timing claim.',comparison=checks,work_equal=work,totals=totals,source_sha256={name:hashlib.sha256((w/name).read_bytes()).hexdigest() for name in ['float32_filter_counters.py','build_float32_diagnostic.py','bounded_float32_filters.py']})
(w/'float32-counter-equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
