"""Offline gates after every complete clean trajectory pair has finished."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'.work/vtm'
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
out=w/'analytic-face-broad'
rows=json.loads((out/'commands.json').read_text())
assert len(rows)==11 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in rows[:10])
assert rows[-1]['name']=='armadillo-graded' and rows[-1]['variant']=='control' and not rows[-1]['quota_clean']
recheck=w/'analytic-face-broad-recheck2'
rechecks=json.loads((recheck/'commands.json').read_text())
assert len(rechecks)==2 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in rechecks)
summary={}
for name in ['graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    directory=recheck if name=='armadillo-graded' else out
    before,after=[directory/(name+'-'+variant) for variant in ['control','analytic']]
    a,b=[json.loads((p/'result.json').read_text()) for p in [before,after]]
    x,y=[np.load(p/'mesh.npz') for p in [before,after]]
    for d in [a,b]:quota(d)
    comparison=compare(x,y)
    assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,comparison)
    for k in ['input_points','input_faces']:np.testing.assert_array_equal(x[k],y[k])
    work={k:a[k]==b[k] for k in ['rounds','steps','moves','refinement_rejections','quality_rejections','counts']}
    assert all(work.values()),work
    params=lambda d:{k:v for k,v in d['parameters'].items() if k!='output'}
    assert params(a)==params(b)
    summary[name]=dict(before_ms=a['median_additional_replay_ms'],after_ms=b['median_additional_replay_ms'],comparison=comparison,work_equal=work,inputs_parameters_equal=True,independent=validate(after/'mesh.npz',after/'mesh.vtu'))
    (w/'analytic-face-broad-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,summary[name]['before_ms'],summary[name]['after_ms'],comparison,flush=True)
