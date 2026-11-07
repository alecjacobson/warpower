"""Independent verification of all private publication/storage variants."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'.work/vtm'
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
summary={}
pairs=[]
for variant in ['local-dewall-vtm-v3','shared-wall-vtm-v1']:
    commands=json.loads((w/variant/'commands.json').read_text())
    assert len(commands)==4 and all(r['exit_code']==0 for r in commands)
    for n in [1000,4096]:
        pairs.append((variant+'-'+str(n),w/variant/f'torus-{n}-control',w/variant/f'torus-{n}-candidate',False))
commands=json.loads((w/'shared-wall-broad-v1/commands.json').read_text())
assert len(commands)==12 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
for name in ['graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']:
    pairs.append((name,w/'shared-wall-broad-v1'/(name+'-control'),w/'shared-wall-broad-v1'/(name+'-candidate'),True))
for name,a,b,gallery in pairs:
    ap,bp=(a/'result.json',b/'result.json') if gallery else (a.with_suffix('.json'),b.with_suffix('.json'))
    an,bn=(a/'mesh.npz',b/'mesh.npz') if gallery else (a.with_suffix('.npz'),b.with_suffix('.npz'))
    da,db=[json.loads(p.read_text()) for p in [ap,bp]]
    for d in [da,db]:quota(d);assert d['valid']==1 and d['error']==0
    params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['save_mesh','output','dot']}
    assert params(da)==params(db)
    x,y=np.load(an),np.load(bn);comparison=compare(x,y)
    assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,comparison)
    for k in ['input_points','input_faces']:np.testing.assert_array_equal(x[k],y[k])
    keys=['counts','rounds','refinement_rejections','quality_rejections']+(['steps','moves'] if gallery else ['optimizer_steps','quality_moves'])
    if 'reconstruction_routing' in da:keys.append('reconstruction_routing')
    work={k:da[k]==db[k] for k in keys};assert all(work.values()),(name,work)
    changed=np.argwhere(x['points'].view(np.uint32)!=y['points'].view(np.uint32))
    row=dict(before_ms=da.get('median_gpu_milliseconds',da.get('median_additional_replay_ms')),after_ms=db.get('median_gpu_milliseconds',db.get('median_additional_replay_ms')),comparison=comparison,work_equal=work,input_parameters_equal=True,changed_coordinates=[[int(i),int(j),float(x['points'][i,j]),float(y['points'][i,j])] for i,j in changed],before_pool_bytes=da.get('persistent_workspace_pool_bytes'),after_pool_bytes=db.get('persistent_workspace_pool_bytes'),independent=validate(bn,b/'mesh.vtu' if gallery else None))
    summary[name]=row
    (w/'shared-wall-private-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,row['before_ms'],row['after_ms'],comparison,flush=True)
