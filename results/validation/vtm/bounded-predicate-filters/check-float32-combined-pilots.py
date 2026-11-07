"""Offline original-coordinate checks after all clean pilot jobs finish."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'.work/vtm'
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
out=w/'float32-combined-pilot';rows=json.loads((out/'commands.json').read_text())
assert len(rows)==4 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in rows)
summary={}
for mode in ['float32']:
    for n in [1000,4096]:
        a,b=[json.loads((out/f'{mode}-{n}-{phase}.json').read_text()) for phase in ['control','candidate']]
        x,y=[np.load(out/f'{mode}-{n}-{phase}.npz') for phase in ['control','candidate']]
        for d in [a,b]:quota(d)
        comparison=compare(x,y)
        assert all(comparison[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(mode,n,comparison)
        for k in ['input_points','input_faces']:np.testing.assert_array_equal(x[k],y[k])
        params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh']}
        assert params(a)==params(b)
        keys=['counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections']
        if 'reconstruction_routing' in a:keys.append('reconstruction_routing')
        work={k:a[k]==b[k] for k in keys};assert all(work.values()),work
        summary[f'{mode}-{n}']=dict(before_ms=a['median_gpu_milliseconds'],after_ms=b['median_gpu_milliseconds'],comparison=comparison,work_equal=work,inputs_parameters_equal=True,before_pool_bytes=a['persistent_workspace_pool_bytes'],after_pool_bytes=b['persistent_workspace_pool_bytes'],independent=validate(out/f'{mode}-{n}-candidate.npz'))
        (w/'float32-combined-pilot-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(mode,n,summary[f'{mode}-{n}']['before_ms'],summary[f'{mode}-{n}']['after_ms'],comparison,flush=True)
