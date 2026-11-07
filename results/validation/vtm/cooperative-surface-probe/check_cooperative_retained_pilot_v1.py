"""Offline independent mesh checks after all paired replay timings."""
import json
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
helper=(w/'check-current-selection-final.py').read_text().split("out=w/'current-selection-final';commands=")[0]
exec(compile(helper,'current-selection-validation-helper','exec'))
out=w/'cooperative-retained-pilot-v1';commands=json.loads((out/'commands.json').read_text())
assert len(commands)==4 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
summary={}
for name in ['torus1000','torus4096']:
    a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['control','candidate']]
    for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
    params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh']}
    assert params(a)==params(b)
    x,y=[np.load(out/(name+'-'+phase+'.npz')) for phase in ['control','candidate']]
    comparison=compare(x,y)
    assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal','points_bitwise_equal'])
    for key in ['input_points','input_faces']:np.testing.assert_array_equal(x[key],y[key])
    keys=['counts','rounds','refinement_rejections','quality_rejections','optimizer_steps','quality_moves','reconstruction_routing']
    work={k:a[k]==b[k] for k in keys if k in a or k in b};assert all(work.values())
    result=dict(before_ms=a['median_gpu_milliseconds'],after_ms=b['median_gpu_milliseconds'],comparison=comparison,
                work_equal=work,parameters_equal=True,all_first_and_timed_quotas_clean=True,
                independent=validate(out/(name+'-candidate.npz')))
    summary[name]=result
    (w/'cooperative-retained-pilot-v1-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,result['before_ms'],result['after_ms'],comparison,flush=True)
