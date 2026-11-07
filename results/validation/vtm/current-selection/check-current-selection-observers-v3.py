"""Compare observer outputs with the independently checked current runtime."""
import json,hashlib
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
helper=(w/'check-shared-wall-final.py').read_text().split("out=w/'shared-wall-final';commands=")[0]
exec(compile(helper,'shared-wall-final-helper','exec'))
baseproof=json.loads((w/'shared-wall-final-independent.json').read_text());out=w/'current-selection-observers-v3';rows={}
commands=json.loads((out/'commands.json').read_text());assert len(commands)==5 and all(r['exit_code']==0 for r in commands)
for command in commands:
 name=command['name'];gallery=not name.startswith('torus');before=w/'shared-wall-final'/(name+'-after');after=out/name
 ap,an=(before/'result.json',before/'mesh.npz') if gallery else (before.with_suffix('.json'),before.with_suffix('.npz'))
 bp,bn=(after/'result.json',after/'mesh.npz') if gallery else (after.with_suffix('.json'),after.with_suffix('.npz'))
 a,b=[json.loads(p.read_text()) for p in [ap,bp]];x,y=np.load(an),np.load(bn);comparison=compare(x,y)
 assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 for key in ['input_points','input_faces']:np.testing.assert_array_equal(x[key],y[key])
 keys=['counts','rounds','refinement_rejections','quality_rejections']+(['steps','moves'] if gallery else ['optimizer_steps','quality_moves'])
 if 'reconstruction_routing' in a:keys.append('reconstruction_routing')
 work={k:a[k]==b[k] for k in keys};assert all(work.values())
 parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['save_mesh','output','dot','repeats']};assert parameters(a)==parameters(b)
 repetitions=1+len(b['additional_replay_gpu_ms']) if gallery else 2+len(b['gpu_milliseconds'])
 observed=json.loads((out/(name+'-observer.json')).read_text())['instances'];assert len(observed)==1;observed=observed[0]
 fields=['calls','select_calls','check_calls','identical_input_calls','check_to_select_duplicates'];assert all(observed[k]%repetitions==0 for k in fields)
 per={k:observed[k]//repetitions for k in fields}
 independent=dict(scope='Positions are bitwise identical and connectivity/boundary match the independently certified baseline; its geometric proof applies to the same mesh.',baseline_report='.work/vtm/shared-wall-final-independent.json',baseline_mesh_sha256=hashlib.sha256(an.read_bytes()).hexdigest()) if comparison['points_bitwise_equal'] else validate(bn,after/'mesh.vtu.gz' if gallery else None)
 rows[name]=dict(comparison=comparison,work_equal=work,parameters_match_except_timing_output=True,repetitions=repetitions,per_replay=per,all_repeated_outputs_bitwise_equal=observed['all_repeated_outputs_bitwise_equal'],independent=independent,mesh_sha256=hashlib.sha256(bn.read_bytes()).hexdigest())
 (w/'current-selection-observers-independent-v3.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,per,comparison,flush=True)
