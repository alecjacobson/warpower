"""Independent final geometry, iteration-work, quota and API-regression gates."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'.work/vtm';out=w/'large-reuse-final'
# Reuse only the offline helper definitions, without executing the pilot matrix.
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
out=w/'large-reuse-final'
commands=json.loads((out/'commands.json').read_text())
assert len(commands)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
def paths(stem):
 return (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu') if stem.is_dir() else (stem.with_suffix('.json'),stem.with_suffix('.npz'),None)
def timing(d):return d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
def parameters(d):return {k:v for k,v in d['parameters'].items() if k not in ['save_mesh','dot','output']}
summary=dict(runtime='f8b5cfaf2625f264dd95448a272849b88c0c353b',baseline='cca7a1de4f5ddf3c190f321cfda53c66f0525948',cases={},standalone={})
saved_summary=w/'large-reuse-final-independent.json'
if saved_summary.exists():summary=json.loads(saved_summary.read_text())
pairs=[('torus4096',w/'extraction-split-4096/control')]+[(s,w/'large-reuse-proof-pairs'/(s+'-control')) for s in ['graded-torus4096','armadillo-uniform','armadillo-graded']]
pairs +=[(s,out/(s+'-before')) for s in ['torus1000','torus2050','torus3072','reserved4096-target1000','reserved4096-target2048','reserved8192-target4096']]
for name,before in pairs:
 ap,an,av=paths(before);bp,bn,bv=paths(out/(name+'-after'))
 a,b=[json.loads(p.read_text()) for p in [ap,bp]]
 if name in summary['cases']:
  prior=summary['cases'][name]
  assert prior['before_sha256']==hashlib.sha256(an.read_bytes()).hexdigest() and prior['after_sha256']==hashlib.sha256(bn.read_bytes()).hexdigest()
  print('retained completed independent validation',name,flush=True);continue
 for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
 assert parameters(a)==parameters(b),(name,parameters(a),parameters(b))
 x,y=np.load(an),np.load(bn);comparison=compare(x,y)
 assert all(v for k,v in comparison.items() if k!='maximum_coordinate_difference'),(name,comparison)
 inputs={k:bool(np.array_equal(x[k],y[k])) for k in ['input_points','input_faces']}
 assert all(inputs.values())
 work_keys=['counts','rounds','refinement_rejections','quality_rejections']
 work_keys+=['optimizer_steps','quality_moves'] if 'optimizer_steps' in a else ['steps','moves']
 if 'reconstruction_routing' in a or 'reconstruction_routing' in b:work_keys.append('reconstruction_routing')
 work={k:a[k]==b[k] for k in work_keys};assert all(work.values()),(name,work)
 row=dict(before_result=str(ap.relative_to(root)),after_result=str(bp.relative_to(root)),before_sha256=hashlib.sha256(an.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(bn.read_bytes()).hexdigest(),before_ms=timing(a),after_ms=timing(b),speedup=timing(a)/timing(b),comparison=comparison,input_arrays_equal=inputs,work_equal=work,parameters_equal=True,all_first_and_timed_quotas_clean=True,independent=validate(bn,bv))
 for phase,d in [('before',a),('after',b)]:
  row[phase+'_setup_seconds']=d['setup_seconds'];row[phase+'_first_replay_ms']=d.get('first_replay_gpu_milliseconds',d.get('first_replay_gpu_ms'))
  if 'persistent_workspace_pool_bytes' in d:row[phase+'_pool_bytes']=d['persistent_workspace_pool_bytes']
 summary['cases'][name]=row
 (w/'large-reuse-final-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('checked',name,row['before_ms'],row['after_ms'],flush=True)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
 if 'full' in a:
  for d in [a,b]:
   assert all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
   assert d['error']==0
  skip={'first_replay_ms','full','base','auxiliary','certificate'}
  unchanged={k:a[k]==b[k] for k in a if k not in skip};assert all(unchanged.values()),(name,unchanged)
  am,bm=a['full']['median_ms'],b['full']['median_ms']
 else:
  keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics']
  unchanged={k:a[k]==b[k] for k in keys};assert all(unchanged.values()),(name,unchanged)
  assert not a['cpu_throttled'] and not b['cpu_throttled']
  am,bm=1000*a['median_seconds'],1000*b['median_seconds']
 summary['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=unchanged,all_timed_quotas_clean=True)
(w/'large-reuse-final-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['standalone'],indent=2),flush=True)
