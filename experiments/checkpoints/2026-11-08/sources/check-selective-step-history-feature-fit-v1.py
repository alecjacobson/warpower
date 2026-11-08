"""Independent crease-chain and exact-corner diagnostics for the mechanical gallery."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from benchmarks.feature_quality import measure
w=Path('.work/vtm');out=w/'selective-step-history-feature-fit-v1.json';assert not out.exists()
start=time.monotonic()
while True:
 p=w/'selective-step-history-final-v1-independent.json'
 if p.exists() and json.loads(p.read_text())['complete']:break
 assert time.monotonic()-start<4200
 time.sleep(10)
proof=json.loads(p.read_text());report=dict(scope=__doc__,prior_proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path('benchmarks/feature_quality.py'),Path('tests/test_domain.py')]},cases={},complete=False)
for name in ['thin-hand-seed7','plate']:
 row={};report['cases'][name]=row
 for phase in ['before','after']:
  folder=w/'selective-step-history-final-v1'/(name+'-'+phase);path=folder/'mesh.npz'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['cases'][name]['independent'][phase]['mesh_sha256']
  params=json.loads((folder/'result.json').read_text())['parameters']
  with np.load(path) as d:
   row[phase]=measure(d['points'],d['boundary'],d['roles'],np.unique(d['tetrahedra']),d['input_points'],d['input_faces'],feature_angle=params['feature_angle'],corner_angle=params['corner_angle'])
  row[phase]['mesh_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  assert all(c['exact_coordinate_match'] and c['role']==3 for c in row[phase]['corners'])
  out.write_text(json.dumps(report,indent=2)+'\n')
 row['exact_corners_equal']=row['before']['corners']==row['after']['corners']
 row['chain_diagnostics_equal']=row['before']['chains']==row['after']['chains']
 row['rms_difference']=row['after']['feature_to_boundary']['rms']-row['before']['feature_to_boundary']['rms']
 row['maximum_sample_distance_difference']=row['after']['feature_to_boundary']['maximum_sample_distance']-row['before']['feature_to_boundary']['maximum_sample_distance']
 print(name,row['exact_corners_equal'],row['chain_diagnostics_equal'],row['rms_difference'],row['maximum_sample_distance_difference'],flush=True)
report.update(complete=True);out.write_text(json.dumps(report,indent=2)+'\n')
