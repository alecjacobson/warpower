"""Validate the preserved throttled pair without including it in primary timing."""
from pathlib import Path
import hashlib,json,numpy as np
w=Path('.work/vtm');s=(w/'check-prepared-new-cell-final-v2.py').read_text();prefix=s[:s.index('report=dict(scope=')].replace(';assert not dest.exists()','');exec(compile(prefix,str(w/'check-prepared-new-cell-final-v2.py'),'exec'))
out=w/'prepared-new-cell-final-v1';dest=w/'prepared-new-cell-throttled-pair-independent-v1.json';assert not dest.exists();report=dict(scope='Independent geometry of preserved production pair with observed candidate CPU throttling',included_in_primary_timing=False,cases={});meshes=[];data=[]
for phase in ['before','after']:
 result,mesh,vtu=paths(out/('graded-torus16384-'+phase));d=json.loads(result.read_text());assert d['valid']==1 and d['error']==0
 row=dict(result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),median_ms=timing(d),quota_observations={k:v for k,v in d.items() if 'quota' in k},independent=independent(mesh,vtu))
 report['cases'][phase]=row;data.append(d);meshes.append(mesh);dest.write_text(json.dumps(report,indent=2)+'\n')
with np.load(meshes[0]) as a,np.load(meshes[1]) as b:report['comparison']=compare(a,b)
report['work_equal']={k:work(data[0])[k]==work(data[1])[k] for k in work(data[0])}
assert all(report['work_equal'].values()) and all(report['comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
assert all(q['throttled_periods']==0 for q in data[0]['additional_replays_cpu_quota_delta'].values())
assert any(q['throttled_periods']>0 for q in data[1]['additional_replays_cpu_quota_delta'].values())
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(report['comparison'],report['work_equal'],flush=True)
