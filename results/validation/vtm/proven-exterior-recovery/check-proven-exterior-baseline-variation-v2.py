"""Audit the single canonical mismatch against an independent identical baseline run."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');out=w/'proven-exterior-baseline-variation-v2.json';assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report_paths=[w/'proven-recovery-broad-v1-independent.json',w/'proven-exterior-final-independent-v2.json']
old,final=[json.loads(p.read_text()) for p in report_paths]
healthy=final['healthy_equivalence'];assert len(healthy)==9
assert [k for k,v in healthy.items() if not v['comparison']['canonical_unchanged']]==['hand-seed1']
paths={'earlier_baseline':w/'proven-recovery-broad-v1/hand-seed1-before','final_baseline':w/'proven-exterior-final/hand-seed1-before','production':w/'proven-exterior-final/hand-seed1-after'}
commands=[json.loads((w/'proven-recovery-broad-v1/commands.json').read_text()),json.loads((w/'proven-exterior-final/commands.json').read_text())]
records=[old['cases']['hand-seed1']['before'],final['cases']['hand-seed1']['before'],final['cases']['hand-seed1']['after']]
rows={};data={}
for i,(label,folder) in enumerate(paths.items()):
 result=folder/'result.json';raw=json.loads(result.read_text());d=np.load(folder/'mesh.npz');data[label]=d
 command=next(c for c in commands[int(i>0)] if c['name']=='hand-seed1' and c['phase']==('after' if i==2 else 'before'))
 assert sha(result)==command['result_sha256']
 assert command['source']==('fe3e46910270e28f0b53394a042d09d2a229cee9' if i==2 else 'a34910d64a27dbb8e0fb03f0813e76e00e799744')
 assert raw['valid']==1 and raw['error']==0
 independent=records[i]['independent'];assert independent['full_hull']['valid'] and independent['all_boundary_vertex_links_single_cycles']
 rows[label]=dict(path=str(folder),source=command['source'],result_sha256=sha(result),mesh_sha256=sha(folder/'mesh.npz'),parameters={k:v for k,v in raw['parameters'].items() if k not in ['output','proven_exterior_recovery']},work={k:raw[k] for k in ['counts','rounds','steps','moves','quality_rejections','refinement_rejections']},quality=raw['quality'],independent_quality=independent['quality'])
 if i==2:
  assert raw['proven_exterior_recovery']['enabled_after_fitting']==0 and not any(raw['proven_exterior_recovery']['midpoint_statistics'])
assert len({json.dumps(x['parameters'],sort_keys=True) for x in rows.values()})==1
assert len({json.dumps(x['work'],sort_keys=True) for x in rows.values()})==1
pairs={}
for a,b in [('earlier_baseline','final_baseline'),('earlier_baseline','production'),('final_baseline','production')]:
 x,y=data[a],data[b]
 for key in ['input_points','input_faces','roles']:np.testing.assert_array_equal(x[key],y[key])
 changed=np.argwhere(x['points'].view(np.uint32)!=y['points'].view(np.uint32))
 expected=[[125,0],[280,2]] if a=='final_baseline' else [[125,0],[280,2],[433,2],[830,2]]
 assert changed.tolist()==expected,changed
 delta=float(np.max(np.abs(x['points'].astype(float)-y['points'].astype(float))));assert delta<1e-17
 differences={}
 for key in ['full_tetrahedra','tetrahedra','boundary']:
  aa=set(map(tuple,np.sort(x[key],axis=1)));bb=set(map(tuple,np.sort(y[key],axis=1)))
  differences[key]=dict(removed=[list(map(int,v)) for v in sorted(aa-bb)],added=[list(map(int,v)) for v in sorted(bb-aa)])
 np.testing.assert_allclose(np.sort(x['radius_ratio']),np.sort(y['radius_ratio']),rtol=1e-11,atol=1e-14)
 for key in ['radius_ratio_percentiles']:
  np.testing.assert_allclose(list(rows[a]['independent_quality'][key].values()),list(rows[b]['independent_quality'][key].values()),rtol=1e-11,atol=1e-14)
 np.testing.assert_allclose(rows[a]['independent_quality']['two_sided_area_weighted_rms'],rows[b]['independent_quality']['two_sided_area_weighted_rms'],rtol=1e-11,atol=1e-14)
 pairs[a+'__'+b]=dict(maximum_coordinate_difference=delta,changed_coordinates=[dict(site=int(i),axis=int(j),role=int(x['roles'][i]),before=float(x['points'][i,j]),after=float(y['points'][i,j])) for i,j in changed],differences=differences)
assert all(not row['added'] and not row['removed'] for row in pairs['earlier_baseline__production']['differences'].values())
assert [len(v['added']) for v in pairs['earlier_baseline__final_baseline']['differences'].values()]==[8,6,2]
assert pairs['earlier_baseline__final_baseline']['differences']=={k:{'removed':v['added'],'added':v['removed']} for k,v in pairs['final_baseline__production']['differences'].items()}
report=dict(scope='The sole final canonical mismatch also occurs between identical-parameter baseline runs at a34910d. Production matches one baseline connectivity. This demonstrates pre-existing output variation, not its exact kernel origin. All three meshes were independently certified; this is not a claim of bitwise reproducibility.',source_sha256=sha(Path(__file__)),independent_reports=[dict(path=str(p),sha256=sha(p)) for p in report_paths],variants=rows,pairs=pairs,audited_existing_baseline_variation=True,other_healthy_pairs_canonical_unchanged=True,all_nine_healthy_pairs_same_work=True)
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['variants','pairs']},indent=2))
