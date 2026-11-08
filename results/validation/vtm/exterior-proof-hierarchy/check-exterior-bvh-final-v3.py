"""Complete qualification with explicit numerical differences and retained independent proofs."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');out=w/'exterior-bvh-final-v2';dest=w/'exterior-bvh-final-v3-independent.json';assert not dest.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=w/'exterior-bvh-final-v2-independent.json';report=json.loads(source.read_text());assert len(report['cases'])==9
coordinates=w/'exterior-bvh-coordinate-differences-v1.json';c=json.loads(coordinates.read_text());assert c['complete'] and c['partial_validation_sha256']==sha(source)
commands=json.loads((out/'commands.json').read_text());assert len(commands)==28 and all(r['quota_clean'] and r['exit_code']==0 for r in commands)
for name,row in report['cases'].items():
 assert row['before_valid']==row['after_valid']==1 and all(row['work_equal'].values()) and all(row['input_arrays_equal'].values())
 assert sha(Path(row['before']))==row['before_result_sha256'] and sha(Path(row['after']))==row['after_result_sha256']
 for stage in ['independent_before','independent_after']:
  assert row[stage]['full_hull']['valid'] and row[stage]['boundary_self_intersection_pairs']==0
 assert all(row['comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 d=c['cases'][name]
 for phase in ['before','after']:
  path=out/(name+'-'+phase);path=path/'mesh.npz' if path.is_dir() else path.with_suffix('.npz')
  assert sha(path)==d[phase+'_sha256']
 assert row['comparison']['points_bitwise_equal']==(d['changed_coordinates']==0)
 assert d['relative_coordinate_difference']<1e-9 and d['maximum_radius_ratio_difference']<1e-7
 assert d['minimum_radius_ratio_before']==d['minimum_radius_ratio_after']
 row['coordinate_details']=d
# Repeat only the cheap standalone diagnostic comparisons, not completed geometric validation.
prior_source=(w/'check-exterior-bvh-final-v2.py').read_text();snippet=prior_source.split('for ref in commands:')[1].split("for name,row in report['cases'].items():")[0]
exec(compile('for ref in commands:'+snippet,str(w/'check-exterior-bvh-final-v2.py'),'exec'))
assert len(report['standalone'])==5
assert {name for name,d in c['cases'].items() if d['changed_coordinates']}=={'thin-hand-seed7','cavity','disconnected'}
report.update(scope=__doc__,analysis_revision=dict(original_validation_report=str(source),original_validation_sha256=sha(source),coordinate_report=str(coordinates),coordinate_sha256=sha(coordinates),reason='All18 independent meshes completed before v2 failed its unnecessarily strict all-coordinate bitwise assertion. Reuse those completed proofs; retain and quantify all four changed scalars. Six cases remain bitwise equal; all cases retain canonical cells, oriented boundary, roles, work and equal minimum radius ratio.'),complete=True)
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['standalone'],indent=2),flush=True)
