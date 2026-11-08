"""Close the final assessment with explicit baseline-existing material variability."""
import hashlib
import json
from pathlib import Path

w=Path('.work/vtm');dest=w/'surface-query-cache-final-independent-v3.json';assert not dest.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((w/name).read_text())
base_name='surface-query-cache-final-independent-v2.json';r=read(base_name)
assert r['geometry_complete'] and not r['complete'] and r['equivalence_failures']==['hand-seed1']
assert len(r['equivalence'])==13 and len(r['standalone'])==5
assert sum(c['valid'] for phases in r['cases'].values() for c in phases.values())==24
assert r['strict_failure_sha256']==sha(w/'surface-query-cache-final-independent-v1.json')
for name,phases in r['cases'].items():
    for phase,c in phases.items():
        folder=w/'surface-query-cache-final'/f'{name}-{phase}'
        assert c['result_sha256']==sha(folder/'result.json')
        if c['valid']:assert c['independent']['mesh_sha256']==sha(folder/'mesh.npz') and c['independent']['full_hull']['valid']
for c in r['equivalence'].values():assert all(c['work_equal'].values())
local=read('surface-query-cache-material-variation-v1.json')
assert local['independent_report_sha256']==sha(w/base_name)
assert local['mesh_sha256']=={phase:r['cases']['hand-seed1'][phase]['independent']['mesh_sha256'] for phase in ['before','after']}
assert local['features_exactly_equal'] and local['recorded_work_equal'] and local['oriented_patch_boundary_edges_equal']
repeats=read('surface-query-cache-repeats-independent-v1.json');assert repeats['complete']
for phase,v in repeats['phases'].items():
    assert len(v['replays'])==8 and all(n>0 for n in v['material_alternative_counts'].values())
    for c in v['replays']:
        assert c['independent']['full_hull']['valid'] and c['independent']['boundary_self_intersection_pairs']==0
        assert c['features_equal_to_final_reference'] and all(c['recorded_work_equal'].values())
        path=w/f'surface-query-cache-repeat-{phase}-v1'/f"replay-{c['index']}.npz"
        assert c['sha256']==sha(path)==c['independent']['mesh_sha256']
paired=read('surface-query-cache-paired-audit-independent-v1.json');assert paired['complete']
assert all(paired['query_statistics'][k]==0 for k in ['ray_mismatch','nearest_face_mismatch','nearest_distance_bits_mismatch','nearest_point_bits_mismatch'])
assert paired['query_statistics']['ray_hits']>0 and paired['query_statistics']['nearest_hits']>0
assert all(paired['work_equal'].values())
assert paired['independent']['mesh_sha256']==sha(w/'surface-query-cache-paired-audit-hand1-v1/mesh.npz')
r['assessment_source_sha256']=sha(Path(__file__))
r['validated_geometry_report_sha256']=sha(w/base_name)
r['accepted_material_variation']={
    'case':'hand-seed1','issue':220,
    'strict_connectivity_equivalence':False,
    'mean_quality_identity':False,
    'reason':'Both material alternatives occur in unchanged baseline and candidate repeats, with independently valid own-frame geometry, identical feature metrics and work. Exact rational local analysis bounds boundary displacement. The full-trajectory paired audit finds no changed surface-query answers. Accept baseline-existing variability, without claiming equal output distributions or a unique cause of coordinate differences.',
    'evidence':{name:sha(w/name) for name in ['surface-query-cache-material-variation-v1.json','surface-query-cache-repeats-independent-v1.json','surface-query-cache-paired-audit-independent-v1.json','surface-query-cache-paired-source-audit-v1.json']},
    'repeat_alternative_counts':{phase:v['material_alternative_counts'] for phase,v in repeats['phases'].items()},
    'two_sided_output_boundary_upper_bound':local['two_sided_boundary_hausdorff_upper_bound'],
    'exact_bound':local['two_sided_boundary_hausdorff_upper_bound_exact'],
    'mean_radius_ratio':{phase:local['quality'][phase]['mean_radius_ratio'] for phase in ['before','after']},
    'query_statistics':paired['query_statistics']}
# Keep equivalence_failures and the original failed reports: their strict
# connectivity claim remains false. Complete refers to this explicit assessment.
r['complete']=True
dest.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(complete=True,valid_final_meshes=24,valid_repeat_meshes=16,valid_paired_audit_meshes=1,strict_material_mismatch_retained=r['equivalence_failures'],exterior_variation=list(r['exterior_variation'])),indent=2))
