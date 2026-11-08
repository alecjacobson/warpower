"""Independent complete geometry and standalone performance comparisons for unused-query storage removal."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
out=w/'quality-query-storage-final-v1';dest=w/'quality-query-storage-final-v2-independent.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==20
prior_path=w/'quality-query-storage-final-v1-independent.json'
prior=json.loads(prior_path.read_text())['cases']['torus16384-fixed32']
prior_hash=hashlib.sha256(prior_path.read_bytes()).hexdigest()
reused_meshes={str((out/('torus16384-fixed32-'+phase+'.npz')).resolve()):prior['independent_'+phase] for phase in ['before','after']}
hand_path=w/'quality-query-storage-hand-variation-v2.json'
hand=json.loads(hand_path.read_text());assert hand['complete']
assert hand['baseline_variation_matches_production_difference'] and hand['all_changed_full_cells_are_outside_material']
for phase in ['before','after']:
    assert hashlib.sha256((out/('thin-hand-seed7-'+phase)/'mesh.npz').read_bytes()).hexdigest()==hand['meshes'][phase]['sha256']


def paths(stem):return (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'),stem.with_suffix('.npz'),None)
def timing(d):return d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
def work(d):return {k:d[k] for k in ['counts','rounds','topology','quality_rejections','refinement_rejections']}|dict(steps=d.get('steps',d.get('optimizer_steps')),moves=d.get('moves',d.get('quality_moves')))
def independent(path,vtu):
    if str(path.resolve()) in reused_meshes:
        old=dict(reused_meshes[str(path.resolve())])
        assert old['mesh_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert old['full_hull']['valid'] and old['boundary_self_intersection_pairs']==0
        old['reused_from']='quality-query-storage-final-v1-independent.json'
        old['reused_report_sha256']=prior_hash
        return old
    row=validate(path,vtu)
    if vtu is None:
        with np.load(path) as m,tempfile.TemporaryDirectory(prefix='extended-boundary-',dir='/opt/warpower-generated-cache') as tmp:
            p=Path(tmp)/'boundary.off'
            with p.open('w') as f:
                f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n')
                np.savetxt(f,m['points'],fmt='%.17g');np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
            result=subprocess.run([str(w/'surface-reference'),str(p)],text=True,capture_output=True,check=True)
            assert int(result.stdout.splitlines()[0].split()[-1])==0,result.stdout
            row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=result.stdout)
    return row
report=dict(scope='Independent storage-change qualification; retain exact current data, reuse already-proved torus snapshots, and explicitly account for demonstrated baseline hand exterior-hull variation.',cases={},standalone={})
names=['torus16384-fixed32','thin-hand-seed7','armadillo-graded4096','armadillo-graded8192-recovery','armadillo-uniform8192-recovery']

def differences(x,y,case):
    row=compare(x,y)
    a,b=x['points'].astype(float),y['points'].astype(float)
    row['changed_scalar_coordinates']=int(np.count_nonzero(x['points'].view(np.uint32)!=y['points'].view(np.uint32)))
    row['maximum_coordinate_difference']=float(np.max(np.abs(a-b)))
    row['relative_coordinate_difference']=row['maximum_coordinate_difference']/max(1.0,float(np.linalg.norm(np.ptp(a,axis=0))))
    assert all(row[k] for k in ['roles_equal','material_tets_equal','oriented_boundary_equal']),row
    if not row['full_tets_equal']:
        assert case=='thin-hand-seed7' and hand['production_full_hull_matches_other_unchanged_baseline']
        row['full_hull_variation_report_sha256']=hashlib.sha256(hand_path.read_bytes()).hexdigest()
        row['full_hull_matches_other_certified_unchanged_baseline']=True
        row['changed_full_cells_are_exterior_only']=True
        row['removed_full_cells']=hand['removed_full_cells'];row['added_full_cells']=hand['added_full_cells']

    assert row['relative_coordinate_difference']<1e-9,row
    if 'radius_ratio' in x and 'radius_ratio' in y:
        def sorted_quality(z):
            t=np.sort(z['tetrahedra'],axis=1);return z['radius_ratio'][np.lexsort(t.T[::-1])]
        qa,qb=sorted_quality(x),sorted_quality(y)
        row['maximum_radius_ratio_difference']=float(np.max(np.abs(qa-qb)))
        row['minimum_radius_ratio_before']=float(qa.min());row['minimum_radius_ratio_after']=float(qb.min())
        assert row['maximum_radius_ratio_difference']<1e-7,row
        assert abs(row['minimum_radius_ratio_before']-row['minimum_radius_ratio_after'])<1e-12,row
    return row

for name in names:
    before,after=out/(name+'-before'),out/(name+'-after')
    ap,an,av=paths(before);bp,bn,bv=paths(after);a,b=[json.loads(p.read_text()) for p in [ap,bp]]
    quota(a);quota(b)
    parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh','dot']}
    assert parameters(a)==parameters(b),(name,parameters(a),parameters(b))
    row=dict(parameters_equal=True,before=str(ap),after=str(bp),before_result_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),after_result_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),before_ms=timing(a),after_ms=timing(b),before_valid=a['valid'],after_valid=b['valid'])
    assert a['valid']==b['valid']==1 and a['error']==b['error']==0,(name,a,b)
    row['independent_after']=independent(bn,bv)
    if a['valid']:
        assert a['error']==0
        row['independent_before']=independent(an,av)
        row['work_equal']={k:work(a)[k]==work(b)[k] for k in work(a)}
        assert all(row['work_equal'].values()),(name,row['work_equal'])
        with np.load(an) as x,np.load(bn) as y:
            row['comparison']=differences(x,y,name);row['input_arrays_equal']={k:bool(np.array_equal(x[k],y[k])) for k in ['input_points','input_faces']}
        assert all(row['input_arrays_equal'].values())
        row['speedup']=row['before_ms']/row['after_ms']
    for key in ['persistent_workspace_pool_bytes','setup_seconds','first_replay_gpu_milliseconds','first_replay_gpu_ms']:
        if key in a and key in b:row[key]=dict(before=a[key],after=b[key])
    if 'torus' not in name:
        observers=[json.loads((out/(name+'-'+phase+'-recovery.json')).read_text()) for phase in ['before','after']]
        row['constructor']={k:{phase:d[k] for phase,d in zip(['before','after'],observers)} for k in ['constructor_seconds','persistent_workspace_pool_bytes']}
        guards=[json.loads((out/(name+'-'+phase+'-storage.json')).read_text()) for phase in ['before','after']]
        assert not any(guards[0]['recovery_fields_omitted'].values()) and all(guards[1]['recovery_fields_omitted'].values())
        assert all(g['candidate_cache_retained'] and g['approximate_capture_callback_forbidden'] for g in guards)
        assert all(all(fields.values()) for g in guards for fields in g['ordinary_fields_present'].values())
        row['saved_workspace_bytes']=observers[0]['persistent_workspace_pool_bytes']-observers[1]['persistent_workspace_pool_bytes']
        row['predicted_payload_bytes']=guards[1]['predicted_removed_payload_bytes']
        assert row['saved_workspace_bytes']==row['predicted_payload_bytes']
        row['storage_guards']=guards
        for field in ['chosen','accepted','invoked','attempts','error','status','rejection_reason','point_count','full_count','retained_count']:
            assert observers[0]['recovery'][field]==observers[1]['recovery'][field],(name,field)
        r=observers[1]['recovery'];assert r is not None and r['error']==[0]
        row['recovery']=r
        if '8192' in name:
            assert r['accepted']==r['invoked']==[1] and r['attempts']==([1] if 'graded' in name else [22])
        else:assert r['accepted']==r['invoked']==r['attempts']==[0]
    report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,row.get('speedup'),row.get('comparison',row.get('private_prototype_comparison')),flush=True)
for ref in commands:
    name=ref['name']
    if ref['phase']!='before' or not name.startswith(('delaunay','diagram')):continue
    a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
    if 'full' in a:
        for d in [a,b]:assert d['error']==0 and all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
        skip={'first_replay_ms','full','base','auxiliary','certificate'}
        unchanged={k:a[k]==b[k] for k in a if k not in skip};am,bm=a['full']['median_ms'],b['full']['median_ms']
    else:
        keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics']
        unchanged={k:a[k]==b[k] for k in keys};assert not a['cpu_throttled'] and not b['cpu_throttled'];am,bm=1000*a['median_seconds'],1000*b['median_seconds']
    assert all(unchanged.values()),(name,unchanged)
    report['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=unchanged)

assert len(report['cases'])==5 and len(report['standalone'])==5
assert not any(row['before_valid']==0 for row in report['cases'].values())
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['standalone'],indent=2),flush=True)
