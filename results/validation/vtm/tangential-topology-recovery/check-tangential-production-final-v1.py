"""Independent geometry, unchanged-case comparisons and newly recovered Armadillo meshes."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
out=w/'tangential-production-final-v1';dest=w/'tangential-production-final-v1-independent.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==28

def paths(stem):return (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'),stem.with_suffix('.npz'),None)
def timing(d):return d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
def work(d):return {k:d[k] for k in ['counts','rounds','topology','quality_rejections','refinement_rejections']}|dict(steps=d.get('steps',d.get('optimizer_steps')),moves=d.get('moves',d.get('quality_moves')))
def independent(path,vtu):
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
report=dict(scope=__doc__,cases={},standalone={})
names=['torus16384-fixed32','torus1000','thin-hand-seed7','cavity','disconnected','plate','armadillo-graded4096','armadillo-graded8192-recovery','armadillo-uniform8192-recovery']

def differences(x,y):
    row=compare(x,y)
    a,b=x['points'].astype(float),y['points'].astype(float)
    row['changed_scalar_coordinates']=int(np.count_nonzero(x['points'].view(np.uint32)!=y['points'].view(np.uint32)))
    row['maximum_coordinate_difference']=float(np.max(np.abs(a-b)))
    row['relative_coordinate_difference']=row['maximum_coordinate_difference']/max(1.0,float(np.linalg.norm(np.ptp(a,axis=0))))
    assert all(row[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),row
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
    assert b['valid']==1 and b['error']==0,(name,b)
    row['independent_after']=independent(bn,bv)
    if a['valid']:
        assert a['error']==0
        row['independent_before']=independent(an,av)
        row['work_equal']={k:work(a)[k]==work(b)[k] for k in work(a)}
        assert all(row['work_equal'].values()),(name,row['work_equal'])
        with np.load(an) as x,np.load(bn) as y:
            row['comparison']=differences(x,y);row['input_arrays_equal']={k:bool(np.array_equal(x[k],y[k])) for k in ['input_points','input_faces']}
        assert all(row['input_arrays_equal'].values())
        row['speedup']=row['before_ms']/row['after_ms']
    else:
        assert name in ['armadillo-graded8192-recovery','armadillo-uniform8192-recovery']
        assert a['error']==64 and not any(a['counts']) and not any(a['topology'])
        assert a['input_off_sha256']==b['input_off_sha256']
        row.update(successful_mesh_speedup_claim=False,baseline_failed_zero_output=True)
        reference=w/'exterior-bvh-final-v2'/(name+'-after');rp=reference/'mesh.npz'
        with np.load(rp) as x,np.load(bn) as y:row['private_prototype_comparison']=differences(x,y)
        old=json.loads((reference/'result.json').read_text())
        row['private_prototype_work_equal']={k:work(old)[k]==work(b)[k] for k in work(old)}
        assert all(row['private_prototype_work_equal'].values())
        row['private_prototype_mesh_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest()
    for key in ['persistent_workspace_pool_bytes','setup_seconds','first_replay_gpu_milliseconds','first_replay_gpu_ms']:
        if key in a and key in b:row[key]=dict(before=a[key],after=b[key])
    if 'torus' not in name:
        observers=[json.loads((out/(name+'-'+phase+'-recovery.json')).read_text()) for phase in ['before','after']]
        row['constructor']={k:{phase:d[k] for phase,d in zip(['before','after'],observers)} for k in ['constructor_seconds','persistent_workspace_pool_bytes']}
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

assert len(report['cases'])==9 and len(report['standalone'])==5
assert sum(row['before_valid']==0 for row in report['cases'].values())==2
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['standalone'],indent=2),flush=True)
