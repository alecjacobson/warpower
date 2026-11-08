"""Independent full geometry and work checks for private large group traversal pairs."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
out=w/'large-stackless-pilot-v1';dest=w/'large-stackless-pilot-v1-independent.json';assert not dest.exists()
commands=json.loads((out/'commands.json').read_text());assert len(commands)==12

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
pairs=[(name,out/(name+'-before'),out/(name+'-after'),True) for name in ['torus16384-fixed32','torus8192-progressive','reserved16384-target4096','graded-torus16384','torus4096','torus1000']]
for name,before,after,check_before in pairs:
    ap,an,av=paths(before);bp,bn,bv=paths(after);a,b=[json.loads(p.read_text()) for p in [ap,bp]]
    quota(a);quota(b)
    parameters=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh','dot']}
    assert parameters(a)==parameters(b),(name,parameters(a),parameters(b))
    row=dict(parameters_equal=True,before=str(ap),after=str(bp),before_result_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),after_result_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),before_ms=timing(a),after_ms=timing(b),work_equal={k:work(a)[k]==work(b)[k] for k in work(a)},before_valid=a['valid'],after_valid=b['valid'])
    report['cases'][name]=row
    if not a['valid'] or not b['valid']:
        row['successful_mesh_speedup_claim']=False
        row['failed_state_equal']={k:a[k]==b[k] for k in ['valid','error','status','counts','topology','rounds','steps','moves','last_errors','last_bad_extraction','quality_rejections','refinement_rejections']}
        assert all(row['failed_state_equal'].values()),(name,row['failed_state_equal'])
        assert a['valid']==b['valid']==0 and a['error']==b['error']!=0 and not any(a['counts']) and not any(b['counts'])
    else:
        assert a['error']==b['error']==0
        row['independent_after']=independent(bn,bv)
        if check_before:row['independent_before']=independent(an,av)
        else:row['before_validation_scope']='Independent control validation previously retained in fixed-site-current-larger-v1-independent.json or extended-reuse-gallery-pilot-v2-independent.json.'
        with np.load(an) as x,np.load(bn) as y:
            row['comparison']=compare(x,y);row['input_arrays_equal']={k:bool(np.array_equal(x[k],y[k])) for k in ['input_points','input_faces']}
        row['speedup']=row['before_ms']/row['after_ms']
    for key in ['persistent_workspace_pool_bytes','setup_seconds','first_replay_gpu_milliseconds','first_replay_gpu_ms']:
        if key in a and key in b:row[key]=dict(before=a[key],after=b[key])
    dest.write_text(json.dumps(report,indent=2)+'\n');print(name,row.get('speedup'),row.get('comparison'),row['work_equal'],flush=True)
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
for name,row in report['cases'].items():
    assert all(row['work_equal'].values()),name
    if 'comparison' in row:
        assert all(row['input_arrays_equal'].values()),name
        assert all(row['comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),name
        assert row['comparison']['maximum_coordinate_difference']<=1e-8,name
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['standalone'],indent=2),flush=True)
