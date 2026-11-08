"""Production geometry and performance qualification with explicit exact-proof reuse."""
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

w=Path('.work/vtm')
helper=w/'check-surface-bound-reuse-final-v2.py'
exec(compile(helper.read_text().split("out=w/'surface-bound-reuse-final'")[0],str(helper),'exec'))
out=w/'surface-query-order-final-v1'
dest=w/'surface-query-order-final-v1-independent.json'
assert not dest.exists()
commands=json.loads((out/'commands.json').read_text())
assert len(commands)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
source=json.loads((w/'surface-query-order-production-source-v1.json').read_text())
orientation_helper=w/'check-tangential-production-memory-v3.py'
fn=next(n for n in ast.parse(orientation_helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='oriented_rows')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(orientation_helper),'exec'))
references={}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def paths(stem):return (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'),stem.with_suffix('.npz'),None)

for stem in ['surface-query-order-pilot-v2','surface-query-order-pilot-v1']:
    proof_path=w/(stem+'-independent.json')
    proof=json.loads(proof_path.read_text())
    assert proof['complete']
    for name,c in proof['cases'].items():
        for phase in ['after','before']:
            references.setdefault(name,[]).append((paths(w/stem/(name+'-'+phase))[1],c['independent'][phase],proof_path,sha(proof_path)))
proof_path=w/'tangential-production-final-v1-independent.json'
proof=json.loads(proof_path.read_text())
assert proof['complete']
for name,c in proof['cases'].items():
    references.setdefault(name,[]).append((paths(w/'tangential-production-final-v1'/(name+'-after'))[1],c['independent_after'],proof_path,sha(proof_path)))

def exact_arrays(reference,path):
    checks={}
    with np.load(reference) as x,np.load(path) as y:
        if x.files!=y.files:return None
        for key in x.files:
            a,b=x[key],y[key]
            if a.shape!=b.shape or a.dtype!=b.dtype:return None
            if key in ['full_tetrahedra','tetrahedra','boundary']:
                equal=np.array_equal(oriented_rows(a),oriented_rows(b));kind='equal oriented simplex connectivity'
            elif key=='radius_ratio':
                ids=[np.lexsort(np.sort(z['tetrahedra'],axis=1).T[::-1]) for z in [x,y]]
                equal=a[ids[0]].tobytes()==b[ids[1]].tobytes();kind='bitwise equal quality aligned by cell indices'
            else:
                equal=a.tobytes()==b.tobytes();kind='bitwise equal'
            if not equal:return None
            checks[key]=kind
    return checks

def independent(name,path,vtu):
    for reference,proof,report_path,digest in references.get(name,[]):
        assert sha(reference)==proof['mesh_sha256']
        assert proof['full_hull']['valid'] and proof['boundary_self_intersection_pairs']==0
        # A current VTU still needs its own round trip, even when geometry is identical.
        if vtu is not None:break
        checks=exact_arrays(reference,path)
        if checks is not None:
            row=dict(proof)
            row.update(mesh_sha256=sha(path),transferred_by_exact_geometry=True,reference_mesh=str(reference),reference_mesh_sha256=proof['mesh_sha256'],reference_report=str(report_path),reference_proof_sha256=digest,exact_comparison=checks,orientation_helper_sha256=sha(orientation_helper))
            return row
    row=validate(path,vtu)
    if vtu is None:
        with np.load(path) as m,tempfile.TemporaryDirectory(prefix='ordered-boundary-',dir='/opt/warpower-generated-cache') as tmp:
            p=Path(tmp)/'boundary.off'
            with p.open('w') as f:
                f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n')
                np.savetxt(f,m['points'],fmt='%.17g')
                np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
            ref=subprocess.run([str(w/'surface-reference'),str(p)],text=True,capture_output=True,check=True)
            assert int(ref.stdout.splitlines()[0].split()[-1])==0,ref.stdout
            row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=ref.stdout)
    return row

def work(d):
    return {k:d[k] for k in ['counts','rounds','topology','quality_rejections','refinement_rejections']}|dict(steps=d.get('steps',d.get('optimizer_steps')),moves=d.get('moves',d.get('quality_moves')))

report=dict(scope=__doc__,baseline=source['baseline'],runtime=source['runtime'],cases={},standalone={},complete=False,limits=['Full-hull or coordinate differences remain explicit and require review, even when all current meshes independently validate.','Reference proofs transfer only through exact arrays/oriented connectivity/quality. Gallery meshes receive fresh independent validation including their current VTU round trips.','Measured GPU revision comparisons do not establish the comparable-quality CPU goal.'])
names=['torus16384-fixed32','torus1000','torus8192-fixed32','torus8192-progressive','thin-hand-seed7','cavity','disconnected','plate','armadillo-graded4096','armadillo-graded8192-recovery','armadillo-uniform8192-recovery']
for name in names:
    pair=[paths(out/(name+'-'+phase)) for phase in ['before','after']]
    a,b=[json.loads(p[0].read_text()) for p in pair]
    params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh','dot']}
    assert params(a)==params(b)
    for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
    row=dict(parameters_equal=True,work_equal={k:work(a)[k]==work(b)[k] for k in work(a)},independent={},timing={})
    assert all(row['work_equal'].values()),(name,row['work_equal'])
    for phase,d,(result,mesh,vtu) in zip(['before','after'],[a,b],pair):
        row['independent'][phase]=independent(name,mesh,vtu)
        row['timing'][phase+'_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
        row['timing'][phase+'_setup_seconds']=d['setup_seconds']
        if 'persistent_workspace_pool_bytes' in d:
            row['timing'][phase+'_pool_bytes']=d['persistent_workspace_pool_bytes']
        else:
            observed=json.loads((out/(name+'-'+phase+'-recovery.json')).read_text())
            row['timing'][phase+'_pool_bytes']=observed['persistent_workspace_pool_bytes']
        row[phase+'_result_sha256']=sha(result)
    row['timing']['speedup']=row['timing']['before_ms']/row['timing']['after_ms']
    row['timing']['additional_pool_bytes']=row['timing']['after_pool_bytes']-row['timing']['before_pool_bytes']
    with np.load(pair[0][1]) as x,np.load(pair[1][1]) as y:
        c=compare(x,y)
        c['input_arrays_equal']={k:x[k].tobytes()==y[k].tobytes() for k in ['input_points','input_faces']}
        c['changed_scalar_coordinates']=int(np.count_nonzero(x['points'].view(np.uint32)!=y['points'].view(np.uint32)))
        c['relative_coordinate_difference']=float(np.max(np.abs(x['points'].astype(float)-y['points'].astype(float))))/max(1.,float(np.linalg.norm(np.ptp(x['points'].astype(float),axis=0))))
        assert all(c[k] for k in ['roles_equal','material_tets_equal','oriented_boundary_equal']) and all(c['input_arrays_equal'].values()),(name,c)
        assert c['relative_coordinate_difference']<1e-9,(name,c)
        if 'radius_ratio' in x:
            ids=[np.lexsort(np.sort(z['tetrahedra'],axis=1).T[::-1]) for z in [x,y]]
            qa,qb=x['radius_ratio'][ids[0]],y['radius_ratio'][ids[1]]
            c['maximum_stored_quality_difference']=float(np.max(np.abs(qa-qb)))
            assert c['maximum_stored_quality_difference']<1e-7,(name,c)
        row['comparison']=c
    qa,qb=[row['independent'][phase]['quality']['radius_ratio_percentiles']['0'] for phase in ['before','after']]
    row['minimum_quality']=dict(before=qa,after=qb)
    assert abs(qa-qb)<1e-12,(name,qa,qb)
    row['strict_trajectory_parity']=all(c[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
    report['cases'][name]=row
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['timing'],'strict parity',row['strict_trajectory_parity'],flush=True)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
    a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
    if 'full' in a:
        for d in [a,b]:assert d['error']==0 and all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
        skip={'first_replay_ms','full','base','auxiliary','certificate'}
        unchanged={k:a[k]==b[k] for k in a if k not in skip}
        am,bm=a['full']['median_ms'],b['full']['median_ms']
    else:
        keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics']
        unchanged={k:a[k]==b[k] for k in keys}
        assert not a['cpu_throttled'] and not b['cpu_throttled']
        am,bm=1000*a['median_seconds'],1000*b['median_seconds']
    assert all(unchanged.values()),(name,unchanged)
    report['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=unchanged)
report['complete']=True
report['exact_geometry_transfers']=sum(c['independent'][phase].get('transferred_by_exact_geometry',False) for c in report['cases'].values() for phase in ['before','after'])
report['strict_trajectory_parity']=all(c['strict_trajectory_parity'] for c in report['cases'].values())
dest.write_text(json.dumps(report,indent=2)+'\n')
