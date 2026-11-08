"""Validate paired quadrature trajectories; transfer geometry proofs only through exact arrays."""
import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();assert not args.output.exists()
w=Path('.work/vtm');quadrature_helper_path=w/'check-surface-bound-reuse-final-v2.py'
exec(compile(quadrature_helper_path.read_text().split("out=w/'surface-bound-reuse-final'")[0],str(quadrature_helper_path),'exec'))
orientation_helper=w/'check-tangential-production-memory-v3.py'
fn=next(n for n in ast.parse(orientation_helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='oriented_rows')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(orientation_helper),'exec'))
old=w/'check-surface-query-order-final-v1.py'
defs=[n for n in ast.parse(old.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['sha','paths','exact_arrays','work']]
exec(compile(ast.Module(body=defs,type_ignores=[]),str(old),'exec'))
proof_path=w/'surface-query-order-final-v1-independent.json'
prior=json.loads(proof_path.read_text());assert prior['complete'];prior_digest=sha(proof_path)
commands=json.loads((args.folder/'commands.json').read_text())
assert all(r['exit_code']==0 and r['quota_clean'] for r in commands)
names=list(dict.fromkeys(r['name'] for r in commands if not r['name'].startswith(('delaunay','diagram'))))
report=dict(scope=__doc__,folder=str(args.folder),cases={},standalone={},complete=False,prior_proof=str(proof_path),prior_proof_sha256=prior_digest,source_sha256={str(p):sha(p) for p in [Path(__file__),quadrature_helper_path,old,orientation_helper,w/'check-large-reuse-proof-pairs.py']})


def exports(path,vtu):
    if vtu is None:return {}
    with np.load(path) as d:
        tree=ET.parse(gzip.open(vtu,'rb') if str(vtu).endswith('.gz') else vtu).getroot()
        p=np.fromstring(tree.find('.//Points/DataArray').text,sep=' ',dtype=np.float32).reshape(-1,3)
        t=np.fromstring(tree.find(".//Cells/DataArray[@Name='connectivity']").text,sep=' ',dtype=np.int32).reshape(-1,4)
        q=np.fromstring(tree.find(".//CellData/DataArray[@Name='radius_ratio']").text,sep=' ')
        np.testing.assert_array_equal(p.view(np.uint32),d['points'].view(np.uint32));np.testing.assert_array_equal(t,d['tetrahedra']);np.testing.assert_array_equal(q,d['radius_ratio'])
        off=path.parent/'boundary.off.gz'
        with gzip.open(off,'rt') as f:
            assert f.readline().strip()=='OFF'
            n,m,e=map(int,f.readline().split());assert e==0
            p=np.loadtxt(f,max_rows=n,dtype=np.float32)
            faces=np.loadtxt(f,max_rows=m,dtype=np.int32)
        np.testing.assert_array_equal(p.view(np.uint32),d['points'].view(np.uint32))
        assert np.all(faces[:,0]==3);np.testing.assert_array_equal(faces[:,1:],d['boundary'])
    return dict(vtu_round_trip_exact=True,boundary_off_round_trip_exact=True)


def independent(name,path,vtu):
    for phase in ['after','before']:
        reference=paths(w/'surface-query-order-final-v1'/(name+'-'+phase))[1]
        proof=prior['cases'][name]['independent'][phase]
        assert sha(reference)==proof['mesh_sha256'] and proof['full_hull']['valid'] and proof['boundary_self_intersection_pairs']==0
        checks=exact_arrays(reference,path)
        if checks is not None:
            row=dict(proof);row.update(mesh_sha256=sha(path),transferred_by_exact_geometry=True,reference_mesh=str(reference),reference_mesh_sha256=proof['mesh_sha256'],reference_proof_sha256=prior_digest,exact_comparison=checks)
            row.update(exports(path,vtu));return row
    row=validate(path,vtu)
    if vtu is None:
        with np.load(path) as d,tempfile.TemporaryDirectory(prefix='quadrature-boundary-',dir='/opt/warpower-generated-cache') as tmp:
            p=Path(tmp)/'boundary.off'
            with p.open('w') as f:
                f.write(f'OFF\n{len(d["points"])} {len(d["boundary"])} 0\n')
                np.savetxt(f,d['points'],fmt='%.17g');np.savetxt(f,np.column_stack((np.full(len(d['boundary']),3),d['boundary'])),fmt='%d')
            text=subprocess.run([str(w/'surface-reference'),str(p)],capture_output=True,text=True,check=True).stdout
            assert int(text.splitlines()[0].split()[-1])==0,text
            row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=text)
    row.update(exports(path,vtu));return row


for name in names:
    pair=[paths(args.folder/(name+'-'+phase)) for phase in ['before','after']]
    a,b=[json.loads(p[0].read_text()) for p in pair]
    params=lambda d:{k:v for k,v in d['parameters'].items() if k not in ['output','save_mesh','dot']}
    assert params(a)==params(b)
    for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
    row=dict(parameters_equal=True,work_equal={k:work(a)[k]==work(b)[k] for k in work(a)},independent={},timing={})
    for phase,d,(result,mesh,vtu) in zip(['before','after'],[a,b],pair):
        row['independent'][phase]=independent(name,mesh,vtu)
        row['timing'][phase+'_ms']=d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
        row[phase+'_result_sha256']=sha(result)
        observer_path=args.folder/(name+'-'+phase+'-observer.json')
        observed=json.loads(observer_path.read_text()) if observer_path.exists() else d
        row['timing'][phase+'_pool_bytes']=observed['persistent_workspace_pool_bytes']
        row['timing'][phase+'_setup_seconds']=observed['constructor_seconds'] if observer_path.exists() else observed['setup_seconds']
        row['timing'][phase+'_setup_scope']='mesher constructor' if observer_path.exists() else 'benchmark setup including domain'
    row['timing']['speedup']=row['timing']['before_ms']/row['timing']['after_ms']
    row['timing']['additional_pool_bytes']=row['timing']['after_pool_bytes']-row['timing']['before_pool_bytes']
    with np.load(pair[0][1]) as x,np.load(pair[1][1]) as y:
        c=compare(x,y);c['input_arrays_equal']={k:x[k].tobytes()==y[k].tobytes() for k in ['input_points','input_faces']}
        c['changed_scalar_coordinates']=int(np.count_nonzero(x['points'].view(np.uint32)!=y['points'].view(np.uint32)))
        c['full_oriented_tets_equal']=bool(np.array_equal(oriented_rows(x['full_tetrahedra']),oriented_rows(y['full_tetrahedra'])))
        c['material_oriented_tets_equal']=bool(np.array_equal(oriented_rows(x['tetrahedra']),oriented_rows(y['tetrahedra'])))
        row['comparison']=c
    row['strict_trajectory_parity']=all(row['work_equal'].values()) and all(c['input_arrays_equal'].values()) and all(c[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal','full_oriented_tets_equal','material_oriented_tets_equal'])
    report['cases'][name]=row;args.output.write_text(json.dumps(report,indent=2)+'\n');print(name,row['timing'],'strict_parity',row['strict_trajectory_parity'],flush=True)
for name in dict.fromkeys(r['name'] for r in commands if r['name'].startswith(('delaunay','diagram'))):
    a,b=[json.loads((args.folder/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
    if 'full' in a:
        for d in [a,b]:assert d['error']==0 and all(v['throttled_periods']==0 for v in d['full']['cpu'].values())
        skip={'first_replay_ms','full','base','auxiliary','certificate'};unchanged={k:a[k]==b[k] for k in a if k not in skip}
        am,bm=a['full']['median_ms'],b['full']['median_ms']
    else:
        keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics'];unchanged={k:a[k]==b[k] for k in keys}
        assert not a['cpu_throttled'] and not b['cpu_throttled'];am,bm=1000*a['median_seconds'],1000*b['median_seconds']
    assert all(unchanged.values()),(name,unchanged);report['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=unchanged)
report['complete']=True;report['strict_trajectory_parity']=all(r['strict_trajectory_parity'] for r in report['cases'].values())
report['exact_geometry_transfers']=sum(r['independent'][phase].get('transferred_by_exact_geometry',False) for r in report['cases'].values() for phase in ['before','after'])
args.output.write_text(json.dumps(report,indent=2)+'\n')
