"""Functional CPU optimizer controls and backward compatibility, not timing samples."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from benchmarks.cpu_mesher import read_mesh,read_surface
from benchmarks.mesh_quality import measure
from tests.test_extraction import independent_topology
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
w=Path('.work/vtm');out=w/'cpu-quality-optimizers-contracts-v2';assert not out.exists()
build=json.loads((w/'cpu-quality-optimizers-build-v1.json').read_text());assert build['complete']
out.mkdir();surface=w/'cpu-feature-policy-contracts-v3/cube.off';points,faces=read_surface(surface)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(scope=__doc__,build_sha256=sha(w/'cpu-quality-optimizers-build-v1.json'),source_sha256={str(p):sha(p) for p in [Path(__file__),w/'cpu_mesher_quality_optimizers_v1.py',w/'cpu-mesh3-quality-optimizers-v1.cpp']},surface_sha256=sha(surface),native={},runner={},invalid=[],complete=False,limits=['Functional tests can overlap GPU memory audits; their timings are not benchmark samples.','CGAL native regular-triangulation validity plus independent material topology/quality/self-intersections; no unweighted full-hull Delaunay claim.'])
def save(): (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
def run(name,cmd,timeout=120):
 result=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
 (out/(name+'.stdout')).write_text(result.stdout);(out/(name+'.stderr')).write_text(result.stderr)
 return result
def check_mesh(name,path):
 p,t=read_mesh(path);q=measure(p,t,points,faces);assert q['components']==1 and q['boundary_euler']==2
 boundary,_,_=independent_topology(p,t);off=out/(name+'-boundary.off')
 with off.open('w') as f:
  f.write(f'OFF\n{len(p)} {len(boundary)} 0\n');np.savetxt(f,p,fmt='%.17g');np.savetxt(f,np.column_stack((np.full(len(boundary),3),boundary)),fmt='%d')
 exact=subprocess.run([str(w/'surface-reference'),str(off)],text=True,capture_output=True,check=True,timeout=120)
 assert int(exact.stdout.splitlines()[0].split()[-1])==0,exact.stdout
 (out/(name+'-self-intersections.txt')).write_text(exact.stdout)
 return dict(independent_quality=q,boundary_self_intersections=0,mesh_sha256=sha(path))
cases=[('old-default','cpu-mesh3-feature-policy-v2-sequential',1,0,None),('new-default','cpu-mesh3-quality-optimizers-v1-sequential',1,0,None),('explicit-disabled','cpu-mesh3-quality-optimizers-v1-sequential',1,0,[0,0,0,0,0,0]),('perturb','cpu-mesh3-quality-optimizers-v1-sequential',1,0,[1,.2,15,0,0,0]),('exude','cpu-mesh3-quality-optimizers-v1-sequential',1,0,[0,0,0,1,.2,15]),('odt-both','cpu-mesh3-quality-optimizers-v1-sequential',1,2,[1,.2,15,1,.2,15]),('parallel-disabled','cpu-mesh3-quality-optimizers-v1-parallel',4,0,[0,0,0,0,0,0]),('parallel-both','cpu-mesh3-quality-optimizers-v1-parallel',4,2,[1,.2,15,1,.2,15])]
prior_path=w/'cpu-quality-optimizers-contracts-v1/result.json'
prior=json.loads(prior_path.read_text())
assert prior['build_sha256']==report['build_sha256']
for name,binary,threads,odt,options in cases:
    row=prior['native'][name];mesh=Path(row['command'][2])
    assert sha(mesh)==row['mesh_sha256'] and row['exit_code']==0 and row['boundary_self_intersections']==0
    assert row['independent_quality']['components']==1 and row['independent_quality']['boundary_euler']==2
    report['native'][name]=dict(row,proof_reused_from=str(prior_path),proof_sha256=sha(prior_path))
report['reused_native_proofs']=8
save()
assert len({report['native'][n]['mesh_sha256'] for n in ['old-default','new-default','explicit-disabled']})==1
report['historical_disabled_geometry_exact']=True
# Both optimization entry points genuinely affect this coarse cube.
assert report['native']['perturb']['mesh_sha256']!=report['native']['new-default']['mesh_sha256']
assert report['native']['exude']['result']['material_vertices_with_nonzero_weight']>report['native']['new-default']['result']['material_vertices_with_nonzero_weight']
assert report['native']['exude']['mesh_sha256']!=report['native']['new-default']['mesh_sha256']
assert report['native']['exude']['independent_quality']['dihedral_degrees_min']>=15
for name,binary,extra in [('legacy-default','cpu-mesh3-feature-policy-v2-sequential',[]),('new-default-runner','cpu-mesh3-quality-optimizers-v1-sequential',[]),('both-runner','cpu-mesh3-quality-optimizers-v1-sequential',['--perturb','--exude','--perturb-time-limit','.2','--exude-time-limit','.2','--perturb-sliver-bound','15','--exude-sliver-bound','15'])]:
 cmd=['.venv/bin/python',str(w/'cpu_mesher_quality_optimizers_v1.py'),'--binary',str(w/binary),'--surface',str(surface),'--size','.7','--facet-distance','.05','--odt-iterations','0','--repeats','1','--output',str(out/name),*extra]
 r=run(name,cmd);assert r.returncode==0,(name,r.stderr)
 d=json.loads((out/name/'result.json').read_text());assert d['valid'] and d['complete'] and d['successful_samples']==1
 report['runner'][name]=dict(command=cmd,exit_code=0,result_sha256=sha(out/name/'result.json'));save()
# New controls must fail explicitly against an older binary, retaining the failed run.
name='legacy-rejects-optimizers';cmd=['.venv/bin/python',str(w/'cpu_mesher_quality_optimizers_v1.py'),'--binary',str(w/'cpu-mesh3-feature-policy-v2-sequential'),'--surface',str(surface),'--size','.7','--facet-distance','.05','--odt-iterations','0','--repeats','1','--perturb','--output',str(out/name)]
r=run(name,cmd);assert r.returncode==1
d=json.loads((out/name/'result.json').read_text());assert d['complete'] and not d['valid'] and d['median_complete_seconds'] is None and d['runs'][0]['exit_code']==2
report['failed_native_retained']=True;save()
for label,options in [('bad-flag',[2,0,0,0,0,0]),('negative-time',[1,-1,0,0,0,0]),('nan-time',[1,'nan',0,0,0,0]),('bad-bound',[0,0,0,1,0,181]),('nan-bound',[0,0,0,1,0,'nan'])]:
 mesh=out/(label+'.txt');cmd=[str(w/'cpu-mesh3-quality-optimizers-v1-sequential'),str(surface),str(mesh),'.7','.05','0','1','.02','.01','45','non_manifold',*map(str,options)]
 r=run(label,cmd,30);assert r.returncode==1 and not mesh.exists();report['invalid'].append(dict(command=cmd,exit_code=r.returncode,stderr=r.stderr,no_mesh=True));save()
for label,options in [('negative-limit',['--perturb','--perturb-time-limit','-1']),('missing-flag',['--perturb-time-limit','1']),('bad-runner-bound',['--exude','--exude-sliver-bound','181'])]:
 cmd=['.venv/bin/python',str(w/'cpu_mesher_quality_optimizers_v1.py'),'--binary',str(w/'cpu-mesh3-quality-optimizers-v1-sequential'),'--surface',str(surface),'--size','.7','--facet-distance','.05','--output',str(out/label),*options]
 r=run(label,cmd,30);assert r.returncode==2 and not (out/label).exists();report['invalid'].append(dict(command=cmd,exit_code=r.returncode,stderr=r.stderr,no_output=True));save()
report['complete']=True;save();print('Eight native cases, three successful runner cases, retained old-binary rejection and eight invalid-option cases passed.')
