"""Functional native feature controls and historical-default compatibility; not performance samples."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.cpu_mesher import read_mesh
from benchmarks.mesh_quality import measure
from tests.test_domain import cube
from tests.test_extraction import independent_topology

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
w=Path('.work/vtm')
out=w/'cpu-feature-policy-contracts-v2'
assert not out.exists()
build=json.loads((w/'cpu-feature-policy-build-v1.json').read_text())
assert build['complete'] and build['historical_binaries_unchanged']
out.mkdir()
points,faces=cube()
surface=out/'cube.off'

def write_off(path,p,f):
    with path.open('w') as stream:
        stream.write(f'OFF\n{len(p)} {len(f)} 0\n')
        np.savetxt(stream,p,fmt='%.17g')
        np.savetxt(stream,np.column_stack((np.full(len(f),3),f)),fmt='%d')

write_off(surface,points,faces)
report=dict(scope=__doc__,build_sha256=hashlib.sha256((w/'cpu-feature-policy-build-v1.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),surface_sha256=hashlib.sha256(surface.read_bytes()).hexdigest(),native={},runner={},invalid=[],complete=False,limits=['Diagnostic functionality only; CPU timings overlap GPU functional tests and are not benchmark samples.','CGAL native triangulation validity and independent material quality/topology/self-intersection are checked; no independent unweighted complete-hull claim is made for feature-protected Mesh_3 outputs.'])
cases=[('historical-default','cpu-mesh3-target-sequential',1,None),('new-default','cpu-mesh3-feature-policy-v1-sequential',1,None),('new-explicit45','cpu-mesh3-feature-policy-v1-sequential',1,45),('new120','cpu-mesh3-feature-policy-v1-sequential',1,120),('parallel120','cpu-mesh3-feature-policy-v1-parallel',4,120)]
for name,binary,threads,angle in cases:
    mesh=out/(name+'.txt')
    cmd=[str(w/binary),str(surface),str(mesh),'0.7','0.05','0',str(threads),'0.02','0.01']
    if angle is not None:cmd.append(str(angle))
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    (out/(name+'.stdout')).write_text(result.stdout)
    (out/(name+'.stderr')).write_text(result.stderr)
    assert result.returncode==0,(name,result.stderr)
    data=json.loads(result.stdout)
    assert data['feature_angle']==(45 if angle is None else angle)
    p,t=read_mesh(mesh)
    quality=measure(p,t,points,faces)
    assert quality['components']==1 and quality['boundary_euler']==2
    boundary,_,_=independent_topology(p,t)
    path=out/(name+'-boundary.off')
    write_off(path,p,boundary)
    exact=subprocess.run([str(w/'surface-reference'),str(path)],text=True,capture_output=True,check=True,timeout=120)
    assert int(exact.stdout.splitlines()[0].split()[-1])==0,exact.stdout
    (out/(name+'-self-intersections.txt')).write_text(exact.stdout)
    report['native'][name]=dict(command=cmd,exit_code=0,result=data,mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),independent_quality=quality,boundary_self_intersections=0)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
digests=[report['native'][n]['mesh_sha256'] for n in ['historical-default','new-default','new-explicit45']]
assert len(set(digests))==1
assert report['native']['new120']['mesh_sha256']!=digests[0]
report['historical_default_mesh_bitwise_preserved']=True
report['explicit_policy_changes_featured_cube_mesh']=True
for name,binary,angle in [('legacy-default','cpu-mesh3-target-sequential',None),('explicit120','cpu-mesh3-feature-policy-v1-sequential',120)]:
    cmd=['.venv/bin/python',str(w/'cpu_mesher_feature_policy_v1.py'),'--binary',str(w/binary),'--surface',str(surface),'--size','0.7','--facet-distance','0.05','--odt-iterations','0','--repeats','1','--output',str(out/name)]
    if angle is not None:cmd+=['--feature-angle',str(angle)]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    (out/(name+'-runner.stdout')).write_text(result.stdout)
    (out/(name+'-runner.stderr')).write_text(result.stderr)
    assert result.returncode==0,result.stderr
    data=json.loads((out/name/'result.json').read_text())
    assert data['parameters']['feature_angle']==(45 if angle is None else angle)
    assert data['runs'][0]['feature_angle']==data['parameters']['feature_angle']
    report['runner'][name]=dict(command=cmd,exit_code=0,result_sha256=hashlib.sha256((out/name/'result.json').read_bytes()).hexdigest())
for angle in ['0','180','nan']:
    mesh=out/('invalid-'+angle+'.txt')
    cmd=[str(w/'cpu-mesh3-feature-policy-v1-sequential'),str(surface),str(mesh),'0.7','0.05','0','1','0.02','0.01',angle]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    assert result.returncode==1 and not mesh.exists() and 'Feature angle must be finite' in result.stderr
    report['invalid'].append(dict(command=cmd,exit_code=result.returncode,stderr=result.stderr,no_mesh=True))
report['complete']=True
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('Native defaults preserve the historical mesh; explicit policy, parallel mode, runner compatibility and invalid criteria pass.')
