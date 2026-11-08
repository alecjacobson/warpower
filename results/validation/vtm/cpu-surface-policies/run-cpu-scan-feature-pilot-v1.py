"""Initial scanned-domain CPU quality control; no quality-matched speedup claim."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from benchmarks.cpu_mesher import read_mesh, read_surface
from tests.test_extraction import independent_topology

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root=Path.cwd()
w=root/'.work/vtm'
out=w/'cpu-scan-feature-pilot-v1'
assert not out.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
build=json.loads((w/'cpu-feature-policy-build-v1.json').read_text())
assert build['complete'] and build['historical_binaries_unchanged']
assert json.loads((w/'cpu-distance-policy-contracts-v1/result.json').read_text())['complete']
surface=w/'surface-query-order-final-v1/armadillo-uniform8192-recovery-after/input.off'
gpu_path=w/'surface-query-order-final-v1/armadillo-uniform8192-recovery-after/mesh.npz'
p,f=read_surface(surface)
with np.load(gpu_path) as gpu:
    np.testing.assert_array_equal(p.astype(np.float32),gpu['input_points'])
    np.testing.assert_array_equal(f,gpu['input_faces'])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
out.mkdir()
report=dict(scope=__doc__,source_sha256={str(path.relative_to(root)):sha(path) for path in [Path(__file__).resolve(),w/'cpu_mesher_feature_policy_v2.py',w/'cpu-mesh3-feature-policy-v1.cpp',root/'benchmarks/mesh_quality.py',root/'benchmarks/surface_distance.py',w/'surface-distance',w/'surface-reference']},build_sha256=sha(w/'cpu-feature-policy-build-v1.json'),surface=str(surface),surface_sha256=sha(surface),gpu_mesh_sha256=sha(gpu_path),identical_input_arrays=True,configuration=dict(size=.045,facet_distance=.005,feature_angle=120,odt_iterations=32,convergence=.02,freeze_bound=.01),selection='Initial size estimate only, not selected for an asserted match. Results determine subsequent size/fit/ODT controls.',runs={},complete=False,limitations=['GPU corner-angle and boundary-update semantics are not established equivalent to CGAL feature protection merely by equal feature-angle values.','CPU outputs are material complexes with feature-protection weights; no independent unweighted complete-hull claim is made.','Native complete time includes domain/setup, meshing and ODT, while GPU warmed replay excludes workspace setup. Both scopes and native component times remain explicit.','This first control is not a matched-quality speedup comparison. All throttled/failed samples remain retained.'])
dest=out/'report.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
for mode,threads in [('sequential',1),('parallel',4)]:
    binary=w/f'cpu-mesh3-feature-policy-v1-{mode}'
    row=next(r for r in build['builds'] if r['mode']==mode)
    assert sha(binary)==row['binary_sha256']
    case=out/mode
    cmd=[str(root/'.venv/bin/python'),str(w/'cpu_mesher_feature_policy_v2.py'),'--binary',str(binary),'--surface',str(surface),'--size','.045','--facet-distance','.005','--odt-iterations','32','--feature-angle','120','--convergence','.02','--freeze-bound','.01','--threads',str(threads),'--repeats','3','--distance-binary',str(w/'surface-distance'),'--output',str(case)]
    start=time.time()
    with (out/(mode+'.log')).open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    report['runs'][mode]=dict(command=cmd,exit_code=result.returncode,wall_seconds=time.time()-start,binary_sha256=sha(binary),log_sha256=sha(out/(mode+'.log')))
    dest.write_text(json.dumps(report,indent=2)+'\n')
    assert result.returncode==0,mode
    data=json.loads((case/'result.json').read_text())
    checks={}
    for i,r in enumerate(data['runs']):
        mesh=case/f'mesh-{i}.txt';assert sha(mesh)==r['mesh_sha256']
        if r['mesh_sha256'] in checks:continue
        points,tets=read_mesh(mesh)
        boundary,_,_=independent_topology(points,tets)
        path=case/f'boundary-{i}.off'
        with path.open('w') as stream:
            stream.write(f'OFF\n{len(points)} {len(boundary)} 0\n')
            np.savetxt(stream,points,fmt='%.17g')
            np.savetxt(stream,np.column_stack((np.full(len(boundary),3),boundary)),fmt='%d')
        checked=subprocess.run([str(w/'surface-reference'),str(path)],capture_output=True,text=True,timeout=120)
        (case/f'boundary-{i}-reference.txt').write_text(checked.stdout+'\n'+checked.stderr)
        assert checked.returncode==0 and int(checked.stdout.splitlines()[0].split()[-1])==0
        q=r['independent_quality']
        assert q['components']==1 and q['boundary_euler']==2
        checks[r['mesh_sha256']]=dict(positive_material_topology=True,boundary_self_intersection_pairs=0,boundary_sha256=sha(path))
    report['runs'][mode].update(result_sha256=sha(case/'result.json'),independent_checks=checks,unthrottled_samples=data['unthrottled_samples'],median_complete_seconds=data['median_complete_seconds'],median_unthrottled_complete_seconds=data['median_unthrottled_complete_seconds'])
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(mode,report['runs'][mode]['median_complete_seconds'],data['unthrottled_samples'],[(r['vertices'],r['tetrahedra']) for r in data['runs']],flush=True)
report['complete']=True
dest.write_text(json.dumps(report,indent=2)+'\n')
