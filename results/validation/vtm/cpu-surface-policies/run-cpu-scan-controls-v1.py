"""Retain initial topology/fit/ODT controls without selecting a CPU speedup winner."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root=Path.cwd();w=root/'.work/vtm';out=w/'cpu-scan-controls-v1'
assert not out.exists()
assert json.loads((w/'cpu-scan-feature-pilot-v2-review.json').read_text())['complete']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
out.mkdir()
binary=w/'cpu-mesh3-feature-policy-v2-sequential'
runner=w/'cpu_mesher_feature_policy_v3.py'
surface=w/'surface-query-order-final-v1/armadillo-uniform8192-recovery-after/input.off'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[('odt0',.045,.005,0,.02,.01),('odt1',.045,.005,1,.02,.01),('odt8',.045,.005,8,.02,.01),('tight-fit',.045,.002,32,.02,.01),('loose-fit',.045,.01,32,.02,.01),('loose-finer',.04,.01,32,.02,.01),('loose-strict-odt',.045,.01,100,.001,.001)]
report=dict(scope=__doc__,surface_sha256=sha(surface),runner_sha256=sha(runner),binary_sha256=sha(binary),driver_sha256=sha(Path(__file__)),reference_gpu_source='520a68cf5f8dffc77ac8117bf6fb6ce0f9ed2219',feature_angle=120,topology='manifold',threads=1,repeats=1,cases={},complete=False,limits=['Single initial control per setting; not a repeated performance estimate or a quality-matched speedup claim.','Native triangulation/material-metric status is recorded separately. Further vertex-link/self-intersection review is required for each initially valid output.','All unsuccessful outputs, native timing/quota records and exceptions are retained.'])
dest=out/'report.json';dest.write_text(json.dumps(report,indent=2)+'\n')
for name,h,d,it,conv,freeze in rows:
    cmd=[str(root/'.venv/bin/python'),str(runner),'--binary',str(binary),'--surface',str(surface),'--size',str(h),'--facet-distance',str(d),'--odt-iterations',str(it),'--convergence',str(conv),'--freeze-bound',str(freeze),'--feature-angle','120','--topology','manifold','--threads','1','--repeats','1','--distance-binary',str(w/'surface-distance'),'--output',str(out/name)]
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    row=dict(command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start,log_sha256=sha(out/(name+'.log')))
    report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n')
    assert p.returncode in [0,1]
    data=json.loads((out/name/'result.json').read_text());assert data['complete']
    row.update(result_sha256=sha(out/name/'result.json'),initial_material_valid=data['valid'],successful_samples=data['successful_samples'],unthrottled_samples=data['unthrottled_samples'])
    assert (p.returncode==0)==data['valid']
    dest.write_text(json.dumps(report,indent=2)+'\n')
    r=data['runs'][0]
    print(name,'valid',data['valid'],'sites',r.get('vertices'),'native seconds',r.get('complete_seconds'),'throttled',r['cpu_throttled'],flush=True)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
