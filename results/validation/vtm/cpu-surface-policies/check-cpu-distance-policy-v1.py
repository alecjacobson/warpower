"""Check optional native distance validation against the original offline metric."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
w=Path('.work/vtm')
out=w/'cpu-distance-policy-contracts-v1'
assert not out.exists()
out.mkdir()
report=dict(scope=__doc__,source_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['cpu_mesher_feature_policy_v2.py','cpu-mesh3-feature-policy-v1-sequential','surface-distance']},runs={},complete=False,limits='Functional diagnostic only; not a clean CPU performance sample.')
for name in ['brute','native']:
    cmd=['.venv/bin/python',str(w/'cpu_mesher_feature_policy_v2.py'),'--binary',str(w/'cpu-mesh3-feature-policy-v1-sequential'),'--surface',str(w/'cpu-feature-policy-contracts-v2/cube.off'),'--size','0.7','--facet-distance','0.05','--odt-iterations','0','--feature-angle','120','--repeats','1','--output',str(out/name)]
    if name=='native':cmd+=['--distance-binary',str(w/'surface-distance')]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (out/(name+'.stdout')).write_text(p.stdout)
    (out/(name+'.stderr')).write_text(p.stderr)
    assert p.returncode==0,p.stderr
    r=json.loads((out/name/'result.json').read_text())
    report['runs'][name]=dict(command=cmd,exit_code=p.returncode,result_sha256=hashlib.sha256((out/name/'result.json').read_bytes()).hexdigest(),mesh_sha256=r['runs'][0]['mesh_sha256'])
    if name=='brute':a=r['runs'][0]['independent_quality']
    else:b=r['runs'][0]['independent_quality']
assert report['runs']['brute']['mesh_sha256']==report['runs']['native']['mesh_sha256']
def compare(a,b,path=''):
    if isinstance(a,dict):
        for key,value in a.items():
            if key not in ['scope','distance_oracle']:compare(value,b[key],path+'/'+key)
    elif isinstance(a,(int,float)):
        assert abs(a-b)<=1e-12*max(1.,abs(a),abs(b)),(path,a,b)
    else:assert a==b,(path,a,b)
compare(a,b)
assert b['distance_oracle']['binary_sha256']==report['source_sha256']['surface-distance']
report.update(complete=True,mesh_equal=True,independent_quality_agrees=True)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('Optional native distance oracle preserves the functional mesh and agrees with the original independent metrics.')
