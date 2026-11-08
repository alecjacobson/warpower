"""Independent geometry for every successful frame of the device-controlled large-mesh loop."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
dest=w/'extended-reuse-native-loop-independent-v1.json';assert not dest.exists()
run=w/'extended-reuse-loop-native-v1.json';d=json.loads(run.read_text());report=dict(scope=__doc__,run_sha256=hashlib.sha256(run.read_bytes()).hexdigest(),frames={})
assert [r[0] for r in d['metadata']]==[0,0,1,1,0]
assert [r[1] for r in d['metadata']]==[1,1,0,0,1]
assert all(not any(d['metadata'][i][2:]) and not any(d['volumes'][i]) for i in [2,3])
for i in [0,1,4]:
    path=w/f'extended-reuse-loop-native-v1-frame{i}.npz'
    row=validate(path)
    with np.load(path) as m,tempfile.TemporaryDirectory(prefix='extended-loop-boundary-',dir='/opt/warpower-generated-cache') as tmp:
        p=Path(tmp)/'boundary.off'
        with p.open('w') as f:
            f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n');np.savetxt(f,m['points'],fmt='%.17g');np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
        result=subprocess.run([str(w/'surface-reference'),str(p)],text=True,capture_output=True,check=True)
        assert int(result.stdout.splitlines()[0].split()[-1])==0,result.stdout
        row.update(boundary_self_intersection_pairs=0,boundary_reference_stdout=result.stdout)
        row['counts_match_capture']=[len(m[k]) for k in ['points','full_tetrahedra','tetrahedra','boundary']]==d['metadata'][i][2:6]
        assert row['counts_match_capture']
    report['frames'][str(i)]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(i,row['quality']['radius_ratio_percentiles'],flush=True)
with np.load(w/'extended-reuse-loop-native-v1-frame0.npz') as a,np.load(w/'extended-reuse-loop-native-v1-frame4.npz') as b:
    report['recovery_comparison']=compare(a,b)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(report['recovery_comparison'],flush=True)
