"""Independent final uniform8K mesh from the native full-solve trace."""
import json,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
w=Path('.work/vtm');dest=w/'tangential-production-native-full-v1-independent.json';assert not dest.exists()
description=__doc__
s=(w/'check-tangential-production-final-v1.py').read_text().split('report=dict(scope=')[0]
# Load the same frozen geometry validator and independent boundary helper only.
s=s.replace("dest=w/'tangential-production-final-v1-independent.json';assert not dest.exists()", "unused_destination=w/'tangential-production-final-v1-independent.json'")
exec(compile(s,str(w/'check-tangential-production-final-v1.py'),'exec'))
path=w/'tangential-production-uniform-native-v1.npz';d=json.loads(path.with_suffix('.json').read_text())
assert d['work']['error']==[0] and d['work']['valid']==[1]
report=dict(scope=description,native_result_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),independent=independent(path,None))
reference=w/'tangential-production-final-v1/armadillo-uniform8192-recovery-after/mesh.npz'
with np.load(path) as x,np.load(reference) as y:report['timed_mesh_comparison']=compare(x,y)
assert all(report['timed_mesh_comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
assert report['timed_mesh_comparison']['maximum_coordinate_difference']<1e-8
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n');print(report['timed_mesh_comparison'],flush=True)
