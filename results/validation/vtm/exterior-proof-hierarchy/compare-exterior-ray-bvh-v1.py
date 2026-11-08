"""Compare geometry/work to independently validated private v2, not clean timing."""
import json,numpy as np
from pathlib import Path
w=Path('.work/vtm');out=[]
def canon(t):
    s=np.sort(t,axis=1);return s[np.lexsort(s.T[::-1])]
for name in ['graded8192','uniform8192','graded4096']:
    b=w/'tangential-topology-pilot-v2'/name;a=w/'tangential-ray-bvh-pilot-v1'/name
    row={'case':name,'before_ms':json.loads((b/'result.json').read_text())['first_replay_gpu_ms'],'after_ms':json.loads((a/'result.json').read_text())['first_replay_gpu_ms']}
    with np.load(a/'mesh.npz') as x,np.load(b/'mesh.npz') as y:
        row['points_bitwise']=np.array_equal(x['points'].view(np.uint32),y['points'].view(np.uint32));row['roles']=np.array_equal(x['roles'],y['roles'])
        row['canonical']={k:np.array_equal(canon(x[k]),canon(y[k])) for k in ['full_tetrahedra','tetrahedra','boundary']}
    da=json.loads((a/'result.json').read_text());db=json.loads((b/'result.json').read_text());row['work']={k:da[k]==db[k] for k in ['counts','rounds','steps','moves','topology','quality_rejections']}
    assert row['points_bitwise'] and row['roles'] and all(row['canonical'].values()) and all(row['work'].values()),row
    out.append(row)
p=w/'exterior-ray-bvh-pilot-v1-comparison.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,default=bool)+'\n');print(json.dumps(out,default=bool))
