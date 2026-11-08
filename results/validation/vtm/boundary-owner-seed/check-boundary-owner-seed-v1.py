"""Original/seeded nearest-site answers for stale hints, active prefixes, ties and scales."""
import hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from warpower import _bvh
from warpower._site_bvh import SiteHierarchy,nearest_site
from boundary_owner_seed_kernels_v1 import nearest_seeded
wp.set_module_options({"enable_backward":False,"fast_math":False})
@wp.kernel
def reference(tree:_bvh.Tree,points:wp.array[wp.vec3],queries:wp.array[wp.vec3d],active:wp.array[int],occupied:wp.array[int],only_free:bool,seeds:wp.array[int],indices:wp.array[int],distances:wp.array[wp.float64]):
 i=wp.tid();r=nearest_site(tree,points,queries[i],active,occupied,only_free);indices[i]=r.index;distances[i]=r.distance2
@wp.kernel
def seeded(tree:_bvh.Tree,points:wp.array[wp.vec3],queries:wp.array[wp.vec3d],active:wp.array[int],occupied:wp.array[int],only_free:bool,seeds:wp.array[int],indices:wp.array[int],distances:wp.array[wp.float64]):
 i=wp.tid();r=nearest_seeded(tree,points,queries[i],active,occupied,only_free,seeds[i]);indices[i]=r.index;distances[i]=r.distance2
out=Path('.work/vtm/boundary-owner-seed-query-check-v1.json');assert not out.exists();rng=np.random.default_rng(221);rows=[]
for kind in ['random','ties-duplicates']:
 for scale in [1e-30,1.0,1e30]:
  points=rng.normal(size=(65,3)).astype(np.float32)
  if kind=='ties-duplicates':
   points[:]=np.tile(np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,0,0]],np.float32),(10,1))[:65]
  points=(points*scale).astype(np.float32);p=wp.array(points,dtype=wp.vec3);hierarchy=SiteHierarchy(p)
  q=(rng.normal(size=(200,3))*scale).astype(np.float32).astype(np.float64)
  q[:12]=np.array([[0,0,0],[-0.,0,0],[1e150,0,0],[1e200,0,0],[np.nan,0,0],[np.inf,0,0],[0,-np.inf,0],*points[:5].astype(float).tolist()],np.float64)
  queries=wp.array(q,dtype=wp.vec3d);ids=wp.array(rng.integers(-3,len(points)+5,len(q),dtype=np.int32),dtype=int)
  occupied=wp.array(np.where(np.arange(len(points))%2==0,0,-1).astype(np.int32),dtype=int)
  for count in [0,1,17,65,70]:
   active=wp.full(1,count,dtype=int)
   for free in [False,True]:
    outputs=[]
    for kernel in [reference,seeded]:
     index=wp.empty(len(q),dtype=int);distance=wp.empty(len(q),dtype=wp.float64)
     wp.launch(kernel,len(q),inputs=[hierarchy.tree,p,queries,active,occupied,free,ids,index,distance],block_dim=32)
     outputs.append((index.numpy(),distance.numpy()))
    np.testing.assert_array_equal(outputs[0][0],outputs[1][0]);np.testing.assert_array_equal(outputs[0][1].view(np.uint64),outputs[1][1].view(np.uint64))
    if kind=='ties-duplicates' and count>0:assert outputs[0][0][0]==(1 if free and count>1 else -1 if free else 0)
    rows.append(dict(kind=kind,scale=scale,active=count,only_free=free,queries=len(q),owner_ids_equal=True,distance_bits_equal=True))
    out.write_text(json.dumps(dict(scope=__doc__,cases=rows,complete=False),indent=2)+'\n')
r=dict(scope=__doc__,cases=rows,complete=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/boundary_owner_seed_kernels_v1.py'),Path('warpower/_site_bvh.py'),Path('warpower/_boundary.py')]});out.write_text(json.dumps(r,indent=2)+'\n');print(len(rows),'cases',sum(x['queries'] for x in rows),'queries match exactly',flush=True)
