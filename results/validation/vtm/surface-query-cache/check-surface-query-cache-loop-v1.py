"""Captured query reuse across deformation, cutoff changes, reorder, invalid input and recovery."""
import hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from warpower import SurfaceDomain,_extraction
from warpower._surface_bvh import SurfaceTree,material_side,closest_point
from tests.test_domain import cube
import surface_query_cache_kernels_v1 as k
from surface_query_cache_v1 import Cache,install

wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def produce(phase:wp.array[int],surfaces:wp.array2d[wp.vec3],queries:wp.array2d[wp.vec3d],all_limits:wp.array2d[wp.float64],all_counts:wp.array[int],all_refit:wp.array[int],surface:wp.array[wp.vec3],centers:wp.array[wp.vec3d],limits:wp.array[wp.float64],count:wp.array[int],refit:wp.array[int],stats:wp.array[int]):
    i=wp.tid();p=phase[0]
    if i<surface.shape[0]:surface[i]=surfaces[p,i]
    if i<centers.shape[0]:centers[i]=queries[p,i];limits[i]=all_limits[p,i]
    if i==0:count[0]=all_counts[p];refit[0]=all_refit[p]
    if i<stats.shape[0]:stats[i]=0

@wp.kernel
def query(tree:SurfaceTree,centers:wp.array[wp.vec3d],limits:wp.array[wp.float64],count:wp.array[int],error:wp.array[int],current:k.QueryState,previous:k.QueryState,table:wp.array[int],stats:wp.array[int]):
    i=wp.tid();current.valid[i]=0;current.rays[i]=0;current.nearest[i]=0
    if error[0]!=0 or i>=count[0]:return
    center=centers[i];limit=limits[i];j=k.lookup(center,previous,table)
    if j>=0:
        if previous.rays[j]!=0:wp.atomic_add(stats,0,1)
        if previous.nearest[j]!=0 and wp.cast(limit,wp.uint64)==wp.cast(previous.limits[j],wp.uint64):wp.atomic_add(stats,1,1)
    side=material_side(tree,center);cached=k.cached_side(tree,center,previous,table)
    hit=closest_point(tree,center,limit);other=k.cached_nearest(tree,center,limit,previous,table)
    equal=side==cached and hit.face==other.face and wp.cast(hit.distance2,wp.uint64)==wp.cast(other.distance2,wp.uint64) and k.same_center(hit.point,other.point)
    if not equal:wp.atomic_add(stats,2,1)
    current.valid[i]=1;current.centers[i]=center;current.rays[i]=1;current.sides[i]=cached
    current.nearest[i]=1;current.limits[i]=limit;current.faces[i]=other.face;current.distances[i]=other.distance2;current.hit_points[i]=other.point

@wp.kernel
def consume(phase:wp.array[int],condition:wp.array[int],error:wp.array[int],count:wp.array[int],stats:wp.array[int],rows:wp.array2d[int]):
    p=phase[0];rows[p,0]=error[0];rows[p,1]=count[0]
    for j in range(3):rows[p,j+2]=stats[j]
    phase[0]=p+1;condition[0]=int(p+1<rows.shape[0])

install();surface,faces=cube();domain=SurfaceDomain(surface,faces)
n=64;phases=13
points=np.random.default_rng(218).uniform(-3,3,(n,3)).astype(np.float64)
points[:7]=[[1.5,0,0],[1,0,0],[1,1,0],[1,1,1],[0,0,0],[-0.,0,0],[3,3,3]]
surfaces=np.repeat(surface[None],phases,axis=0);surfaces[2:5]*=2;surfaces[4,0]=np.nan
queries=np.stack([np.roll(points,i%3,axis=0) for i in range(phases)])
limits=np.full((phases,n),1e150,np.float64);limits[8]=0.1
counts=np.full(phases,n,np.int32);counts[6]=0;counts[11]=n+1
refits=np.array([1,0,1,0,1,1,0,0,0,0,0,0,0],np.int32)
cache=Cache(domain,n,'captured-query-loop')
surface_data=wp.array(surfaces,dtype=wp.vec3);query_data=wp.array(queries,dtype=wp.vec3d);limit_data=wp.array(limits,dtype=wp.float64)
count_data=wp.array(counts,dtype=int);refit_data=wp.array(refits,dtype=int)
current_points=wp.empty(n,dtype=wp.vec3d);current_limits=wp.empty(n,dtype=wp.float64)
phase=wp.zeros(1,dtype=int);condition=wp.ones(1,dtype=int);count=wp.zeros(1,dtype=int);refit=wp.zeros(1,dtype=int);error=wp.zeros(1,dtype=int);source_error=wp.zeros(1,dtype=int);stats=wp.zeros(3,dtype=int);rows=wp.zeros((phases,5),dtype=int)
wp.load_module(module=__name__,device=domain.device)
def step():
    wp.launch(produce,max(n,len(surface)),inputs=[phase,surface_data,query_data,limit_data,count_data,refit_data,domain.points,current_points,current_limits,count,refit,stats])
    wp.capture_if(refit,domain.refit)
    wp.launch(_extraction.input_ready,1,inputs=[domain.error,source_error,count,n,error])
    wp.launch(query,n,inputs=[domain.tree,current_points,current_limits,count,error,cache.current,cache.previous,cache.table,stats],block_dim=32)
    cache.after(error)
    wp.launch(consume,1,inputs=[phase,condition,error,count,stats,rows])
with wp.ScopedCapture() as capture:
    phase.zero_();condition.fill_(1);wp.capture_while(condition,step)
results=[]
for _ in range(2):
    wp.capture_launch(capture.graph);r=rows.numpy();results.append(r.tolist())
    np.testing.assert_array_equal(r[:,4],0)
    assert r[4,0]!=0 and r[11,0]==2
    np.testing.assert_array_equal(np.delete(r[:,0],[4,11]),0)
    for index in [0,2,4,5,6,7,11,12]:np.testing.assert_array_equal(r[index,2:4],0)
    for index in [1,3,10]:np.testing.assert_array_equal(r[index,2:4],[n,n])
    for index in [8,9]:np.testing.assert_array_equal(r[index,2:4],[n,0])
np.testing.assert_array_equal(results[0],results[1])
out=Path('.work/vtm/surface-query-cache-loop-check-v1.json');assert not out.exists()
out.write_text(json.dumps(dict(scope=__doc__,columns=['error','active_count','cached_ray_hits','cached_nearest_hits','bitwise_query_mismatches'],replays=results,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/surface_query_cache_v1.py'),Path('.work/vtm/surface_query_cache_kernels_v1.py')]},indent=2)+'\n')
print(json.dumps(results),flush=True)
