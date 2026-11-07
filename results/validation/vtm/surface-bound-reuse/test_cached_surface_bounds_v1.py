"""Private exact box-cache complete-nearest regression probes."""
import numpy as np
import pytest
import warp as wp
from warpower import SurfaceDomain
from warpower._surface_bvh import SurfaceTree,Closest,box_distance
from tests.test_domain import cube,torus
from examples.vtm_inputs import surface,counterbored_plate
from surface_box_filter_v1 import prepare,interval,pruned,nearer
import cached_surface_bounds_v1 as nearest

wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.kernel
def box_values(tree:SurfaceTree,queries:wp.array[wp.vec3d],values:wp.array[wp.float64]):
    i=wp.tid();values[i]=box_distance(tree,i,queries[i])


@wp.kernel
def box_checks(tree:SurfaceTree,queries:wp.array[wp.vec3d],thresholds:wp.array[wp.float64],checks:wp.array2d[int]):
    i=wp.tid();p=queries[i];prepared=prepare(p);value=interval(tree,i,prepared)
    original=box_distance(tree,i,p)
    checks[i,0]=value.supported
    checks[i,1]=int(value.supported!=0 and not (value.lower<=original and original<=value.upper))
    checks[i,2]=int(pruned(tree,i,p,prepared,thresholds[i]) != (original>=thresholds[i]))
    checks[i,3]=int(nearer(tree,i,i^1,p,prepared) != (original<box_distance(tree,i^1,p)))


@pytest.mark.parametrize('fuse',[False,True])
@pytest.mark.parametrize('exponent',[-60,0,60])
@pytest.mark.parametrize('case',['cube','torus','plate','thin-hand','cavity','disconnected'])
def test_complete_nearest_queries(case,exponent,fuse):
    wp.set_module_options({'fuse_fp':fuse},module=nearest)
    p,f=cube() if case=='cube' else torus(16,8) if case=='torus' else counterbored_plate(16) if case=='plate' else surface(case)
    p=(p*np.float32(2.0**exponent)).astype(np.float32)
    domain=SurfaceDomain(p,f,surface_subdivisions=1,crease_subdivisions=1)
    assert domain.error.numpy()[0]==0
    rng=np.random.default_rng(918);lo,hi=p.min(axis=0).astype(float),p.max(axis=0).astype(float)
    ordinary=rng.uniform(lo-.1*(hi-lo),hi+.1*(hi-lo),(100,3));tri=p[f].astype(float);centers=tri.mean(axis=1)
    q=np.concatenate([ordinary,p,tri[:,:2].mean(axis=1),centers,np.nextafter(centers,np.inf),np.nextafter(centers,-np.inf)])
    points=wp.array(q,dtype=wp.vec3d);a=wp.empty(len(q),dtype=Closest);b=wp.empty_like(a)
    full=np.full(len(q),1e150)
    limits=wp.array(full,dtype=wp.float64)
    wp.launch(nearest.reference,len(q),inputs=[domain.tree,points,limits,a],block_dim=32)
    distances=np.sqrt(a.numpy()['distance2'])
    for bounds in [full,np.zeros(len(q)),distances,np.nextafter(distances,0),np.nextafter(distances,np.inf),np.full(len(q),1e-150)]:
        wp.copy(limits,wp.array(bounds,dtype=wp.float64))
        for kernel,out in [(nearest.reference,a),(nearest.candidate,b)]:
            wp.launch(kernel,len(q),inputs=[domain.tree,points,limits,out],block_dim=32)
        x,y=a.numpy(),b.numpy()
        for field in ['point','distance2','face']:np.testing.assert_array_equal(x[field],y[field])
