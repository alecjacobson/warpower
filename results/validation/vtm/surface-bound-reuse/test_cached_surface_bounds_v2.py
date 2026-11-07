"""Private exact box-cache complete-nearest regression probes."""
import numpy as np
import pytest
import warp as wp
from warpower import SurfaceDomain
from warpower._surface_bvh import SurfaceTree,Closest,box_distance
from tests.test_domain import cube,torus
from examples.vtm_inputs import surface,counterbored_plate
import cached_surface_bounds_v2 as nearest

wp.set_module_options({'enable_backward':False,'fast_math':False})


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


@pytest.mark.parametrize('fuse',[False,True])
@pytest.mark.parametrize('mode',['ordinary','nonfinite-points','nonfinite-root','nonfinite-child','empty'])
def test_extreme_centers_limits_and_nonfinite_bounds(fuse,mode):
    wp.set_module_options({'fuse_fp':fuse},module=nearest)
    p,f=cube();domain=SurfaceDomain(p,f,surface_subdivisions=1,crease_subdivisions=1)
    if mode=='nonfinite-points':
        wp.copy(domain.tree.points,wp.full(len(domain.tree.points),wp.vec3(wp.nan),dtype=wp.vec3))
    elif mode in ['nonfinite-root','nonfinite-child']:
        lower=domain.tree.lower.numpy();upper=domain.tree.upper.numpy()
        node=1 if mode=='nonfinite-root' else domain.tree.leaf_base
        lower[node]=[np.nan,-np.inf,np.finfo(np.float32).max]
        upper[node]=[np.inf,np.nan,-np.finfo(np.float32).max]
        wp.copy(domain.tree.lower,wp.array(lower,dtype=wp.vec3));wp.copy(domain.tree.upper,wp.array(upper,dtype=wp.vec3))
    elif mode=='empty':domain.tree.n=0
    centers=np.array([[0,0,0],[1,0,0],[1,1,0],[1,1,1],[-1,-1,-1],[1+1e-13,1,1],
                      [1e-300,0,0],[1e140,0,0],[1e150,0,0],[1e160,0,0],
                      [np.finfo(float).max,0,0],[np.inf,0,0],[-np.inf,0,0],[np.nan,0,0]],float)
    limits=np.array([0.,np.nextafter(0.,1.),1e-150,1e-100,.01,1.,2.,1e150,1e160,np.inf,np.nan,-1.])
    q=np.repeat(centers,len(limits),axis=0);bounds=np.tile(limits,len(centers))
    points=wp.array(q,dtype=wp.vec3d);search=wp.array(bounds,dtype=wp.float64)
    a=wp.empty(len(q),dtype=Closest);b=wp.empty_like(a)
    for kernel,out in [(nearest.reference,a),(nearest.candidate,b)]:
        wp.launch(kernel,len(q),inputs=[domain.tree,points,search,out],block_dim=32)
    x,y=a.numpy(),b.numpy()
    for field in ['point','distance2','face']:np.testing.assert_array_equal(x[field],y[field])
