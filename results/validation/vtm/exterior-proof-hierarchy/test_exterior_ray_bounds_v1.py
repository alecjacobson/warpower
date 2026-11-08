"""Fraction oracle for rational-centroid box pruning, plus disjoint coplanarity."""
from fractions import Fraction as F
import numpy as np
import warp as wp
from warpower._surface_bvh import SurfaceHierarchy,SurfaceTree
from warpower._exterior_proof import centroid_ray as exhaustive
from exterior_ray_bvh_v1 import centroid_box,centroid_ray

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def boxes(lo:wp.array[wp.vec3],hi:wp.array[wp.vec3],p:wp.array2d[wp.vec3],d:wp.array[wp.vec3],out:wp.array[int]):
    i=wp.tid();out[i]=int(centroid_box(lo[i],hi[i],p[i,0],p[i,1],p[i,2],p[i,3],d[i]))

def intersects(lo,hi,p,d):
    origin=[sum(F(float(v)) for v in p[:,k])/4 for k in range(3)]
    enter=[F(0)];leave=[]
    for k in range(3):
        a=(F(float(lo[k]))-origin[k])/F(float(d[k]))
        b=(F(float(hi[k]))-origin[k])/F(float(d[k]))
        enter.append(min(a,b));leave.append(max(a,b))
    return max(enter)<=min(leave)

def test_box_pruning_never_discards_exact_rational_intersection():
    rng=np.random.default_rng(239);data=[]
    dirs=np.array([[11,23,37],[-17,29,43],[31,-47,19]],np.float32)
    for exponent in [-140,-100,-20,0,20,100,125]:
        scale=np.float32(2.**exponent)
        for i in range(192):
            p=rng.uniform(-2,2,(4,3)).astype(np.float32)*scale
            a=rng.uniform(-3,3,3).astype(np.float32)*scale
            b=rng.uniform(-3,3,3).astype(np.float32)*scale
            if i%3==0:
                center=p.astype(np.float64).sum(axis=0)/4
                a=np.nextafter(center.astype(np.float32),np.float32(-np.inf))
                b=np.nextafter(center.astype(np.float32),np.float32(np.inf))
            data.append((np.minimum(a,b),np.maximum(a,b),p,dirs[i%3]))
    lo,hi,p,d=(np.stack([row[k] for row in data]) for k in range(4))
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(p).all()
    args=[wp.array(v,dtype=wp.vec3,device='cuda:0') for v in [lo,hi,p,d]]
    result=wp.empty(len(data),dtype=int,device='cuda:0');wp.load_module(module=__name__,device='cuda:0')
    with wp.ScopedCapture() as cap:wp.launch(boxes,len(data),inputs=args+[result])
    wp.capture_launch(cap.graph);actual=result.numpy()
    exact=np.array([intersects(*row) for row in data])
    assert exact.sum()>400 and (actual==0).sum()>100
    assert np.all(actual[exact]==1), np.flatnonzero(exact & (actual==0))

@wp.kernel
def ray_pair(tree:SurfaceTree,p:wp.array[wp.vec3],out:wp.array[wp.vec2i]):
    d=wp.vec3(11.,23.,37.)
    out[0]=exhaustive(tree,p[0],p[1],p[2],p[3],d)
    out[1]=centroid_ray(tree,p[0],p[1],p[2],p[3],d)

def test_disjoint_coplanar_triangle_is_safely_pruned():
    direction=np.array([11,23,37],np.float32);edge=np.array([23,-11,0],np.float32)
    v=np.array([1000*edge,1001*edge,1000*edge+direction],np.float32)
    cells=np.array([[-1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1]],np.float32)
    assert not intersects(v.min(0),v.max(0),cells,direction)
    points=wp.array(v,dtype=wp.vec3,device='cuda:0');faces=wp.array([0,1,2],dtype=int,device='cuda:0')
    h=SurfaceHierarchy(points,faces);p=wp.array(cells,dtype=wp.vec3,device='cuda:0')
    out=wp.empty(2,dtype=wp.vec2i,device='cuda:0')
    wp.launch(ray_pair,1,inputs=[h.tree,p,out]);actual=out.numpy()
    # Exhaustive code conservatively reports an infinite-plane ambiguity. The
    # exact AABB proof establishes that this triangle cannot meet the forward ray.
    np.testing.assert_array_equal(actual,[[0,1],[0,0]])
