"""Private conservative bounded-range filter for scalar box comparisons."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, box_distance
wp.set_module_options({'enable_backward':False,'fast_math':False})
B_MIN=wp.constant(wp.float32(2.0**-30))
ERROR_FACTOR=wp.constant(wp.float32(2.0**-16))


@wp.struct
class PreparedPoint:
    point: wp.vec3
    magnitude: float
    supported: int


@wp.struct
class Interval:
    lower: wp.float64
    upper: wp.float64
    supported: int


@wp.func(inline=True)
def prepare(p:wp.vec3d):
    out=PreparedPoint()
    if (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
        and wp.abs(p[0])<=wp.float64(16.0) and wp.abs(p[1])<=wp.float64(16.0)
        and wp.abs(p[2])<=wp.float64(16.0)):
        out.point=wp.vec3(p)
        a=wp.abs(out.point)
        out.magnitude=wp.max(wp.max(a[0],wp.max(a[1],a[2])),B_MIN)
        out.supported=1
    return out


@wp.func(inline=True)
def interval(tree:SurfaceTree,node:int,p:PreparedPoint):
    result=Interval()
    lo=tree.lower[node]
    hi=tree.upper[node]
    if (p.supported==0 or not wp.isfinite(lo[0]) or not wp.isfinite(lo[1])
        or not wp.isfinite(lo[2]) or not wp.isfinite(hi[0])
        or not wp.isfinite(hi[1]) or not wp.isfinite(hi[2])):
        return result
    magnitude=wp.max(wp.abs(lo),wp.abs(hi))
    bound=wp.max(p.magnitude,wp.max(magnitude[0],wp.max(magnitude[1],magnitude[2])))
    if bound>16.0:
        return result
    d=wp.max(wp.max(lo-p.point,p.point-hi),wp.vec3(0.0))
    value=wp.dot(d,d)
    error=(bound*bound)*ERROR_FACTOR
    result.lower=wp.float64(value)-wp.float64(error)
    result.upper=wp.float64(value)+wp.float64(error)
    result.supported=1
    return result


@wp.func(inline=True)
def pruned(tree:SurfaceTree,node:int,p:wp.vec3d,prepared:PreparedPoint,threshold:wp.float64):
    value=interval(tree,node,prepared)
    if value.supported!=0:
        if value.lower>=threshold:
            return True
        if value.upper<threshold:
            return False
    return box_distance(tree,node,p)>=threshold


@wp.func(inline=True)
def nearer(tree:SurfaceTree,left:int,right:int,p:wp.vec3d,prepared:PreparedPoint):
    a=interval(tree,left,prepared)
    b=interval(tree,right,prepared)
    if a.supported!=0 and b.supported!=0:
        if a.upper<b.lower:
            return True
        if a.lower>=b.upper:
            return False
    return box_distance(tree,left,p)<box_distance(tree,right,p)
