"""Private #175 probe: ordered strict scan and exact fallback by query lane."""
import warp as wp
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._prepared_seeds import PreparedSeedQueries, point_value, prepare

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def strict_check(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int],
    tree:Tree, live:wp.array[int], error:wp.array[int],
    shapes:wp.array[Sphere], boxes:wp.array[sb.Box], pending:wp.array[wp.uint32],
):
    tid=wp.tid()
    i,lane=tid//32,tid%32
    if error[0]!=0 or i>=count[0]:
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=shapes[i]
    box=boxes[i]
    uncertain=int(0)
    for q in range(lane,points.shape[0],32):
        if tree.owners[q]!=q or q==ids[0] or q==ids[1] or q==ids[2] or q==ids[3]:
            continue
        p=points[q]
        if sb.outside(box,p,p):
            continue
        value=point_value(shape,p)
        if wp.isfinite(value.value) and wp.isfinite(value.error) and wp.abs(value.value)>value.error:
            if value.value>wp.float64(0.0):
                wp.atomic_and(live,i,0)
        else:
            uncertain=1
    if uncertain!=0:
        wp.atomic_or(pending,i,wp.uint32(1)<<wp.uint32(lane))

@wp.kernel
def exact_check(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int],
    tree:Tree, live:wp.array[int], error:wp.array[int],
    shapes:wp.array[Sphere], boxes:wp.array[sb.Box], pending:wp.array[wp.uint32],
):
    tid=wp.tid()
    i,lane=tid//32,tid%32
    if error[0]!=0 or i>=count[0]:
        return
    if (pending[i] & (wp.uint32(1)<<wp.uint32(lane)))==wp.uint32(0):
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=shapes[i]
    box=boxes[i]
    # Recompute this complete original lane, even when a strict query already
    # rejected the cell. Its uncertain signs still require exact diagnostics.
    for q in range(lane,points.shape[0],32):
        if tree.owners[q]!=q or q==ids[0] or q==ids[1] or q==ids[2] or q==ids[3]:
            continue
        p=points[q]
        if sb.outside(box,p,p):
            continue
        value=point_value(shape,p)
        if wp.isfinite(value.value) and wp.isfinite(value.error) and wp.abs(value.value)>value.error:
            if value.value>wp.float64(0.0):
                wp.atomic_and(live,i,0)
            continue
        result=symbolic_sphere(points,ids,q)
        if result.valid==0:
            wp.atomic_or(error,0,2)
            wp.atomic_and(live,i,0)
        elif result.sign>=0:
            wp.atomic_and(live,i,0)

class SplitPreparedSeedQueries(PreparedSeedQueries):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pending=wp.zeros(self.capacity,dtype=wp.uint32,device=self.device)
        self._split_inputs=[*self._check_inputs,self.pending]
        wp.load_module(module=__name__,device=self.device,block_dim=self.block_dim)

    def run(self):
        self.pending.zero_()
        wp.launch(prepare,self.capacity,inputs=self._prepare_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(strict_check,32*self.capacity,inputs=self._split_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(exact_check,32*self.capacity,inputs=self._split_inputs,block_dim=self.block_dim,device=self.device)

def install():
    import warpower._prepared_seeds as module
    module.PreparedSeedQueries=SplitPreparedSeedQueries
