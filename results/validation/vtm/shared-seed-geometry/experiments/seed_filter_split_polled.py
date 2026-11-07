"""Private bounds-only prepass experiment; exact decisions are unchanged."""
import warp as wp

from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import outside, sphere, symbolic_sphere
from warpower._facet_seeds import CertifiedSeeds, check_seeds

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def bounds_only(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int], tree:Tree,
    live:wp.array[int], error:wp.array[int], pending:wp.array[int],
):
    tid=wp.tid()
    i,lane=tid//32,tid%32
    if error[0]!=0 or i>=count[0]:
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=sphere(points[ids[0]],points[ids[1]],points[ids[2]],points[ids[3]])
    box=sb.bounds(shape.origin,shape.nx,shape.ny,shape.nz,shape.determinant)
    for q in range(lane,points.shape[0],32):
        # Once any lane requires the old checker, other lanes may stop this
        # prepass: the following kernel rescans the entire cell's query set.
        if pending[i]!=0:
            return
        if tree.owners[q]!=q or q==ids[0] or q==ids[1] or q==ids[2] or q==ids[3]:
            continue
        p=points[q]
        if sb.outside(box,p,p) or outside(shape,p,p):
            continue
        wp.atomic_or(pending,i,1)
        return


@wp.kernel
def exact_pending(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int], tree:Tree,
    live:wp.array[int], error:wp.array[int], pending:wp.array[int],
):
    tid=wp.tid()
    i,lane=tid//32,tid%32
    if error[0]!=0 or i>=count[0]:
        return
    # Immutable during this kernel: a sibling lane rejecting the cell must
    # not suppress another lane's existing predicate/error handling.
    if pending[i]==0:
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=sphere(points[ids[0]],points[ids[1]],points[ids[2]],points[ids[3]])
    box=sb.bounds(shape.origin,shape.nx,shape.ny,shape.nz,shape.determinant)
    for q in range(lane,points.shape[0],32):
        if tree.owners[q]!=q or q==ids[0] or q==ids[1] or q==ids[2] or q==ids[3]:
            continue
        p=points[q]
        if sb.outside(box,p,p) or outside(shape,p,p):
            continue
        result=symbolic_sphere(points,ids,q)
        if result.valid==0:
            wp.atomic_or(error,0,2)
            wp.atomic_and(live,i,0)
        elif result.sign>=0:
            wp.atomic_and(live,i,0)


class SplitSeeds(CertifiedSeeds):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pending=wp.zeros(len(self.source),dtype=int,device=self.device)
        wp.load_module(module=__name__,device=self.device,block_dim=self._block_dim)

    def run(self):
        original=wp.launch

        def launch(kernel,*args,**kwargs):
            if kernel is check_seeds:
                self.pending.zero_()
                kwargs=dict(kwargs,inputs=kwargs['inputs']+[self.pending])
                original(bounds_only,*args,**kwargs)
                return original(exact_pending,*args,**kwargs)
            return original(kernel,*args,**kwargs)

        wp.launch=launch
        try:
            super().run()
        finally:
            wp.launch=original
