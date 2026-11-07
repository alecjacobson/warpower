"""Private alternative: prepare current-input sphere bounds once per cell."""
import warp as wp

from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, sphere, symbolic_sphere
from warpower._facet_seeds import CertifiedSeeds, check_seeds

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def prepare_bounds(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int], error:wp.array[int],
    shapes:wp.array[Sphere], boxes:wp.array[sb.Box],
):
    i=wp.tid()
    if error[0]!=0 or i>=count[0]:
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=sphere(points[ids[0]],points[ids[1]],points[ids[2]],points[ids[3]])
    shapes[i]=shape
    boxes[i]=sb.bounds(shape.origin,shape.nx,shape.ny,shape.nz,shape.determinant)


@wp.kernel
def check_prepared(
    points:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int], tree:Tree,
    live:wp.array[int], error:wp.array[int], shapes:wp.array[Sphere], boxes:wp.array[sb.Box],
):
    tid=wp.tid()
    i,lane=tid//32,tid%32
    if error[0]!=0 or i>=count[0]:
        return
    ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
    shape=shapes[i]
    box=boxes[i]
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


class PreparedSeeds(CertifiedSeeds):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shapes=wp.empty(len(self.source),dtype=Sphere,device=self.device)
        self.boxes=wp.empty(len(self.source),dtype=sb.Box,device=self.device)
        wp.load_module(module=__name__,device=self.device,block_dim=self._block_dim)

    def run(self):
        original=wp.launch

        def launch(kernel,*args,**kwargs):
            if kernel is check_seeds:
                p,t,n,tree,live,error=kwargs['inputs']
                original(prepare_bounds,len(self.unique),inputs=[p,t,n,error,self.shapes,self.boxes],block_dim=self._block_dim,device=self.device)
                kwargs=dict(kwargs,inputs=kwargs['inputs']+[self.shapes,self.boxes])
                return original(check_prepared,*args,**kwargs)
            return original(kernel,*args,**kwargs)

        wp.launch=launch
        try:
            super().run()
        finally:
            wp.launch=original
