"""Private #192 probe: preserve error checks using a finite-distance witness.

Only internal selective recovery recording substitutes this kernel. Full public
extraction and quality-move/mesher diagnostics keep the original implementation.
The private selection extraction's unused distance summaries become zero (its
ordinary reset still runs); no public diagnostic API is changed by this probe.
"""
import warp as wp
from warpower import _extraction,_selective
from warpower._surface_bvh import SurfaceTree,closest_point,closest_triangle
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def surface_status(tree:SurfaceTree,points:wp.array[wp.vec3],faces:wp.array2d[int],count:wp.array[int],integral:wp.array[wp.float64],maximum:wp.array[wp.float64],error:wp.array[int]):
    i=wp.tid()
    if i>=count[0] or error[0]!=0:return
    a=wp.vec3d(points[faces[i,0]]);b=wp.vec3d(points[faces[i,1]]);c=wp.vec3d(points[faces[i,2]])
    for k in range(3):
        dominant=a
        if k==1:dominant=b
        elif k==2:dominant=c
        sample=(a+b+c)/wp.float64(6.0)+dominant/wp.float64(2.0)
        proved=False
        if tree.n>0:
            f=tree.ids[0]
            x=wp.vec3d(tree.points[tree.indices[3*f]])
            y=wp.vec3d(tree.points[tree.indices[3*f+1]])
            z=wp.vec3d(tree.points[tree.indices[3*f+2]])
            q=closest_triangle(sample,x,y,z)
            distance=wp.length_sq(sample-q)
            limit=wp.float64(1.0e150)
            proved=wp.isfinite(distance) and distance<limit*limit
        if not proved:
            # Keep the complete old query when the witness is unavailable.
            hit=closest_point(tree,sample,wp.float64(1.0e150))
            if hit.face<0 or not wp.isfinite(hit.distance2):wp.atomic_or(error,0,16)

def install():
    wp.load_module(module=__name__,device='cuda:0')
    original=_selective.SelectiveUpdate._evaluate_recovered
    def selection(self):
        prior=_extraction.surface_error
        _extraction.surface_error=surface_status
        try:return original(self)
        finally:_extraction.surface_error=prior
    _selective.SelectiveUpdate._evaluate_recovered=selection
