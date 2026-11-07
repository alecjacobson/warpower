"""Private #164 incumbent-box pruning with the original strided scan."""
import warp as wp
from warpower import _filtered_predicates as fp, _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, sphere, symbolic_sphere
from warpower._mesh_topology import face_vertices
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def search_linear_box(
 points:wp.array[wp.vec3],tree:Tree,tets:wp.array2d[int],faces:wp.array[int],
 count:wp.array[int],candidates:wp.array[int],workers:int,error:wp.array[int],
 supported:wp.array[int],statistics:wp.array[int],
):
 tid=wp.tid();i,lane=tid//workers,tid%workers
 candidates[tid]=-1
 if i>=count[0]:return
 if lane==0:wp.atomic_add(statistics,wp.where(supported[i]!=0,0,1),1)
 if supported[i]!=0:return
 face=face_vertices(tets,faces[i])
 a,b,c=points[face[0]],points[face[1]],points[face[2]]
 best=int(-1);box=sb.Box()
 for q in range(lane,points.shape[0],workers):
  if tree.owners[q]!=q or q==face[0] or q==face[1] or q==face[2]:continue
  p=points[q]
  # A strict exterior witness cannot beat the incumbent in the exact lifted
  # ordering. Uncertain/invalid bounds and every physical tie retain the scan.
  if best>=0:
   if sb.outside(box,p,p):continue
  orientation=fp.orient3d(a,b,c,p)
  if orientation.valid==0:
   wp.atomic_or(error,0,2);return
  if orientation.sign<=0:continue
  replace=best<0
  if best>=0:
   result=symbolic_sphere(points,wp.vec4i(face[0],face[1],face[2],best),q)
   if result.valid==0 or result.sign==0:
    wp.atomic_or(error,0,2);return
   replace=result.sign>0
  if replace:
   best=q
   shape=sphere(a,b,c,p)
   box=sb.bounds(shape.origin,shape.nx,shape.ny,shape.nz,shape.determinant)
 candidates[tid]=best
