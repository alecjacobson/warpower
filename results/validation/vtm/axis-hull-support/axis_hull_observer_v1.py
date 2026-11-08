"""Count exact coordinate-plane opportunities before unchanged hull checking."""
import warp as wp
from warpower._mesh_topology import face_vertices
from warpower._bvh import Tree
from warpower import _proved_completion
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
instances=[]
COLUMNS=['calls','faces','axis_faces','root_supported','nonzero_axis_faces','nonzero_supported','x_faces','y_faces','z_faces','axis_bound_not_proved']

@wp.kernel
def observe(points:wp.array[wp.vec3],tets:wp.array2d[int],boundary:wp.array[int],count:wp.array[int],tree:Tree,stats:wp.array[int]):
 slot=wp.tid()
 if slot==0:wp.atomic_add(stats,0,1)
 if slot>=count[0]:return
 wp.atomic_add(stats,1,1);f=boundary[slot];ids=face_vertices(tets,f)
 a=points[ids[0]];b=points[ids[1]];c=points[ids[2]];d=points[tets[f//4,f%4]]
 for axis in range(3):
  plane=a[axis]
  if plane!=b[axis] or plane!=c[axis] or plane==d[axis]:continue
  if not wp.isfinite(plane) or not wp.isfinite(d[axis]):continue
  wp.atomic_add(stats,2,1);wp.atomic_add(stats,6+axis,1)
  nonzero=int(plane!=0.0);wp.atomic_add(stats,4,nonzero)
  support=bool(False)
  if d[axis]>plane:support=tree.lower[1][axis]>=plane
  else:support=tree.upper[1][axis]<=plane
  if support:
   wp.atomic_add(stats,3,1);wp.atomic_add(stats,5,nonzero)
  else:wp.atomic_add(stats,9,1)
  return

def install():
 cls=_proved_completion.ProvedCertificate;original_init=cls.__init__;original_hull=cls._hull_support
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.axis_observer_stats=wp.zeros(len(COLUMNS),dtype=int)
   wp.load_module(module=__name__,device=self.device)
  instances.append(self)
 def hull(self,tree):
  wp.launch(observe,len(self._boundary),inputs=[self.points,self.tetrahedra,self._boundary,self._boundary_count,tree,self.axis_observer_stats],device=self.device)
  original_hull(self,tree)
 cls.__init__=init;cls._hull_support=hull
 return instances
