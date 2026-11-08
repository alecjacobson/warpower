"""Observe previous-complex validity before unmodified production reconstruction."""
import warp as wp
from warpower._delaunay_certificate import DelaunayCertificate
from warpower import _wall_reuse, _active_delaunay

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
Base=_wall_reuse.WallReuseDelaunayWorkspace
instances=[]
COLUMNS=['previous_cells','positive_cells','certificate_error','sphere_queries','sphere_node_visits','local_certificate','global_certificate','changed_sites','changed_owners','snapshot_index','published_cells','published_error','reuse_accepted','wall_accepted']

@wp.kernel
def differences(points:wp.array[wp.vec3],old:wp.array[wp.vec3],owners:wp.array[int],old_owners:wp.array[int],counts:wp.array[int]):
 i=wp.tid()
 a=points[i];b=old[i]
 if a[0]!=b[0] or a[1]!=b[1] or a[2]!=b[2]:wp.atomic_add(counts,0,1)
 if owners[i]!=old_owners[i]:wp.atomic_add(counts,1,1)

@wp.kernel
def begin_call(slot:wp.array[int]):slot[0]=-1

@wp.kernel
def record_before(count:wp.array[int],positive:wp.array[int],error:wp.array[int],statistics:wp.array[int],local:wp.array[int],glob:wp.array[int],diff:wp.array[int],calls:wp.array[int],records:wp.array2d[int],groups:wp.array[int],slot:wp.array[int],row:wp.array[int],overflow:wp.array[int]):
 i=calls[0];calls[0]+=1;row[0]=i
 if i>=records.shape[0]:overflow[0]=1;return
 records[i,0]=count[0];records[i,1]=positive[0];records[i,2]=error[0]
 records[i,3]=statistics[0];records[i,4]=statistics[1];records[i,5]=local[0];records[i,6]=glob[0]
 records[i,7]=diff[0];records[i,8]=diff[1];records[i,9]=-1
 group=int(-1)
 if error[0]==8 and count[0]==positive[0] and diff[1]==0:group=0
 elif (error[0]&4)!=0:group=1
 elif (error[0]&64)!=0:group=2
 if group>=0 and groups[group]<2:
  slot[0]=2*group+groups[group];groups[group]+=1;records[i,9]=slot[0]

@wp.kernel
def save_points(points:wp.array[wp.vec3],old:wp.array[wp.vec3],slot:wp.array[int],saved:wp.array3d[wp.vec3]):
 i=wp.tid();s=slot[0]
 if s>=0:saved[s,0,i]=old[i];saved[s,1,i]=points[i]

@wp.kernel
def save_cells(tets:wp.array2d[int],count:wp.array[int],slot:wp.array[int],saved:wp.array4d[int],counts:wp.array2d[int],phase:int):
 i=wp.tid();s=slot[0]
 if s>=0:
  if i==0:counts[s,phase]=count[0]
  if i<count[0]:
   for k in range(4):saved[s,phase,i,k]=tets[i,k]

@wp.kernel
def record_after(count:wp.array[int],error:wp.array[int],reuse:wp.array[int],wall:wp.array[int],row:wp.array[int],records:wp.array2d[int]):
 i=row[0]
 if i>=0 and i<records.shape[0]:
  records[i,10]=count[0];records[i,11]=error[0];records[i,12]=reuse[0];records[i,13]=wall[0]

class Observer:
 def __init__(self,mesh):
  self.mesh=mesh;n=mesh.n;m=len(mesh._previous)
  self.cert=DelaunayCertificate(mesh.points,mesh._previous,mesh._previous_count,mesh._original.tree)
  self.previous_points=wp.empty_like(mesh.points);self.previous_owners=wp.empty(n,dtype=int)
  self.diff=wp.zeros(2,dtype=int);self.calls=wp.zeros(1,dtype=int);self.records=wp.zeros((4096,len(COLUMNS)),dtype=int)
  self.groups=wp.zeros(3,dtype=int);self.slot=wp.full(1,-1,dtype=int);self.row=wp.full(1,-1,dtype=int);self.overflow=wp.zeros(1,dtype=int)
  self.points=wp.empty((6,2,n),dtype=wp.vec3);self.tets=wp.empty((6,2,m,4),dtype=int);self.counts=wp.zeros((6,2),dtype=int)
  self.zero=wp.zeros(1,dtype=int);wp.load_module(module=__name__,device=mesh.device)
 def before(self):
  m=self.mesh;c=self.cert;self.diff.zero_();c.run()
  wp.launch(differences,m.n,inputs=[m.points,self.previous_points,m._original.tree.owners,self.previous_owners,self.diff])
  wp.launch(record_before,1,inputs=[m._previous_count,c.tetrahedra_count,c.error,c.statistics,c._local if c._local is not None else self.zero,c._global if c._global is not None else self.zero,self.diff,self.calls,self.records,self.groups,self.slot,self.row,self.overflow])
  wp.launch(save_points,m.n,inputs=[m.points,self.previous_points,self.slot,self.points])
  wp.launch(save_cells,len(m._previous),inputs=[m._previous,m._previous_count,self.slot,self.tets,self.counts,0])
 def after(self):
  m=self.mesh
  wp.launch(record_after,1,inputs=[m.tetrahedra_count,m.error,m.reuse_accepted,m.wall_accepted,self.row,self.records])
  wp.launch(save_cells,len(m.tetrahedra),inputs=[m.tetrahedra,m.tetrahedra_count,self.slot,self.tets,self.counts,1])
  wp.copy(self.previous_points,m.points);wp.copy(self.previous_owners,m._original.tree.owners)

class ObservedMesh(Base):
 def __init__(self,*args,**kwargs):
  self.observer=None;super().__init__(*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.observer=Observer(self);instances.append(self)
   with wp.ScopedCapture() as capture:self._record_reuse()
   self._reuse_graph=capture.graph
  wp.synchronize_device(self.device)
 def _record_reuse(self):
  if self.observer is not None:
   wp.launch(begin_call,1,inputs=[self.observer.slot]);wp.launch(begin_call,1,inputs=[self.observer.row])
  super()._record_reuse()
  if self.observer is not None:self.observer.after()
 def _attempt_reuse(self):
  if self.observer is not None:
   self._refresh_original();self.observer.before()
  super()._attempt_reuse()

def install():
 _wall_reuse.WallReuseDelaunayWorkspace=ObservedMesh
 _active_delaunay.ReuseDelaunayWorkspace=ObservedMesh
 return instances
