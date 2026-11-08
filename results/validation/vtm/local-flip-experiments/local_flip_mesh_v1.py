"""Private VTM dispatch experiment; independent public APIs remain unchanged."""
import warp as wp
from warpower import _wall_reuse,_active_delaunay
from local_flips_v1 import Candidate
Base=_wall_reuse.WallReuseDelaunayWorkspace
instances=[]
COLUMNS=['attempted','already_valid','repaired','stalled_or_exhausted','ineligible','repair_rounds','applied_2_to_3','applied_3_to_2']
wp.set_module_options({'enable_backward':False})
@wp.kernel
def record(accepted:wp.array[int],rounds:wp.array[int],history:wp.array2d[int],counters:wp.array[int]):
 counters[0]+=1;r=rounds[0]
 if accepted[0]!=0:
  if r==0:counters[1]+=1
  else:counters[2]+=1
 else:
  if r>0:counters[3]+=1
  else:counters[4]+=1
 counters[5]+=r
 for i in range(r):
  counters[6]+=history[i,2];counters[7]+=history[i,3]

class FlipMesh(Base):
 def __init__(self,*args,**kwargs):
  self.flips=None;super().__init__(*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.flips=Candidate(self.points,self._previous,self._previous_count,self._original.tree,max_rounds=8)
   self.flips._certificate=self.flips.cert
   self.flip_counters=wp.zeros(len(COLUMNS),dtype=int);instances.append(self)
   wp.load_module(module=__name__,device=self.device)
   with wp.ScopedCapture() as capture:self._record_reuse()
   self._reuse_graph=capture.graph
  wp.synchronize_device(self.device)
 def _attempt_reuse(self):
  if self.flips is None:return super()._attempt_reuse()
  self._refresh_original();self.flips.run()
  wp.launch(record,1,inputs=[self.flips.accepted,self.flips.rounds,self.flips.history,self.flip_counters],device=self.device)
  wp.capture_if(self.flips.accepted,self._publish_flips,lambda:super(FlipMesh,self)._attempt_reuse())
 def _publish_flips(self):
  # Python-only capture-time routing to the existing complete publication
  # machinery, including the matching certified face index for pole extraction.
  previous=self._reuse_completion
  try:
   self._reuse_completion=self.flips
   super()._publish_reuse()
  finally:self._reuse_completion=previous
  self.reuse_accepted.fill_(1);self.reuse_error.zero_()
  wp.copy(self.reuse_seeds,self.flips.tetrahedra_count);wp.copy(self.reuse_rounds,self.flips.rounds)

def install():
 _wall_reuse.WallReuseDelaunayWorkspace=FlipMesh
 _active_delaunay.ReuseDelaunayWorkspace=FlipMesh
 return instances
