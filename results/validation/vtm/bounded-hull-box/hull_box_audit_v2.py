"""Compare old/new complete support decisions throughout a full VTM trajectory."""
import warp as wp
from warpower import _proved_completion
from bounded_hull_box_v2 import install as install_candidate
from hull_box_original_v1 import record_face_support as original_kernel
instances=[]
COLUMNS=['calls','faces','mask_mismatches','error_mismatches','supported_faces']
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def compare(count:wp.array[int],old:wp.array[int],new:wp.array[int],old_error:wp.array[int],new_error:wp.array[int],stats:wp.array[int]):
 i=wp.tid()
 if i==0:
  wp.atomic_add(stats,0,1)
  if old_error[0]!=new_error[0]:wp.atomic_add(stats,3,1)
 if i<count[0]:
  wp.atomic_add(stats,1,1)
  if old[i]!=new[i]:wp.atomic_add(stats,2,1)
  if new[i]!=0:wp.atomic_add(stats,4,1)

def install():
 wp.load_module(module="hull_box_original_v1",device="cuda:0",block_dim=32)
 wp.load_module(module='bounded_hull_box_v2',device='cuda:0',block_dim=32)
 install_candidate()
 cls=_proved_completion.ProvedCertificate;original_init=cls.__init__;original_hull=cls._hull_support
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.hull_box_audit_old=wp.empty_like(self._face_supported);self.hull_box_audit_error=wp.zeros(1,dtype=int);self.hull_box_audit_stats=wp.zeros(len(COLUMNS),dtype=int)
   wp.load_module(module=__name__,device=self.device)
  instances.append(self)
 def hull(self,tree):
  wp.copy(self.hull_box_audit_error,self.error);self.hull_box_audit_old.fill_(1)
  original_hull(self,tree)
  self._launch(original_kernel,self._face_workers*(1<<self._face_depth),[self.points,self.tetrahedra,self._boundary,self._boundary_count,tree,self.hull_box_audit_error,self._face_depth,self._face_workers,self.hull_box_audit_old])
  wp.launch(compare,len(self._boundary),inputs=[self._boundary_count,self.hull_box_audit_old,self._face_supported,self.hull_box_audit_error,self.error,self.hull_box_audit_stats],device=self.device)
 cls.__init__=init;cls._hull_support=hull
 return instances
