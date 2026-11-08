"""Observe production constructor memory and recovery diagnostics outside replay timing."""
import hashlib,json,os,time
from pathlib import Path
import warp as wp
import examples.vtm_gallery as gallery

Base=gallery.VariationalMeshingWorkspace
instances=[]
class MeasuredMesher(Base):
    def __init__(self,domain,*args,**kwargs):
        before=wp.get_mempool_used_mem_current(domain.device);start=time.perf_counter()
        super().__init__(domain,*args,**kwargs)
        self.measured_setup_seconds=time.perf_counter()-start
        self.measured_persistent_bytes=wp.get_mempool_used_mem_current(domain.device)-before
        instances.append(self)
gallery.VariationalMeshingWorkspace=MeasuredMesher
try:gallery.main()
finally:
    assert len(instances)==1
    m=instances[0];r=getattr(m,'topology_recovery',None)
    info=dict(scope=__doc__,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),constructor_seconds=m.measured_setup_seconds,persistent_workspace_pool_bytes=m.measured_persistent_bytes,recovery=None)
    if r is not None:
        info['recovery']={name:getattr(r,name).numpy().tolist() for name in ['chosen','invoked','accepted','attempts','error','status','rejection_reason','metrics','previous_metrics','topology','previous_topology','limit','frame','proposed_position','references_rejected','last_errors','point_count','full_count','retained_count']}
    p=Path(os.environ['WARPOWER_TOPOLOGY_DIAGNOSTICS']);assert not p.exists();p.write_text(json.dumps(info,indent=2)+'\n');print('TOPOLOGY_RECOVERY',info,flush=True)
