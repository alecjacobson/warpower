"""Select private sample quadrature before capture and observe complete-mesher storage."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
import warp as wp
from warpower import _mesher, _extraction
from surface_sample_parallel_install_v1 import install

phase=os.environ['WARPOWER_SAMPLE_PHASE'];assert phase in ['before','after']
if phase=='after':install()
Base=_mesher.VariationalMeshingWorkspace
instances=[]
class ObservedMesher(Base):
    def __init__(self,domain,*args,**kwargs):
        before=wp.get_mempool_used_mem_current(domain.device);start=time.perf_counter()
        super().__init__(domain,*args,**kwargs)
        self.observed_setup_seconds=time.perf_counter()-start
        self.observed_pool_bytes=wp.get_mempool_used_mem_current(domain.device)-before
        instances.append(self)

# Both benchmark and gallery import the public class; keep the complete real
# setup, captured solve and validation path with only this private kernel change.
import warpower
warpower.VariationalMeshingWorkspace=ObservedMesher
_mesher.VariationalMeshingWorkspace=ObservedMesher
w=Path(__file__).parent
try:
    if os.environ['WARPOWER_SAMPLE_KIND']=='benchmark':runpy.run_module('benchmarks.mesher',run_name='__main__')
    else:runpy.run_module('examples.vtm_gallery',run_name='__main__')
finally:
    assert len(instances)==1
    m=instances[0];r=getattr(m,'topology_recovery',None)
    paths=[Path(__file__),w/'surface_sample_parallel_install_v1.py',w/'surface_sample_parallel_v1.py',Path(_extraction.__file__)]
    report=dict(scope=__doc__,phase=phase,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},constructor_seconds=m.observed_setup_seconds,persistent_workspace_pool_bytes=m.observed_pool_bytes,recovery=None)
    if r is not None:report['recovery']={name:getattr(r,name).numpy().tolist() for name in ['chosen','invoked','accepted','attempts','error','status','rejection_reason','metrics','previous_metrics','topology','previous_topology','limit','frame','proposed_position','references_rejected','last_errors','point_count','full_count','retained_count']}
    out=Path(os.environ['WARPOWER_SAMPLE_REPORT']);assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
