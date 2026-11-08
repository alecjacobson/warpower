"""Uninstrumented complete solve with private selective history installed at setup."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
import warp as wp
import warpower
from warpower import _mesher
import selective_step_history_v1 as candidate

phase=os.environ["WARPOWER_HISTORY_PHASE"];assert phase in ["before","after"]
path=Path(os.environ["WARPOWER_HISTORY_REPORT"]);assert not path.exists()
sources=[Path(__file__),Path(candidate.__file__),*Path("warpower").glob("*.py")]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
if phase=="after":candidate.install()
Base=_mesher.VariationalMeshingWorkspace
instances=[]


class MeasuredMesher(Base):
    def __init__(self,domain,*args,**kwargs):
        before=wp.get_mempool_used_mem_current(domain.device);start=time.perf_counter()
        super().__init__(domain,*args,**kwargs)
        self.measured_setup_seconds=time.perf_counter()-start
        self.measured_pool_bytes=wp.get_mempool_used_mem_current(domain.device)-before
        instances.append(self)


warpower.VariationalMeshingWorkspace=MeasuredMesher
_mesher.VariationalMeshingWorkspace=MeasuredMesher
completed=False
try:
    module="benchmarks.mesher" if os.environ["WARPOWER_HISTORY_KIND"]=="benchmark" else "examples.vtm_gallery"
    runpy.run_module(module,run_name="__main__")
    completed=True
finally:
    report=dict(scope=__doc__,phase=phase,completed=completed,source_sha256=hashes)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    if instances:
        assert len(instances)==1
        m=instances[0];c=m.core
        report.update(constructor_seconds=m.measured_setup_seconds,persistent_workspace_pool_bytes=m.measured_pool_bytes)
        stats=getattr(c,"private_step_statistics",None)
        report["history_statistics"]=None if stats is None else dict(zip(
            ["warm_starts","convergence_default_retries","exhaustion_default_retries","warm_accepts"],map(int,stats.numpy())))
        r=m.topology_recovery
        report["recovery"]=None if r is None else {name:getattr(r,name).numpy().tolist() for name in
            ["chosen","invoked","accepted","attempts","error","status","rejection_reason","metrics",
             "previous_metrics","topology","previous_topology","limit","frame","proposed_position",
             "references_rejected","last_errors","point_count","full_count","retained_count"]}
    path.write_text(json.dumps(report,indent=2)+"\n")
