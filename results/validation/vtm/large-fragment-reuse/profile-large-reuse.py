"""Diagnostic full-route outcomes; extra device recording is not clean timing."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import warpower
import warpower._active_delaunay as active
from warpower._reuse_delaunay import ReuseDelaunayWorkspace
from benchmarks.mesher import surface

@wp.kernel
def before(count:wp.array[int],prior:wp.array[int]):
    prior[0]=count[0]

@wp.kernel
def record(prior:wp.array[int],attempted:wp.array[int],accepted:wp.array[int],error:wp.array[int],seeds:wp.array[int],rounds:wp.array[int],final:wp.array[int],rows:wp.array2d[int],count:wp.array[int]):
    i=count[0]
    count[0]=i+1
    if i>=rows.shape[0]:return
    rows[i,0]=prior[0]
    rows[i,1]=attempted[0]
    rows[i,2]=accepted[0]
    rows[i,3]=error[0]
    rows[i,4]=seeds[0]
    rows[i,5]=rounds[0]
    rows[i,6]=final[0]

class Observed(ReuseDelaunayWorkspace):
    def __init__(self,points,**kwargs):
        self._trace=wp.zeros((1024,7),dtype=int,device=points.device)
        self._trace_count=wp.zeros(1,dtype=int,device=points.device)
        self._prior=wp.zeros(1,dtype=int,device=points.device)
        wp.load_module(module=__name__,device=points.device)
        super().__init__(points,**kwargs)
    def clear_reuse(self):
        super().clear_reuse();self._trace_count.zero_()
    def _record_reuse(self):
        wp.launch(before,1,inputs=[self._previous_count,self._prior],device=self.device)
        super()._record_reuse()
        wp.launch(record,1,inputs=[self._prior,self.reuse_attempted,self.reuse_accepted,self.reuse_error,self.reuse_seeds,self.reuse_rounds,self.tetrahedra_count,self._trace,self._trace_count],device=self.device)

active.DelaunayWorkspace=Observed
original_clear=active.ActiveDelaunayWorkspace.clear_reuse

def clear(self):
    original_clear(self);self._full.clear_reuse()
active.ActiveDelaunayWorkspace.clear_reuse=clear

def main():
    wp.init()
    domain=warpower.SurfaceDomain(*surface('torus'),surface_subdivisions=4)
    mesh=warpower.VariationalMeshingWorkspace(domain,4096,initial_count=256,seed=2026)
    with wp.ScopedCapture() as cap:
        domain.refit();mesh.run()
    print('captured diagnostic run',flush=True)
    wp.capture_launch(cap.graph);wp.synchronize()
    full=mesh.core.mesh._full;count=int(full._trace_count.numpy()[0]);assert count<=len(full._trace)
    trace=full._trace.numpy()[:count];valid=int(mesh.valid.numpy()[0]);error=int(mesh.error.numpy()[0])
    assert valid==1 and error==0
    report=dict(scope='One fresh complete torus4096 solve with additional GPU outcome counters; not a clean performance sample.',source=__file__,valid=valid,error=error,counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),routing=mesh.core.mesh.route_calls.numpy().tolist(),columns=['prior_cells','attempted','accepted','reuse_error','retained_seeds','completion_rounds','final_cells'],rows=trace.tolist(),attempted=int(trace[:,1].sum()),accepted=int(trace[:,2].sum()),all_prior_cells_retained=int(np.count_nonzero((trace[:,0]>0)&(trace[:,0]==trace[:,4]))))
    Path('.work/vtm/large-reuse-outcomes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['counts','steps','routing','attempted','accepted','all_prior_cells_retained']}),flush=True)
if __name__=='__main__':main()
