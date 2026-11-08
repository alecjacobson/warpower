"""Private quality-gated selective alpha history; not a production implementation.

The original start/finish bodies below are AST-identical copies. Every actual
proposal still executes the original Delaunay/objective/selective checks.
History resets each core call. A warm search retries the full default search
once on exhaustion or apparent convergence, avoiding monotonicity assumptions
for untested larger scales and premature small-displacement termination.
Local vertex rollback is deliberately ineligible in this first experiment.
"""
import warp as wp
from warpower import _vtm, _mesher

wp.set_module_options({"enable_backward":False,"fast_math":False})

@wp.func
def original_start_attempt(alpha: wp.array[wp.float64], damping: wp.float64, attempt: wp.array[int], accepted: wp.array[int], backtracks: wp.array[int], error: wp.array[int], running: wp.array[int]):
    alpha[0] = damping
    attempt[0] = int(error[0] == 0)
    if error[0] != 0:
        running[0] = 0
    accepted[0] = 0
    backtracks[0] = 0

@wp.func
def original_finish_attempt(mesh_error: wp.array[int], boundary_error: wp.array[int], accepted: wp.array[int], attempt: wp.array[int], alpha: wp.array[wp.float64], backtracks: wp.array[int], rejected: wp.array[int], last_geometry: wp.array[int], last_boundary: wp.array[int], iterations: wp.array[int], running: wp.array[int], termination: wp.array[int], maximum: wp.array[wp.float64], tolerance: wp.float64, max_iterations: int, max_backtracks: int, energy: wp.array[wp.float64], boundary_energy: wp.array[wp.float64], history: wp.array2d[wp.float64]):
    if accepted[0] != 0:
        attempt[0] = 0
        iterations[0] += 1
        history[iterations[0], 0] = energy[0]
        history[iterations[0], 1] = boundary_energy[0]
        history[iterations[0], 2] = boundary_energy[1]
        converged = maximum[0] <= tolerance
        termination[0] = int(converged)
        running[0] = int(not converged and iterations[0] < max_iterations)
    else:
        if mesh_error[0] != 0:
            last_geometry[0] = mesh_error[0]
        if boundary_error[0] != 0:
            last_boundary[0] = boundary_error[0]
        rejected[0] += 1
        backtracks[0] += 1
        alpha[0] *= wp.float64(0.5)
        attempt[0] = int(backtracks[0] <= max_backtracks)
        if attempt[0] == 0:
            running[0] = 0
            termination[0] = 2


@wp.kernel
def start_attempt(
    alpha: wp.array[wp.float64], damping: wp.float64, attempt: wp.array[int],
    accepted: wp.array[int], backtracks: wp.array[int], error: wp.array[int],
    running: wp.array[int], mode: wp.array[int], history: wp.array[int],
    statistics: wp.array[int], max_backtracks: int, selection_rejections: wp.array[int],
):
    original_start_attempt(alpha,damping,attempt,accepted,backtracks,error,running)
    history[2]=0  # full default search has not been retried in this iteration
    history[3]=0  # this iteration starts below the default scale
    history[4]=1  # every observed rejection in this search was quality-only
    history[5]=selection_rejections[0]
    if mode[0]==1 and error[0]==0 and history[0]!=0:
        level=wp.max(0,wp.min(history[1]-1,max_backtracks))
        for k in range(level):
            alpha[0]*=wp.float64(0.5)
        # Absolute halving level preserves the original minimum trial scale.
        # rejected_steps continues to count only proposals actually rejected.
        backtracks[0]=level
        history[3]=int(level>0)
        statistics[0]+=int(level>0)


@wp.kernel
def finish_attempt(
    mesh_error: wp.array[int], boundary_error: wp.array[int], accepted: wp.array[int],
    attempt: wp.array[int], alpha: wp.array[wp.float64], backtracks: wp.array[int],
    rejected: wp.array[int], last_geometry: wp.array[int], last_boundary: wp.array[int],
    iterations: wp.array[int], running: wp.array[int], termination: wp.array[int],
    maximum: wp.array[wp.float64], tolerance: wp.float64, max_iterations: int,
    max_backtracks: int, energy: wp.array[wp.float64], boundary_energy: wp.array[wp.float64],
    energy_history: wp.array2d[wp.float64], mode: wp.array[int], history: wp.array[int],
    statistics: wp.array[int], damping: wp.float64,
    selection_rejections: wp.array[int], selection_last_error: wp.array[int],
):
    fresh_quality_failure=(selection_rejections[0]>history[5] and selection_last_error[0]==4
        and mesh_error[0]==0 and boundary_error[0]==0)
    history[5]=selection_rejections[0]
    if accepted[0]==0 and not fresh_quality_failure:
        history[4]=0
    warm=mode[0]==1 and history[3]!=0 and history[2]==0
    retry_convergence=warm and accepted[0]!=0 and maximum[0]<=tolerance
    retry_exhaustion=warm and accepted[0]==0 and backtracks[0]>=max_backtracks and fresh_quality_failure
    retry_nonquality=warm and accepted[0]==0 and not fresh_quality_failure
    if retry_convergence:
        # This valid trial is discarded, not counted as an accepted iteration
        # or a rejected proposal. Recompute from the unchanged original sites.
        history[2]=1
        history[4]=1
        statistics[1]+=1
        alpha[0]=damping
        backtracks[0]=0
        accepted[0]=0
        attempt[0]=1
        running[0]=1
        termination[0]=0
        return
    original_finish_attempt(mesh_error,boundary_error,accepted,attempt,alpha,backtracks,
        rejected,last_geometry,last_boundary,iterations,running,termination,maximum,
        tolerance,max_iterations,max_backtracks,energy,boundary_energy,energy_history)
    if retry_exhaustion or retry_nonquality:
        # The failed warm trial has been counted by the unchanged finish body.
        # Retry all default scales once; no rejection monotonicity is assumed.
        history[2]=1
        history[4]=1
        statistics[2]+=int(retry_exhaustion)
        statistics[4]+=int(retry_nonquality)
        alpha[0]=damping
        backtracks[0]=0
        accepted[0]=0
        attempt[0]=1
        running[0]=1
        termination[0]=0
        return
    if mode[0]==1:
        if accepted[0]!=0:
            history[0]=history[4]
            history[1]=backtracks[0]
            statistics[3]+=int(history[3]!=0 and history[2]==0)
        elif attempt[0]==0:
            history[0]=0


INSTALLED=False


def install():
    global INSTALLED
    assert not INSTALLED
    INSTALLED=True
    cls=_vtm.VTMWorkspace
    base_init=cls.__init__
    def init(self,domain,*args,**kwargs):
        self.private_step_history=None
        self.private_step_statistics=None
        if kwargs.get("selective_enabled") is not None and kwargs.get("local_quality_retries",0)==0:
            self.private_step_history=wp.zeros(6,dtype=int,device=domain.device)
            self.private_step_statistics=wp.zeros(5,dtype=int,device=domain.device)
            wp.load_module(module=__name__,device=domain.device)
        base_init(self,domain,*args,**kwargs)
    cls.__init__=init
    base_record=cls._record
    def record(self):
        if self.private_step_history is not None:
            self.private_step_history.zero_()
        base_record(self)
    cls._record=record
    base_launch=cls._launch
    def launch(self,kernel,dim,args,**kwargs):
        if self.private_step_history is not None:
            if kernel is _vtm.start_attempt:
                kernel=start_attempt
                args=[*args,self.controls.mode,self.private_step_history,self.private_step_statistics,self.max_backtracks,self.selection.rejections]
            elif kernel is _vtm.finish_attempt:
                kernel=finish_attempt
                args=[*args,self.controls.mode,self.private_step_history,self.private_step_statistics,self.damping,self.selection.rejections,self.selection.last_error]
        base_launch(self,kernel,dim,args,**kwargs)
    cls._launch=launch
    base_mesher_record=_mesher.VariationalMeshingWorkspace._record
    def mesher_record(self):
        if self.core.private_step_statistics is not None:
            self.core.private_step_statistics.zero_()
        base_mesher_record(self)
    _mesher.VariationalMeshingWorkspace._record=mesher_record
