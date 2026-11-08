"""Read-only device trace of original VTM stages, trials, guards and reconstruction.

Private instrumentation: writes only owned trace buffers. No solver kernel,
acceptance guard, argument or allocation is replaced. Trace timing is not a
performance result. Host hooks choose graph recording sites, not replay paths.
"""
import functools
import warp as wp
from warpower import _mesher, _vtm, _selective, _selective_backtracking

wp.set_module_options({"enable_backward": False, "fast_math": False})

EVENTS = ["unused", "core_begin", "core_end", "start_attempt", "candidate_ready",
          "finish_pre", "finish_post", "reconstruct_pre", "reconstruct_post",
          "selection_pre", "selection_post", "stage_enter", "stage_exit",
          "trial_begin", "trial_end", "local_retry"]
STAGES = ["setup", "_solve", "_growth", "_refine", "_refine_one", "_polish",
          "_recovery", "_proven_recovery", "_quality_round", "_jitter", "_interior",
          "_initial_extract", "_extract", "topology_recovery"]
PHASES = ["outside_core", "core_solve", "line_trial", "line_restore"]
INT_COLUMNS = ["event", "stage", "phase", "active", "mode", "budget", "iterations",
               "backtracks", "rejected", "accepted", "attempt", "running", "termination",
               "core_error", "mesh_error", "boundary_error", "invalid", "full_count",
               "selection_error", "selection_count", "selection_rejections", "selection_last_error",
               "local_used", "local_retries", "local_blocked"]
FLOAT_COLUMNS = ["alpha", "maximum", "energy", "old_energy", "surface", "crease",
                 "old_surface", "old_crease", "minimum", "previous_minimum"]


@wp.struct
class Inputs:
    active: wp.array[int]
    mode: wp.array[int]
    budget: wp.array[int]
    iterations: wp.array[int]
    backtracks: wp.array[int]
    rejected: wp.array[int]
    accepted: wp.array[int]
    attempt: wp.array[int]
    running: wp.array[int]
    termination: wp.array[int]
    core_error: wp.array[int]
    mesh_error: wp.array[int]
    boundary_error: wp.array[int]
    invalid: wp.array[int]
    full_count: wp.array[int]
    selection_error: wp.array[int]
    selection_count: wp.array[int]
    selection_rejections: wp.array[int]
    selection_last_error: wp.array[int]
    local_used: wp.array[int]
    local_statistics: wp.array[int]
    alpha: wp.array[wp.float64]
    maximum: wp.array[wp.float64]
    energy: wp.array[wp.float64]
    old_energy: wp.array[wp.float64]
    boundary: wp.array[wp.float64]
    old_boundary: wp.array[wp.float64]
    minimum: wp.array[wp.float64]
    previous_minimum: wp.array[wp.float64]


@wp.kernel
def snapshot(s: Inputs, event: int, stage: int, phase: int, count: wp.array[int],
             overflow: wp.array[int], ints: wp.array2d[int], floats: wp.array2d[wp.float64]):
    row = count[0]
    count[0] += 1
    if row >= ints.shape[0]:
        overflow[0] = 1
        return
    ints[row, 0] = event
    ints[row, 1] = stage
    ints[row, 2] = phase
    ints[row, 3] = s.active[0]
    ints[row, 4] = s.mode[0]
    ints[row, 5] = s.budget[0]
    ints[row, 6] = s.iterations[0]
    ints[row, 7] = s.backtracks[0]
    ints[row, 8] = s.rejected[0]
    ints[row, 9] = s.accepted[0]
    ints[row, 10] = s.attempt[0]
    ints[row, 11] = s.running[0]
    ints[row, 12] = s.termination[0]
    ints[row, 13] = s.core_error[0]
    ints[row, 14] = s.mesh_error[0]
    ints[row, 15] = s.boundary_error[0]
    ints[row, 16] = s.invalid[0]
    ints[row, 17] = s.full_count[0]
    ints[row, 18] = s.selection_error[0]
    ints[row, 19] = s.selection_count[0]
    ints[row, 20] = s.selection_rejections[0]
    ints[row, 21] = s.selection_last_error[0]
    ints[row, 22] = s.local_used[0]
    ints[row, 23] = s.local_statistics[0]
    ints[row, 24] = s.local_statistics[1]
    # Saved objective arrays are uninitialized before the first actual step.
    # Outside a step these values are deliberately unavailable, not stale data.
    for j in range(10):
        floats[row, j] = wp.float64(wp.nan)
    if phase == 2 or event == 3:
        floats[row, 0] = s.alpha[0]
        floats[row, 1] = s.maximum[0]
        floats[row, 2] = s.energy[0]
        floats[row, 3] = s.old_energy[0]
        floats[row, 4] = s.boundary[0]
        floats[row, 5] = s.boundary[1]
        floats[row, 6] = s.old_boundary[0]
        floats[row, 7] = s.old_boundary[1]
        floats[row, 8] = s.minimum[0]
        floats[row, 9] = s.previous_minimum[0]


class Observer:
    def __init__(self, device):
        self.device = device
        self.stage, self.phase = 0, 0
        self.capacity = 65536
        self.count = wp.zeros(1, dtype=int, device=device)
        self.overflow = wp.zeros(1, dtype=int, device=device)
        self.ints = wp.empty((self.capacity, len(INT_COLUMNS)), dtype=int, device=device)
        self.floats = wp.empty((self.capacity, len(FLOAT_COLUMNS)), dtype=wp.float64, device=device)
        self.zero_int = wp.zeros(2, dtype=int, device=device)
        self.zero_float = wp.zeros(1, dtype=wp.float64, device=device)
        self.inputs = None
        wp.load_module(module=__name__, device=device)

    def bind(self, c):
        s = self.inputs = Inputs()
        for name, value in dict(active=c.active_count, mode=c.controls.mode, budget=c.controls.budget,
            iterations=c.iterations, backtracks=c._backtracks, rejected=c.rejected_steps,
            accepted=c._accepted, attempt=c._attempt, running=c._running, termination=c.termination,
            core_error=c.error, mesh_error=c.mesh.error, boundary_error=c.boundary.error,
            invalid=c._invalid, full_count=c.full_tetrahedra_count, alpha=c._alpha,
            maximum=c._max_move, energy=c.energy, old_energy=c._old_energy,
            boundary=c.boundary.energy, old_boundary=c._old_boundary).items():
            setattr(s, name, value)
        for name, attribute in [("selection_error", "error"), ("selection_count", "retained_count"),
                                ("selection_rejections", "rejections"), ("selection_last_error", "last_error")]:
            setattr(s, name, self.zero_int if c.selection is None else getattr(c.selection, attribute))
        for name, attribute in [("minimum", "minimum"), ("previous_minimum", "previous_minimum")]:
            setattr(s, name, self.zero_float if c.selection is None else getattr(c.selection, attribute))
        s.local_used = self.zero_int if c.local_backtracking is None else c.local_backtracking.used
        s.local_statistics = self.zero_int if c.local_backtracking is None else c.local_backtracking.statistics

    def record(self, c, event):
        if self.inputs is None:
            self.bind(c)
        wp.launch(snapshot, 1, inputs=[self.inputs, EVENTS.index(event), self.stage, self.phase,
                  self.count, self.overflow, self.ints, self.floats], device=self.device)

    def reset(self):
        self.count.zero_()
        self.overflow.zero_()


INSTALLED = False
HOOKS = []


def install():
    global INSTALLED
    assert not INSTALLED
    INSTALLED = True
    cls = _vtm.VTMWorkspace
    original_init = cls.__init__
    def init(self, domain, *args, **kwargs):
        self.step_observer = Observer(domain.device)
        original_init(self, domain, *args, **kwargs)
    cls.__init__ = init
    HOOKS.append("VTMWorkspace.__init__: allocate private observer before original constructor")
    original_launch = cls._launch
    def launch(self, kernel, dim, args, **kwargs):
        o = self.step_observer
        if kernel is _vtm.finish_attempt:
            o.record(self, "finish_pre")
        original_launch(self, kernel, dim, args, **kwargs)
        if kernel is _vtm.start_attempt:
            o.record(self, "start_attempt")
        elif kernel is _vtm.candidate_ready:
            o.record(self, "candidate_ready")
        elif kernel is _vtm.finish_attempt:
            o.record(self, "finish_post")
    cls._launch = launch
    HOOKS.append("VTMWorkspace._launch: original launch unchanged plus neighboring snapshots")

    def around(cls, name, core_fn, before=None, after=None, phase=None, stage=None):
        old = getattr(cls, name)
        @functools.wraps(old)
        def wrapped(self, *args, **kwargs):
            c = core_fn(self)
            o = c.step_observer
            saved = o.phase, o.stage
            if phase is not None:
                o.phase = PHASES.index(phase)
            if stage is not None:
                o.stage = STAGES.index(stage)
            try:
                if before:
                    o.record(c, before)
                result = old(self, *args, **kwargs)
                if after:
                    o.record(c, after)
                return result
            finally:
                o.phase, o.stage = saved
        setattr(cls, name, wrapped)
        HOOKS.append(cls.__name__ + "." + name)

    around(cls, "_record", lambda c:c, "core_begin", "core_end")
    around(cls, "_record_valid", lambda c:c, phase="core_solve")
    around(cls, "_try_step", lambda c:c, "trial_begin", "trial_end", phase="line_trial")
    around(cls, "_restore", lambda c:c, phase="line_restore")
    around(cls, "_reconstruct", lambda c:c, "reconstruct_pre", "reconstruct_post")
    around(_selective.SelectiveUpdate, "_check", lambda s:s.core, "selection_pre", "selection_post")
    around(_selective_backtracking.LocalQualityBacktracking, "_retry", lambda l:l.core,
           after="local_retry")
    for name in STAGES[1:-1]:
        around(_mesher.VariationalMeshingWorkspace, name, lambda m:m.core,
               "stage_enter", "stage_exit", stage=name)
    from warpower._topology_recovery import TopologyRecovery
    around(TopologyRecovery, "run", lambda r:r.core, "stage_enter", "stage_exit", stage="topology_recovery")
    original_record = _mesher.VariationalMeshingWorkspace._record
    def record(self):
        self.core.step_observer.reset()
        original_record(self)
    _mesher.VariationalMeshingWorkspace._record = record
    HOOKS.append("VariationalMeshingWorkspace._record: reset only observer at fresh complete replay")
