"""Observe exact input equality; every original reconstruction still executes."""
import warp as wp
from warpower import _wall_reuse, _active_delaunay

wp.set_module_options({"enable_backward": False, "fast_math": False})
Base = _wall_reuse.WallReuseDelaunayWorkspace
instances = []
COLUMNS = ["epoch", "previous_valid", "changed_sites", "previous_cells",
           "published_cells", "published_error", "reuse_accepted", "wall_accepted"]


@wp.kernel
def fresh(valid: wp.array[int], epoch: wp.array[int]):
    valid[0] = 0
    epoch[0] += 1


@wp.kernel
def differences(bits: wp.array2d[wp.uint32], old: wp.array2d[wp.uint32],
                valid: wp.array[int], changed: wp.array[int]):
    i = wp.tid()
    if valid[0] != 0:
        if bits[i, 0] != old[i, 0] or bits[i, 1] != old[i, 1] or bits[i, 2] != old[i, 2]:
            wp.atomic_add(changed, 0, 1)


@wp.kernel
def record_before(valid: wp.array[int], epoch: wp.array[int], changed: wp.array[int],
                  previous_count: wp.array[int], calls: wp.array[int],
                  records: wp.array2d[int], overflow: wp.array[int]):
    i = calls[0]
    calls[0] += 1
    if i >= records.shape[0]:
        overflow[0] = 1
        return
    records[i, 0] = epoch[0]
    records[i, 1] = valid[0]
    records[i, 2] = wp.where(valid[0] != 0, changed[0], -1)
    records[i, 3] = previous_count[0]


@wp.kernel
def record_after(count: wp.array[int], error: wp.array[int], reuse: wp.array[int],
                 wall: wp.array[int], calls: wp.array[int], records: wp.array2d[int],
                 valid: wp.array[int], previous_count: wp.array[int]):
    i = calls[0] - 1
    valid[0] = int(error[0] == 0 and count[0] > 0)
    previous_count[0] = wp.where(valid[0] != 0, count[0], 0)
    if i >= 0 and i < records.shape[0]:
        records[i, 4] = count[0]
        records[i, 5] = error[0]
        records[i, 6] = reuse[0]
        records[i, 7] = wall[0]


class Observation:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bits = wp.array(ptr=mesh.points.ptr, shape=(mesh.n, 3), dtype=wp.uint32,
                             device=mesh.device)
        self.previous = wp.empty_like(self.bits)
        self.valid = wp.zeros(1, dtype=int)
        self.epoch = wp.zeros(1, dtype=int)
        self.changed = wp.zeros(1, dtype=int)
        self.previous_count = wp.zeros(1, dtype=int)
        self.calls = wp.zeros(1, dtype=int)
        self.overflow = wp.zeros(1, dtype=int)
        self.zero = wp.zeros(1, dtype=int)
        self.records = wp.zeros((4096, len(COLUMNS)), dtype=int)
        wp.load_module(module=__name__, device=mesh.device)

    def before(self):
        m = self.mesh
        self.changed.zero_()
        wp.launch(differences, m.n, inputs=[self.bits, self.previous, self.valid, self.changed], device=m.device)
        wp.launch(record_before, 1, inputs=[self.valid, self.epoch, self.changed,
                  self.previous_count, self.calls, self.records, self.overflow], device=m.device)

    def after(self):
        m = self.mesh
        wp.launch(record_after, 1, inputs=[m.tetrahedra_count, m.error, m.reuse_accepted,
                  getattr(m, "wall_accepted", self.zero), self.calls, self.records,
                  self.valid, self.previous_count], device=m.device)
        wp.copy(self.previous, self.bits)


class ObservedMesh(Base):
    def __init__(self, *args, **kwargs):
        self.observer = None
        super().__init__(*args, **kwargs)
        with wp.ScopedDevice(self.device):
            self.observer = Observation(self)
            instances.append(self)
            with wp.ScopedCapture(device=self.device) as capture:
                self._record_reuse()
            self._reuse_graph = capture.graph
        wp.synchronize_device(self.device)

    def clear_reuse(self):
        super().clear_reuse()
        if self.observer is not None:
            wp.launch(fresh, 1, inputs=[self.observer.valid, self.observer.epoch], device=self.device)

    def _record_reuse(self):
        if self.observer is not None:
            self.observer.before()
        super()._record_reuse()
        if self.observer is not None:
            self.observer.after()


def install():
    _wall_reuse.WallReuseDelaunayWorkspace = ObservedMesh
    _active_delaunay.ReuseDelaunayWorkspace = ObservedMesh
    return instances
