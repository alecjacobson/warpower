"""Private current-input sphere-proof reuse with exact saved source identity."""
import warp as wp
from warpower import _filtered_predicates as fp, _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._facet_completion import sorted_tet
from warpower._facet_seeds import CertifiedSeeds, reset_live, compact
from warpower._grouped_seeds import GroupedSeedQueries, group_bounds
from warpower._prepared_seeds import prepare, point_value

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func(inline=True)
def same_point(a: wp.vec3, b: wp.vec3):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.kernel
def changed_queries(points: wp.array[wp.vec3], owners: wp.array[int],
                    saved_points: wp.array[wp.vec3], saved_owners: wp.array[int],
                    available: wp.array[int], changed: wp.array[int], count: wp.array[int]):
    q = wp.tid()
    if owners[q] != q:
        return
    needs_query = bool(True)
    if available[0] != 0:
        needs_query = saved_owners[q] != q or not same_point(saved_points[q], points[q])
    if needs_query:
        at = wp.atomic_add(count, 0, 1)
        changed[at] = q


@wp.kernel
def snapshot_ready(error: wp.array[int], count: wp.array[int], capacity: int,
                   available: wp.array[int]):
    available[0] = int(error[0] == 0 and count[0] > 0 and count[0] <= capacity)


@wp.kernel
def temporal_seed_keys(
    points: wp.array[wp.vec3],
    source: wp.array2d[int],
    count: wp.array[int],
    owners: wp.array[int],
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    error: wp.array[int],
    saved_points: wp.array[wp.vec3],
    saved_owners: wp.array[int],
    saved_source: wp.array2d[int],
    saved_count: wp.array[int],
    available: wp.array[int],
    proved: wp.array[int],
):
    i = wp.tid()
    live[i] = 0
    proved[i] = 0
    if count[0] < 0 or count[0] > source.shape[0]:
        wp.atomic_or(error, 0, 4)
        return
    if i >= count[0]:
        return
    unchanged = available[0] != 0 and i < saved_count[0]
    v = wp.vec4i(0)
    for k in range(4):
        j = source[i, k]
        if j < 0 or j >= points.shape[0]:
            return
        if unchanged:
            if (saved_source[i, k] != j or owners[j] != j or saved_owners[j] != j
                    or not same_point(saved_points[j], points[j])):
                unchanged = False
        v[k] = owners[j]
        for previous in range(k):
            if v[previous] == v[k]:
                return
    o = fp.orient3d(points[v[0]], points[v[1]], points[v[2]], points[v[3]])
    if o.valid == 0:
        wp.atomic_or(error, 0, 2)
        return
    if o.sign == 0:
        return
    keys[i] = sorted_tet(v)
    live[i] = 1
    proved[i] = int(unchanged)


@wp.kernel
def temporal_unique_pool(
    points: wp.array[wp.vec3],
    keys: wp.array[wp.vec4i],
    live: wp.array[int],
    table: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    source_proved: wp.array[int],
    proved: wp.array[int],
):
    i = wp.tid()
    if live[i] == 0:
        return
    key = keys[i]
    slot = int(
        (
            wp.uint32(key[0]) * wp.uint32(73856093)
            ^ wp.uint32(key[1]) * wp.uint32(19349663)
            ^ wp.uint32(key[2]) * wp.uint32(83492791)
            ^ wp.uint32(key[3]) * wp.uint32(2654435761)
        )
        & wp.uint32(table.shape[0] - 1)
    )
    for _ in range(table.shape[0]):
        old = wp.atomic_cas(table, slot, -1, i)
        if old == -1:
            at = wp.atomic_add(count, 0, 1)
            if at >= output.shape[0]:
                wp.atomic_or(error, 0, 4)
                return
            sign = fp.orient3d(points[key[0]], points[key[1]], points[key[2]], points[key[3]])
            if sign.valid == 0 or sign.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if sign.sign < 0:
                a = int(key[0])
                key[0] = key[1]
                key[1] = a
            for k in range(4):
                output[at, k] = key[k]
            proved[at] = source_proved[i]
            return
        if keys[old] == key:
            return
        slot = (slot + 1) & (table.shape[0] - 1)
    wp.atomic_or(error, 0, 4)


@wp.kernel
def check_temporal_groups(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    proved: wp.array[int],
    changed: wp.array[int],
    changed_count: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    groups = lower.shape[0]
    unchanged = proved[i] != 0
    if unchanged:
        groups = (changed_count[0] + 31) // 32
    for group in range(groups):
        q = int(-1)
        rank = 32 * group + lane
        if unchanged:
            if rank >= changed_count[0]:
                continue
            q = changed[rank]
        else:
            if sb.outside(box, lower[group], upper[group]):
                continue
            if rank >= points.shape[0]:
                continue
            q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p):
            continue
        value = point_value(shape, p)
        if (
            wp.isfinite(value.value)
            and wp.isfinite(value.error)
            and wp.abs(value.value) > value.error
        ):
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


class TemporalSeedQueries(GroupedSeedQueries):
    def __init__(self, *args, proved, changed, changed_count, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal_inputs = [*self._query_inputs, proved, changed, changed_count]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        wp.launch(group_bounds, len(self.lower), inputs=self._group_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(check_temporal_groups, 32 * self.capacity, inputs=self._temporal_inputs,
                  block_dim=self.block_dim, device=self.device)


class TemporalSeeds(CertifiedSeeds):
    def __init__(self, points, source, count, tree, *, block_dim=256, prepare_geometry=False):
        assert prepare_geometry
        super().__init__(points, source, count, tree, block_dim=block_dim, prepare_geometry=False)
        with wp.ScopedDevice(self.device):
            self.saved_points = wp.empty_like(points)
            self.saved_owners = wp.empty_like(tree.owners)
            self.saved_source = wp.empty_like(source)
            self.saved_count = wp.zeros(1, dtype=int)
            self.available = wp.zeros(1, dtype=int)
            self.changed = wp.empty(len(points), dtype=int)
            self.changed_count = wp.zeros(1, dtype=int)
            self.source_proved = wp.empty(len(source), dtype=int)
            self.proved = wp.empty(len(source), dtype=int)
            self._prepared = TemporalSeedQueries(
                points, self.unique, self.unique_count, tree, self.live, self.error,
                block_dim=block_dim, proved=self.proved, changed=self.changed,
                changed_count=self.changed_count)
            wp.load_module(module=__name__, device=self.device, block_dim=block_dim)
            wp.load_module(module=__name__, device=self.device, block_dim=256)

    def remember(self, error):
        # The caller has just completed the certified constructor, copied its
        # output into source, and zeroed count on failure. Publication is last.
        wp.copy(self.saved_points, self.points)
        wp.copy(self.saved_owners, self.tree.owners)
        wp.copy(self.saved_source, self.source)
        wp.copy(self.saved_count, self.count)
        wp.launch(snapshot_ready, 1, inputs=[error, self.count, len(self.source), self.available], device=self.device)

    def clear(self):
        self.available.zero_()
        self.saved_count.zero_()

    def run(self):
        n = len(self.source)
        self.error.zero_()
        self.unique_count.zero_()
        self.tetrahedra_count.zero_()
        self.table.fill_(-1)
        self.proved.zero_()
        self.changed_count.zero_()
        wp.launch(changed_queries, len(self.points), inputs=[
            self.points, self.tree.owners, self.saved_points, self.saved_owners,
            self.available, self.changed, self.changed_count], device=self.device)
        wp.launch(temporal_seed_keys, n, inputs=[
            self.points, self.source, self.count, self.tree.owners, self.keys, self.live,
            self.error, self.saved_points, self.saved_owners, self.saved_source,
            self.saved_count, self.available, self.source_proved], device=self.device)
        wp.launch(temporal_unique_pool, n, inputs=[
            self.points, self.keys, self.live, self.table, self.unique, self.unique_count,
            self.error, self.source_proved, self.proved], device=self.device)
        wp.launch(reset_live, n, inputs=[self.unique_count, self.live], device=self.device, block_dim=self._block_dim)
        self._prepared.run()
        wp.launch(compact, n, inputs=[self.unique, self.live, self.tetrahedra,
                                     self.tetrahedra_count, self.error], device=self.device)


def install():
    from warpower import _reuse_delaunay
    original_record = _reuse_delaunay.ReuseDelaunayWorkspace._record_reuse
    original_clear = _reuse_delaunay.ReuseDelaunayWorkspace.clear_reuse

    def record(self):
        assert self._original.tree.owners.ptr == self._base._hierarchy.owners.ptr
        original_record(self)
        self._reuse_filter.remember(self.error)

    def clear(self):
        original_clear(self)
        self._reuse_filter.clear()

    _reuse_delaunay.CertifiedSeeds = TemporalSeeds
    _reuse_delaunay.ReuseDelaunayWorkspace._record_reuse = record
    _reuse_delaunay.ReuseDelaunayWorkspace.clear_reuse = clear
