"""Generate private #130 edits from unchanged production kernels for review."""
from pathlib import Path

root = Path.cwd()
seeds = (root / 'warpower/_facet_seeds.py').read_text()
completion = (root / 'warpower/_facet_completion.py').read_text()
grouped = (root / 'warpower/_grouped_seeds.py').read_text()

header = '''"""Private current-input sphere-proof reuse with exact saved source identity."""
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
                    available: wp.array[int], changed: wp.array[int], count: wp.array[int],
                    changed_flags: wp.array[int]):
    q = wp.tid()
    changed_flags[q] = 0
    if owners[q] != q:
        return
    needs_query = bool(True)
    if available[0] != 0:
        needs_query = saved_owners[q] != q or not same_point(saved_points[q], points[q])
    if needs_query:
        changed_flags[q] = 1
        at = wp.atomic_add(count, 0, 1)
        changed[at] = q


@wp.kernel
def snapshot_ready(error: wp.array[int], count: wp.array[int], capacity: int,
                   available: wp.array[int]):
    available[0] = int(error[0] == 0 and count[0] > 0 and count[0] <= capacity)


'''

keys = seeds[seeds.index('@wp.kernel\ndef seed_keys('):seeds.index('@wp.kernel\ndef reset_live(')]
keys = keys.replace('def seed_keys(', 'def temporal_seed_keys(')
keys = keys.replace('    error: wp.array[int],\n', '''    error: wp.array[int],
    saved_points: wp.array[wp.vec3],
    saved_owners: wp.array[int],
    saved_source: wp.array2d[int],
    saved_count: wp.array[int],
    available: wp.array[int],
    proved: wp.array[int],
''')
keys = keys.replace('    live[i] = 0\n', '    live[i] = 0\n    proved[i] = 0\n')
keys = keys.replace('    v = wp.vec4i(0)\n', '    unchanged = available[0] != 0 and i < saved_count[0]\n    v = wp.vec4i(0)\n')
keys = keys.replace('        v[k] = owners[j]\n', '''        if unchanged:
            if (saved_source[i, k] != j or owners[j] != j or saved_owners[j] != j
                    or not same_point(saved_points[j], points[j])):
                unchanged = False
        v[k] = owners[j]
''')
keys = keys.replace('    live[i] = 1\n', '    live[i] = 1\n    proved[i] = int(unchanged)\n')

pool = completion[completion.index('@wp.kernel\ndef unique_pool('):completion.index('@wp.kernel\ndef progress(')]
pool = pool.replace('def unique_pool(', 'def temporal_unique_pool(')
pool = pool.replace('    error: wp.array[int],\n', '    error: wp.array[int],\n    source_proved: wp.array[int],\n    proved: wp.array[int],\n')
pool = pool.replace('                output[at, k] = key[k]\n', '                output[at, k] = key[k]\n            proved[at] = source_proved[i]\n')

check = grouped[grouped.index('@wp.kernel\ndef check_groups('):grouped.index('class GroupedSeedQueries(')]
check = check.replace('def check_groups(', 'def check_temporal_groups(')
check = check.replace('    upper: wp.array[wp.vec3],\n', '    upper: wp.array[wp.vec3],\n    proved: wp.array[int],\n    changed: wp.array[int],\n    changed_count: wp.array[int],\n    changed_flags: wp.array[int],\n')
a = check.index('    for group in range(lower.shape[0]):')
b = check.index('        if tree.owners[q]', a)
check = check[:a] + '''    groups = lower.shape[0]
    unchanged = proved[i] != 0
    use_compact = unchanged and changed_count[0] <= 32
    if use_compact:
        groups = (changed_count[0] + 31) // 32
    for group in range(groups):
        q = int(-1)
        rank = 32 * group + lane
        if use_compact:
            if rank >= changed_count[0]:
                continue
            q = changed[rank]
        else:
            if sb.outside(box, lower[group], upper[group]):
                continue
            if rank >= points.shape[0]:
                continue
            q = tree.ids[rank]
''' + check[b:]
check = check.replace('        p = points[q]\n', '        if unchanged and not use_compact and changed_flags[q] == 0:\n            continue\n        p = points[q]\n')

body = '''class TemporalSeedQueries(GroupedSeedQueries):
    def __init__(self, *args, proved, changed, changed_count, changed_flags, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal_inputs = [*self._query_inputs, proved, changed, changed_count, changed_flags]
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
            self.changed_flags = wp.empty(len(points), dtype=int)
            self.source_proved = wp.empty(len(source), dtype=int)
            self.proved = wp.empty(len(source), dtype=int)
            self._prepared = TemporalSeedQueries(
                points, self.unique, self.unique_count, tree, self.live, self.error,
                block_dim=block_dim, proved=self.proved, changed=self.changed,
                changed_count=self.changed_count, changed_flags=self.changed_flags)
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
            self.available, self.changed, self.changed_count, self.changed_flags], device=self.device)
        wp.launch(temporal_seed_keys, n, inputs=[
            self.points, self.source, self.count, self.tree.owners, self.keys, self.live,
            self.error, self.saved_points, self.saved_owners, self.saved_source,
            self.saved_count, self.available, self.source_proved], device=self.device, block_dim=32)
        wp.launch(temporal_unique_pool, n, inputs=[
            self.points, self.keys, self.live, self.table, self.unique, self.unique_count,
            self.error, self.source_proved, self.proved], device=self.device)
        wp.launch(reset_live, n, inputs=[self.unique_count, self.live], device=self.device, block_dim=32)
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
'''

out = root / '.work/vtm/temporal_seed_queries.py'
assert not out.exists()
out.write_text(header + keys + pool + check + body)
print(out)
