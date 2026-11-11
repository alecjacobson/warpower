"""Experimental reuse of oriented hull-face proofs within one completion call.

Never cache across moving inputs. Hash misses (including a full table) perform
the original check. No caller-supplied proof is accepted. Production kernels and
the rest of the full geometric certificate remain unchanged.
"""

import warp as wp

from warpower._mesh_topology import face_vertices
from warpower._proved_completion import record_face_support

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def face_key(ids: wp.vec3i):
    # Preserve orientation: cyclically rotate the smallest index to the front.
    a, b, c = ids[0], ids[1], ids[2]
    if b < a and b < c:
        a, b, c = b, c, a
    elif c < a and c < b:
        a, b, c = c, a, b
    return (
        wp.uint64(a) | (wp.uint64(b) << wp.uint64(21)) | (wp.uint64(c) << wp.uint64(42))
    ) + wp.uint64(1)


@wp.func
def hash_slot(key: wp.uint64, capacity: int):
    value = key
    value = (value ^ (value >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    value = value ^ (value >> wp.uint64(31))
    return int(value & wp.uint64(capacity - 1))


@wp.kernel
def lookup(
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    table: wp.array[wp.uint64],
    supported: wp.array[int],
    hits: wp.array[int],
    statistics: wp.array[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    key = face_key(face_vertices(tets, boundary[i]))
    slot = hash_slot(key, table.shape[0])
    found = int(0)
    for _ in range(table.shape[0]):
        old = table[slot]
        if old == key:
            found = 1
            break
        if old == wp.uint64(0):
            break
        slot = (slot + 1) & (table.shape[0] - 1)
    hits[i] = found
    # Temporarily zero proved faces so the original kernel skips their queries.
    supported[i] = 1 - found
    wp.atomic_add(statistics, found, 1)


@wp.kernel
def remember(
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
    table: wp.array[wp.uint64],
    supported: wp.array[int],
    hits: wp.array[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    if hits[i] != 0:
        supported[i] = 1
        return
    # Unsupported faces are permitted in a fragment. All other errors prevent
    # publishing proofs; in particular an unresolved exact predicate is not proof.
    if supported[i] == 0 or (error[0] & (~64)) != 0:
        return
    key = face_key(face_vertices(tets, boundary[i]))
    slot = hash_slot(key, table.shape[0])
    for _ in range(table.shape[0]):
        old = wp.atomic_cas(table, slot, wp.uint64(0), key)
        if old == wp.uint64(0) or old == key:
            return
        slot = (slot + 1) & (table.shape[0] - 1)
    # Saturation only loses a performance hint; future lookups safely miss.


class HullCache:
    def __init__(self, completion):
        c = completion._certificate
        if len(c.points) >= 2**21:
            raise ValueError("Packed oriented face keys require fewer than 2**21 sites")
        with wp.ScopedDevice(c.device):
            self.table = wp.zeros(1 << (8 * len(c.points) - 1).bit_length(), dtype=wp.uint64)
            self.hits = wp.empty_like(c._boundary)
            self.statistics = wp.zeros(2, dtype=int)
            wp.load_module(module=__name__, device=c.device)
        self.certificate = c
        original_run = completion.run

        def run():
            self.table.zero_()
            self.statistics.zero_()
            return original_run()

        completion.run = run
        c._hull_support = self.check

    def check(self, tree):
        c = self.certificate
        wp.launch(
            lookup,
            len(c._boundary),
            inputs=[
                c.tetrahedra,
                c._boundary,
                c._boundary_count,
                self.table,
                c._face_supported,
                self.hits,
                self.statistics,
            ],
            device=c.device,
        )
        c._launch(
            record_face_support,
            c._face_workers * (1 << c._face_depth),
            [
                c.points,
                c.tetrahedra,
                c._boundary,
                c._boundary_count,
                tree,
                c.error,
                c._face_depth,
                c._face_workers,
                c._face_supported,
            ],
        )
        wp.launch(
            remember,
            len(c._boundary),
            inputs=[
                c.tetrahedra,
                c._boundary,
                c._boundary_count,
                c.error,
                self.table,
                c._face_supported,
                self.hits,
            ],
            device=c.device,
        )


def install(workspace):
    workspace._hull_cache_probe = HullCache(workspace._reuse_completion)
    with wp.ScopedCapture(device=workspace.device) as capture:
        workspace._record_reuse()
    workspace._reuse_graph = capture.graph
    return workspace._hull_cache_probe
