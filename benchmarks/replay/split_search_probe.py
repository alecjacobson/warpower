"""Separate strict floating filters from exact/symbolic facet-search fallback.

An uncertain lane is replayed in full with the original ordered predicates.
Certified lanes preserve the exact same scan order and winner. No float sign
or tie outside the existing analytic bounds is accepted.
"""

import warp as wp

from warpower import _analytic_predicates as analytic
from warpower import _bounded_predicates as bounded
from warpower._mesh_topology import face_vertices

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def strict_search(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    supported: wp.array[int],
    statistics: wp.array[int],
    pending: wp.array[int],
    pending_count: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // workers, tid % workers
    candidates[tid] = -1
    if i >= count[0]:
        return
    if lane == 0:
        wp.atomic_add(statistics, wp.where(supported[i] != 0, 0, 1), 1)
    if supported[i] != 0:
        return
    face = face_vertices(tets, faces[i])
    a, b, c = points[face[0]], points[face[1]], points[face[2]]
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        orientation = bounded.orientation_filter(a, b, c, points[q])
        if not analytic.strict(orientation):
            at = wp.atomic_add(pending_count, 0, 1)
            pending[at] = tid
            return
        if orientation.value < wp.float64(0.0):
            continue
        if best < 0:
            best = q
        else:
            value = bounded.sphere_filter(a, b, c, points[best], points[q])
            if not analytic.strict(value):
                at = wp.atomic_add(pending_count, 0, 1)
                pending[at] = tid
                return
            if value.value > wp.float64(0.0):
                best = q
    candidates[tid] = best


@wp.kernel
def exact_search(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    error: wp.array[int],
    pending: wp.array[int],
    pending_count: wp.array[int],
    parallelism: int,
    statistics: wp.array[int],
):
    thread = wp.tid()
    if thread == 0:
        wp.atomic_add(statistics, 0, pending_count[0])
    for slot in range(thread, pending_count[0], parallelism):
        tid = pending[slot]
        i, lane = tid // workers, tid % workers
        face = face_vertices(tets, faces[i])
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        best = int(-1)
        valid = bool(True)
        for q in range(lane, points.shape[0], workers):
            if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
                continue
            orientation = bounded.orient3d(a, b, c, points[q])
            if orientation.valid == 0:
                wp.atomic_or(error, 0, 2)
                valid = False
                break
            if orientation.sign <= 0:
                continue
            if best < 0:
                best = q
            else:
                result = bounded.symbolic_sphere(
                    points, wp.vec4i(face[0], face[1], face[2], best), q
                )
                if result.valid == 0 or result.sign == 0:
                    wp.atomic_or(error, 0, 2)
                    valid = False
                    break
                if result.sign > 0:
                    best = q
        if valid:
            candidates[tid] = best


class SplitSearch:
    def __init__(self, completion):
        self.completion = completion
        if len(completion._candidates) >= 2**31:
            raise ValueError("Exact-lane queue count must fit a signed int32")
        with wp.ScopedDevice(completion.device):
            self.pending = wp.empty_like(completion._candidates)
            self.pending_count = wp.zeros(1, dtype=int)
            self.statistics = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=completion.device, block_dim=32)
            wp.load_module(module=__name__, device=completion.device, block_dim=256)
        original_run = completion.run

        def run():
            self.statistics.zero_()
            return original_run()

        completion.run = run
        completion._search_faces = self.search

    def search(self, tree):
        w = self.completion
        c = w._certificate
        self.pending_count.zero_()
        wp.launch(
            strict_search,
            len(w._candidates),
            inputs=[
                w.points,
                tree.owners,
                w.tetrahedra,
                c._boundary,
                c._boundary_count,
                w._candidates,
                w._workers,
                c._face_supported,
                c.face_statistics,
                self.pending,
                self.pending_count,
            ],
            device=w.device,
            block_dim=256,
        )
        wp.capture_if(
            self.pending_count,
            lambda: wp.launch(
                exact_search,
                4096,
                inputs=[
                    w.points,
                    tree.owners,
                    w.tetrahedra,
                    c._boundary,
                    w._candidates,
                    w._workers,
                    w.error,
                    self.pending,
                    self.pending_count,
                    4096,
                    self.statistics,
                ],
                device=w.device,
                block_dim=32,
            ),
        )


def install(workspace):
    workspace._split_search_probe = SplitSearch(workspace._reuse_completion)
    with wp.ScopedCapture(device=workspace.device) as capture:
        workspace._record_reuse()
    workspace._reuse_graph = capture.graph
    return workspace._split_search_probe
