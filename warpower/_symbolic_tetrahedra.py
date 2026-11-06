"""Optional symbolic Voronoi candidate construction on original float32 sites."""

import warp as wp

from . import _bvh
from ._cell import Scratch64, plane_at
from ._symbolic_cell import clip_symbolic, init_symbolic
from ._tetrahedra import sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel(enable_backward=False)
def build_cells(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    tree: _bvh.Tree,
    jobs: wp.array[int],
    first: wp.array[int],
    job_count: wp.array[int],
    bounds: wp.array[wp.vec3],
    scratch: Scratch64,
    status: wp.array[int],
    candidates: wp.array2d[int],
    candidate_count: wp.array[int],
):
    slot = wp.tid()
    job = first[0] + slot
    if job >= job_count[0]:
        return
    i = jobs[job]
    if status[i] == 0 or status[i] == 3:
        return
    if tree.owners[i] != i:
        status[i] = 3
        return
    cell = init_symbolic(scratch, slot)
    nodes = wp.zeros(shape=128, dtype=int)
    nodes[0] = 1
    size = int(1)
    while size > 0 and cell.status == 0:
        size -= 1
        node = nodes[size]
        if tree.count[node] > 0:
            for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                j = tree.ids[k]
                if tree.owners[j] == j:
                    cell = clip_symbolic(scratch, cell, i, j, points, bounds)
                    if cell.status != 0:
                        break
        else:
            for side in range(2):
                child = tree.left[node]
                if side == 1:
                    child = tree.right[node]
                if size == 128:
                    cell.status = 4
                    break
                nodes[size] = child
                size += 1
    status[i] = cell.status
    if cell.status != 0:
        return
    # A true finite dual vertex is incident to three site bisectors. Box
    # planes have negative neighbor IDs and never become tetrahedron vertices.
    for k in range(cell.nv):
        v = scratch.vertices[cell.base + 32 * k]
        a = scratch.neighbors[cell.base + 32 * plane_at(v, 0)]
        b = scratch.neighbors[cell.base + 32 * plane_at(v, 1)]
        c = scratch.neighbors[cell.base + 32 * plane_at(v, 2)]
        if i < a and i < b and i < c:
            tet = sorted4(wp.vec4i(i, a, b, c))
            at = wp.atomic_add(candidate_count, 0, 1)
            if at < candidates.shape[0]:
                for j in range(4):
                    candidates[at, j] = tet[j]
