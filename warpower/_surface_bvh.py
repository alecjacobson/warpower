"""Preallocated triangle hierarchy and pure-Warp closest/oriented-ray queries."""

import warp as wp
from warp.utils import radix_sort_pairs

from . import _bvh

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class SurfaceTree:
    points: wp.array[wp.vec3]
    indices: wp.array[int]
    ids: wp.array[int]
    lower: wp.array[wp.vec3]
    upper: wp.array[wp.vec3]
    left: wp.array[int]
    right: wp.array[int]
    escape: wp.array[int]
    leaf_base: int
    n: int


@wp.kernel
def centroids(tree: SurfaceTree, output: wp.array[wp.vec3]):
    f = wp.tid()
    a = wp.vec3d(tree.points[tree.indices[3 * f]])
    b = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
    c = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
    output[f] = wp.vec3((a + b + c) / wp.float64(3.0))


@wp.kernel
def leaves(tree: SurfaceTree):
    leaf = wp.tid()
    lo, hi = wp.vec3(3.4028234e38), wp.vec3(-3.4028234e38)
    for j in range(4 * leaf, wp.min(4 * leaf + 4, tree.n)):
        f = tree.ids[j]
        for k in range(3):
            p = tree.points[tree.indices[3 * f + k]]
            lo, hi = wp.min(lo, p), wp.max(hi, p)
    tree.lower[tree.leaf_base + leaf] = lo
    tree.upper[tree.leaf_base + leaf] = hi


@wp.kernel
def merge(tree: SurfaceTree, first: int):
    i = first + wp.tid()
    tree.lower[i] = wp.min(tree.lower[2 * i], tree.lower[2 * i + 1])
    tree.upper[i] = wp.max(tree.upper[2 * i], tree.upper[2 * i + 1])


@wp.kernel
def group_keys(sorted_keys: wp.array[wp.uint64], keys: wp.array[wp.uint64]):
    i = wp.tid()
    keys[i] = sorted_keys[4 * i]


@wp.kernel
def radix_topology(
    tree: SurfaceTree,
    keys: wp.array[wp.uint64],
    firsts: wp.array[int],
    ends: wp.array[int],
    parents: wp.array[int],
):
    # Binary radix ranges over four-triangle leaf groups. Equal Morton keys
    # are disambiguated by their sorted indices in _bvh.prefix.
    node = wp.tid() + 1
    n = tree.leaf_base
    if node >= n:
        firsts[node] = node - n
        ends[node] = node - n + 1
        tree.left[node] = 0
        tree.right[node] = 0
        return
    i = node - 1
    direction = int(1)
    if _bvh.prefix(keys, i, i + 1, n) < _bvh.prefix(keys, i, i - 1, n):
        direction = -1
    minimum = _bvh.prefix(keys, i, i - direction, n)
    limit = int(2)
    while _bvh.prefix(keys, i, i + limit * direction, n) > minimum:
        limit *= 2
    length, step = int(0), limit // 2
    while step >= 1:
        if _bvh.prefix(keys, i, i + (length + step) * direction, n) > minimum:
            length += step
        step //= 2
    other = i + length * direction
    first, last = wp.min(i, other), wp.max(i, other)
    common = _bvh.prefix(keys, first, last, n)
    # Separate assignments keep the mutable split independent of first in Warp.
    split = first
    step = last - first
    while step > 1:
        step = (step + 1) // 2
        candidate = split + step
        if candidate < last and _bvh.prefix(keys, first, candidate, n) > common:
            split = candidate
    left, right = split + 1, split + 2
    if split == first:
        left = n + split
    if split + 1 == last:
        right = n + split + 1
    tree.left[node] = left
    tree.right[node] = right
    parents[left] = node
    parents[right] = node
    firsts[node] = first
    ends[node] = last + 1


@wp.kernel
def escape_links(tree: SurfaceTree, parents: wp.array[int]):
    # Next subtree in right-first depth-first order. Zero terminates traversal.
    node = wp.tid() + 1
    at = node
    parent = parents[at]
    next_node = int(0)
    while parent > 0:
        if at == tree.right[parent]:
            next_node = tree.left[parent]
            break
        at = parent
        parent = parents[at]
    tree.escape[node] = next_node


@wp.kernel
def radix_bounds(
    tree: SurfaceTree,
    segment: SurfaceTree,
    firsts: wp.array[int],
    ends: wp.array[int],
):
    # Gather exact float32 min/max from a balanced segment tree. Independent
    # node writes avoid inter-block publication/atomic ordering during refit.
    node = wp.tid() + 1
    a, b = firsts[node] + segment.leaf_base, ends[node] + segment.leaf_base
    lo, hi = wp.vec3(3.4028234e38), wp.vec3(-3.4028234e38)
    while a < b:
        if (a & 1) != 0:
            lo, hi = wp.min(lo, segment.lower[a]), wp.max(hi, segment.upper[a])
            a += 1
        if (b & 1) != 0:
            b -= 1
            lo, hi = wp.min(lo, segment.lower[b]), wp.max(hi, segment.upper[b])
        a //= 2
        b //= 2
    tree.lower[node] = lo
    tree.upper[node] = hi


@wp.func(inline=True)
def box_distance(tree: SurfaceTree, node: int, p: wp.vec3d):
    d = wp.max(
        wp.max(wp.vec3d(tree.lower[node]) - p, p - wp.vec3d(tree.upper[node])), wp.vec3d(0.0)
    )
    return wp.dot(d, d)


@wp.func(inline=True)
def closest_triangle(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # Voronoi regions of a triangle; all products use double precision.
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
        return a
    bp = p - b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= wp.float64(0.0) and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= wp.float64(0.0) and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= wp.float64(0.0) and d4 - d3 >= wp.float64(0.0) and d5 - d6 >= wp.float64(0.0):
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    inverse = wp.float64(1.0) / (va + vb + vc)
    return a + (vb * inverse) * ab + (vc * inverse) * ac


@wp.struct
class Closest:
    point: wp.vec3d
    distance2: wp.float64
    face: int


@wp.func(inline=True)
def closest_point(tree: SurfaceTree, p: wp.vec3d, limit: wp.float64):
    result = Closest()
    result.point = wp.vec3d(0.0)
    result.face = -1
    result.distance2 = limit * limit
    # 63 Morton bits plus at most 31 index bits bound the radix depth below 96.
    stack = wp.zeros(shape=128, dtype=int)
    # Child distances are already computed for ordering. The point and tree
    # stay fixed during this query, so retain those rounded scalar values and
    # compare against the current best distance when each node is popped.
    bounds = wp.zeros(shape=128, dtype=wp.float64)
    stack[0] = 1
    bounds[0] = box_distance(tree, 1, p)
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        if bounds[size] >= result.distance2:
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                f = tree.ids[j]
                a = wp.vec3d(tree.points[tree.indices[3 * f]])
                b = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
                c = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
                q = closest_triangle(p, a, b, c)
                distance = wp.length_sq(p - q)
                if distance < result.distance2:
                    result.point = q
                    result.face = f
                    result.distance2 = distance
        else:
            left, right = int(tree.left[node]), int(tree.right[node])
            left_distance = box_distance(tree, left, p)
            right_distance = box_distance(tree, right, p)
            if left_distance < right_distance:
                left, right = right, left
                left_distance, right_distance = right_distance, left_distance
            # Keep the original rejection comparison, including unordered NaNs.
            # The best distance only decreases; a rejected child cannot revive.
            if not (left_distance >= result.distance2):
                stack[size] = left
                bounds[size] = left_distance
                size += 1
            if not (right_distance >= result.distance2):
                stack[size] = right
                bounds[size] = right_distance
                size += 1
    return result


@wp.func(inline=True)
def ray_box(tree: SurfaceTree, node: int, origin: wp.vec3d, inverse: wp.vec3d):
    a = wp.cw_mul(wp.vec3d(tree.lower[node]) - origin, inverse)
    b = wp.cw_mul(wp.vec3d(tree.upper[node]) - origin, inverse)
    lo, hi = wp.min(a, b), wp.max(a, b)
    return wp.min(hi[0], wp.min(hi[1], hi[2])) >= wp.max(
        wp.float64(0.0), wp.max(lo[0], wp.max(lo[1], lo[2]))
    )


@wp.func(inline=True)
def ray_count(tree: SurfaceTree, origin: wp.vec3d, direction: wp.vec3d, excluded_face: int = -1):
    # Return signed intersections and an ambiguity flag. Shared-edge hits are
    # retried with another direction instead of double-counting adjacent faces.
    crossings, ambiguous = int(0), int(0)
    if excluded_face >= 0:
        a = wp.vec3d(tree.points[tree.indices[3 * excluded_face]])
        b = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 1]]) - a
        c = wp.vec3d(tree.points[tree.indices[3 * excluded_face + 2]]) - a
        crossings = -int(wp.dot(direction, wp.cross(b, c)) < wp.float64(0.0))
    inverse = wp.vec3d(
        wp.float64(1.0) / direction[0],
        wp.float64(1.0) / direction[1],
        wp.float64(1.0) / direction[2],
    )
    node = int(1)
    while node > 0:
        if not ray_box(tree, node, origin, inverse):
            node = tree.escape[node]
            continue
        if node >= tree.leaf_base:
            first = 4 * (node - tree.leaf_base)
            for j in range(first, wp.min(first + 4, tree.n)):
                f = tree.ids[j]
                if f == excluded_face:
                    continue
                a = wp.vec3d(tree.points[tree.indices[3 * f]])
                b = wp.vec3d(tree.points[tree.indices[3 * f + 1]]) - a
                c = wp.vec3d(tree.points[tree.indices[3 * f + 2]]) - a
                normal = wp.cross(b, c)
                det = wp.dot(direction, normal)
                scale = wp.length(normal) * wp.length(direction)
                if wp.abs(det) <= wp.float64(1.0e-14) * scale:
                    continue
                delta = origin - a
                t = -wp.dot(delta, normal) / det
                if t <= wp.float64(0.0):
                    continue
                q = delta + t * direction
                inverse_area = wp.float64(1.0) / wp.dot(normal, normal)
                u = wp.dot(wp.cross(q, c), normal) * inverse_area
                v = wp.dot(wp.cross(b, q), normal) * inverse_area
                tolerance = wp.float64(2.0e-12)
                if u >= -tolerance and v >= -tolerance and u + v <= wp.float64(1.0) + tolerance:
                    if u <= tolerance or v <= tolerance or u + v >= wp.float64(1.0) - tolerance:
                        ambiguous = 1
                    crossings += wp.where(det > wp.float64(0.0), 1, -1)
            node = tree.escape[node]
        else:
            node = tree.right[node]
    return wp.vec2i(crossings, ambiguous)


@wp.func(inline=True)
def material_side(tree: SurfaceTree, p: wp.vec3d, excluded_face: int = -1):
    # A generic ray suffices off triangle boundaries. Three fixed, non-axis
    # directions are available for shared-edge/vertex ties, all on device.
    result = ray_count(
        tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319), excluded_face
    )
    if result[1] != 0:
        result = ray_count(
            tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319), excluded_face
        )
    if result[1] != 0:
        result = ray_count(
            tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121), excluded_face
        )
    if result[1] != 0:
        return 128
    return result[0]


class SurfaceHierarchy:
    def __init__(self, points, indices):
        n = len(indices) // 3
        self._segment = tree = SurfaceTree()
        tree.points, tree.indices, tree.n = points, indices, n
        tree.leaf_base = 1 << (((n + 3) // 4 - 1).bit_length())
        device = points.device
        tree.ids = wp.empty(2 * n, dtype=int, device=device)
        tree.lower = wp.empty(2 * tree.leaf_base, dtype=wp.vec3, device=device)
        tree.upper = wp.empty_like(tree.lower)
        centers = wp.empty(n, dtype=wp.vec3, device=device)
        keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
        groups = (n + 255) // 256
        low = wp.empty(groups, dtype=wp.vec3, device=device)
        high = wp.empty_like(low)
        bounds = wp.array([[1e30] * 3, [-1e30] * 3], dtype=wp.vec3, device=device)
        wp.launch(centroids, n, inputs=[tree, centers], device=device)
        wp.launch_tiled(
            _bvh.block_bounds, dim=groups, inputs=[centers, low, high], block_dim=256, device=device
        )
        wp.launch(_bvh.merge_bounds, groups, inputs=[low, high, bounds], device=device)
        wp.launch(_bvh.morton_keys, n, inputs=[centers, bounds, keys, tree.ids], device=device)
        radix_sort_pairs(keys, tree.ids, n, end_bit=63)
        self._setup_buffers = centers, keys, low, high, bounds
        self.tree = radix = SurfaceTree()
        radix.points, radix.indices, radix.ids, radix.n = points, indices, tree.ids, n
        radix.leaf_base = count = (n + 3) // 4
        radix.lower = wp.empty(2 * count, dtype=wp.vec3, device=device)
        radix.upper = wp.empty_like(radix.lower)
        radix.left = wp.empty(2 * count, dtype=int, device=device)
        radix.right = wp.empty_like(radix.left)
        radix.escape = wp.empty_like(radix.left)
        self._firsts = wp.empty_like(radix.left)
        self._ends = wp.empty_like(radix.left)
        self._parents = wp.zeros_like(radix.left)
        self._group_keys = wp.empty(count, dtype=wp.uint64, device=device)
        wp.launch(group_keys, count, inputs=[keys, self._group_keys], device=device)
        wp.launch(
            radix_topology,
            2 * count - 1,
            inputs=[radix, self._group_keys, self._firsts, self._ends, self._parents],
            device=device,
        )
        wp.launch(escape_links, 2 * count - 1, inputs=[radix, self._parents], device=device)
        self.refit()

    def refit(self):
        tree = self._segment
        wp.launch(leaves, tree.leaf_base, inputs=[tree], device=tree.points.device)
        first = tree.leaf_base // 2
        while first:
            wp.launch(merge, first, inputs=[tree, first], device=tree.points.device)
            first //= 2
        wp.launch(
            radix_bounds,
            2 * self.tree.leaf_base - 1,
            inputs=[self.tree, tree, self._firsts, self._ends],
            device=tree.points.device,
        )
