"""Spatial execution ordering for large private mesher surface-query batches.

Morton centroids select a permutation only. Original scalar query arithmetic,
cell IDs, outputs, cache validity and error handling are retained. Small caches
use the original launch path and own no ordering storage.
"""

import warp as wp
from warp.utils import radix_sort_pairs

from . import _surface_query_cache as original
from ._bvh import expand_bits
from ._surface_bvh import SurfaceTree
from ._surface_query_cache import (
    QueryState,
    cached_nearest,
    cached_side,
    selection_center_cached,
)
from ._surface_query_cache import (
    SurfaceQueryCache as BaseCache,
)
from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def centroid_keys(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    keys: wp.array[wp.uint64],
    ids: wp.array[int],
):
    i = wp.tid()
    ids[i] = i
    keys[i] = wp.uint64(18446744073709551615)
    if error[0] != 0 or tree.n <= 0 or i >= count[0] or (i >= tets.shape[0]):
        return
    center = wp.vec3d(0.0)
    for k in range(4):
        site = tets[i, k]
        if site < 0 or site >= points.shape[0]:
            return
        p = points[site]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            return
        center += wp.vec3d(p) * wp.float64(0.25)
    lower = wp.vec3d(tree.lower[1])
    upper = wp.vec3d(tree.upper[1])
    q = wp.vec3d(0.0)
    for k in range(3):
        if not wp.isfinite(lower[k]) or not wp.isfinite(upper[k]):
            return
        extent = upper[k] - lower[k]
        if extent > wp.float64(0.0):
            q[k] = wp.clamp(
                (center[k] - lower[k]) / extent, wp.float64(0.0), wp.float64(1.0)
            ) * wp.float64(2097151.0)
    keys[i] = (
        expand_bits(wp.uint64(q[0]))
        | expand_bits(wp.uint64(q[1])) << wp.uint64(1)
        | expand_bits(wp.uint64(q[2])) << wp.uint64(2)
    )


@wp.kernel
def classify_cached_ordered(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    centers: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    keep: wp.array[int],
    volumes: wp.array[wp.float64],
    quality: wp.array[wp.float64],
    probe: QueryState,
    previous: QueryState,
    table: wp.array[int],
    order: wp.array[int],
):
    i = order[wp.tid()]
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    (a, b, c) = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.nearest[i] = 1
    probe.limits[i] = wp.float64(1e150)
    hit = cached_nearest(tree, center, wp.float64(1e150), previous, table)
    probe.distances[i] = hit.distance2
    probe.faces[i] = hit.face
    probe.hit_points[i] = hit.point
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        probe.rays[i] = 1
        side = cached_side(tree, center, previous, table)
        probe.sides[i] = side
    if side != 0 and side != 1:
        wp.atomic_or(error, 0, 16)
        return
    distance = wp.sqrt(hit.distance2)
    centers[i] = center
    radii[i] = radius
    distances[i] = wp.where(side == 1, -distance, distance)
    keep[i] = int(side == 1 or distance / radius < tolerance)
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))


@wp.kernel
def selection_cached_ordered(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    centers: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    keep: wp.array[int],
    volumes: wp.array[wp.float64],
    quality: wp.array[wp.float64],
    probe: QueryState,
    previous: QueryState,
    table: wp.array[int],
    order: wp.array[int],
):
    i = order[wp.tid()]
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    (a, b, c) = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    status = selection_center_cached(tree, center, radius, tolerance, probe, i, previous, table)
    if status.error != 0:
        wp.atomic_or(error, 0, status.error)
        return
    distances[i] = status.distance_sign
    keep[i] = status.retained
    centers[i] = center
    radii[i] = radius
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))


@wp.kernel
def retained_cached_ordered(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    keep: wp.array[int],
    quality: wp.array[wp.float64],
    probe: QueryState,
    previous: QueryState,
    table: wp.array[int],
    order: wp.array[int],
):
    i = order[wp.tid()]
    keep[i] = 0
    probe.valid[i] = 0
    probe.rays[i] = 0
    probe.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or (not wp.isfinite(p[2])):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a = wp.vec3d(points[ids[1]]) - p0
    b = wp.vec3d(points[ids[2]]) - p0
    c = wp.vec3d(points[ids[3]]) - p0
    (bc, ca, ab) = (wp.cross(b, c), wp.cross(c, a), wp.cross(a, b))
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    (center, radius) = (wp.vec3d(), wp.float64(0.0))
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        (center, radius) = (p0 + offset, wp.length(offset))
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    probe.centers[i] = center
    probe.valid[i] = 1
    probe.rays[i] = 1
    side = cached_side(tree, center, previous, table)
    probe.sides[i] = side
    retained = side == 1
    if not retained or radius >= wp.float64(1e140):
        limit = wp.float64(1e150)
        bounded = side == 0 and tolerance > wp.float64(0.0) and (radius < wp.float64(1e140))
        if bounded:
            limit = wp.min(
                limit,
                wp.max(
                    wp.float64(1e-150), tolerance * radius * (wp.float64(1.0) + wp.float64(1e-12))
                ),
            )
        probe.nearest[i] = 1
        probe.limits[i] = limit
        hit = cached_nearest(tree, center, limit, previous, table)
        probe.distances[i] = hit.distance2
        probe.faces[i] = hit.face
        probe.hit_points[i] = hit.point
        if hit.face < 0:
            if not bounded or limit == wp.float64(1e150):
                wp.atomic_or(error, 0, 16)
                return
        else:
            if not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)
                return
            if hit.distance2 == wp.float64(0.0):
                retained = True
            elif side != 0 and side != 1:
                wp.atomic_or(error, 0, 16)
                return
            else:
                retained = side == 1 or wp.sqrt(hit.distance2) / radius < tolerance
    keep[i] = int(retained)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))


@wp.kernel
def choose_order(count: wp.array[int], error: wp.array[int], minimum: int, enabled: wp.array[int]):
    enabled[0] = int(error[0] == 0 and count[0] >= minimum)


ORDERED = {
    original.classify_cached: classify_cached_ordered,
    original.selection_cached: selection_cached_ordered,
    original.retained_cached: retained_cached_ordered,
}


class OrderedSurfaceQueryCache(BaseCache):
    """Keep sorting optional and all live batch decisions on device."""

    _minimum_query_count = 32768

    def __init__(self, domain, capacity):
        self.minimum_query_count = self._minimum_query_count
        self._has_order_storage = capacity > self.minimum_query_count
        super().__init__(domain, capacity)
        self.order_keys = self.order_ids = self.order_enabled = None
        if self._has_order_storage:
            with wp.ScopedDevice(self.device):
                self.order_keys = wp.zeros(2 * capacity, dtype=wp.uint64)
                self.order_ids = wp.zeros(2 * capacity, dtype=int)
                self.order_enabled = wp.ones(1, dtype=int)
                for block in [32, 256]:
                    wp.load_module(module=__name__, device=self.device, block_dim=block)
                # Initialize the sort outside capture; each conditional capture
                # then owns its retained CUB scratch through Warp's child graph.
                self._sort()
            wp.synchronize_device(self.device)

    def _sort(self):
        radix_sort_pairs(self.order_keys, self.order_ids, self.capacity, end_bit=64)

    def _ordered(self, kernel, dim, args, block_dim):
        wp.launch(
            centroid_keys,
            self.capacity,
            inputs=[args[0], args[1], args[2], args[3], args[5], self.order_keys, self.order_ids],
            device=self.device,
            block_dim=256,
        )
        self._sort()
        wp.launch(
            ORDERED[kernel],
            dim,
            inputs=[*args, self.current, self.previous, self.table, self.order_ids],
            device=self.device,
            block_dim=block_dim,
        )
        self.after(args[5])

    def launch(self, kernel, dim, args, block_dim=32):
        if not self._has_order_storage:
            return BaseCache.launch(self, kernel, dim, args, block_dim)
        assert kernel in ORDERED and dim == self.capacity
        if wp.get_stream(self.device).is_capturing:
            wp.launch(
                choose_order,
                1,
                inputs=[args[3], args[5], self.minimum_query_count, self.order_enabled],
                device=self.device,
                block_dim=32,
            )
            wp.capture_if(
                self.order_enabled,
                lambda: self._ordered(kernel, dim, args, block_dim),
                lambda: BaseCache.launch(self, kernel, dim, args, block_dim),
            )
        else:
            self._ordered(kernel, dim, args, block_dim)
