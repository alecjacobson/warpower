"""Device-resident triangle domains, geometric queries, and boundary quadrature."""

import math
import operator

import numpy as np
import warp as wp

from ._api import _points_array
from ._graph import scan_add, scan_blocks
from ._surface_bvh import SurfaceHierarchy, SurfaceTree, closest_point, material_side
from ._triangle_contact import check_intersections
from ._triangle_contact import excess_contact as excess_contact

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def next_edge(e: int):
    return 3 * (e // 3) + (e + 1) % 3


@wp.func
def previous_edge(e: int):
    return 3 * (e // 3) + (e + 2) % 3


@wp.kernel
def advance_query_epoch(version: wp.array[wp.int64]):
    # Saturation disables cache lookup permanently; an old generation never wraps.
    if version[0] < wp.int64(9223372036854775807):
        version[0] += wp.int64(1)


@wp.kernel
def safe_faces(source: wp.array[int], indices: wp.array[int], n: int, error: wp.array[int]):
    e = wp.tid()
    v = source[e]
    if v < 0 or v >= n:
        wp.atomic_or(error, 0, 1)
        v = 0
    indices[e] = v


@wp.kernel
def safe_points(source: wp.array[wp.vec3], points: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    p = source[i]
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 2)
        p = wp.vec3(0.0)
    points[i] = p


@wp.kernel
def pair_edges(
    indices: wp.array[int],
    table: wp.array[int],
    slots: wp.array[int],
    counts: wp.array[int],
    balance: wp.array[int],
    minimum: wp.array[int],
    maximum: wp.array[int],
    first: wp.array[int],
    degree: wp.array[int],
):
    e = wp.tid()
    a, b = indices[e], indices[next_edge(e)]
    lo, hi = wp.min(a, b), wp.max(a, b)
    slot = int(
        (wp.uint32(lo) * wp.uint32(73856093) ^ wp.uint32(hi) * wp.uint32(19349663))
        & wp.uint32(table.shape[0] - 1)
    )
    while True:
        other = wp.atomic_cas(table, slot, -1, e)
        if other == -1:
            break
        c, d = indices[other], indices[next_edge(other)]
        if wp.min(c, d) == lo and wp.max(c, d) == hi:
            break
        slot = (slot + 1) & (table.shape[0] - 1)
    slots[e] = slot
    wp.atomic_add(counts, slot, 1)
    wp.atomic_add(balance, slot, wp.where(a < b, 1, -1))
    wp.atomic_min(minimum, slot, e)
    wp.atomic_max(maximum, slot, e)
    wp.atomic_min(first, a, e)
    wp.atomic_add(degree, a, 1)


@wp.kernel
def validate_edges(
    slots: wp.array[int],
    counts: wp.array[int],
    balance: wp.array[int],
    minimum: wp.array[int],
    maximum: wp.array[int],
    twins: wp.array[int],
    error: wp.array[int],
):
    e = wp.tid()
    slot = slots[e]
    twins[e] = -1
    if counts[slot] != 2 or balance[slot] != 0:
        wp.atomic_or(error, 0, 8)
    else:
        twins[e] = wp.where(e == minimum[slot], maximum[slot], minimum[slot])


@wp.kernel
def validate_vertex_links(
    first: wp.array[int],
    degree: wp.array[int],
    twins: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0 or degree[i] == 0:
        return
    e = first[i]
    visited = int(0)
    for _ in range(degree[i]):
        visited += 1
        e = twins[previous_edge(e)]
        if e == first[i]:
            break
    if visited != degree[i] or e != first[i]:
        wp.atomic_or(error, 0, 8)


@wp.kernel
def face_geometry(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float64],
    error: wp.array[int],
):
    f = wp.tid()
    a = wp.vec3d(points[indices[3 * f]])
    b = wp.vec3d(points[indices[3 * f + 1]]) - a
    c = wp.vec3d(points[indices[3 * f + 2]]) - a
    cross = wp.cross(b, c)
    norm = wp.length(cross)
    # Reject collapse relative to the local scale before any division.
    scale = wp.dot(b, b) + wp.dot(c, c) + wp.dot(b - c, b - c)
    valid = norm > wp.float64(1.0e-12) * scale and wp.isfinite(norm)
    normals[f] = wp.vec3(0.0)
    areas[f] = wp.float64(0.0)
    if not valid:
        wp.atomic_or(error, 0, 4)
    else:
        normals[f] = wp.vec3(cross / norm)
        areas[f] = norm * wp.float64(0.5)


@wp.kernel
def check_orientation(tree: SurfaceTree, error: wp.array[int]):
    f = wp.tid()
    if error[0] != 0:
        return
    a = wp.vec3d(tree.points[tree.indices[3 * f]])
    b = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
    c = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
    center = a + ((b - a) + (c - a)) / wp.float64(3.0)
    # Exclude this face from the ray, then subtract the crossing expected on
    # its material side. This avoids an offset that could jump a thin gap.
    if material_side(tree, center, f) != 0:
        wp.atomic_or(error, 0, 32)


@wp.kernel
def feature_flags(
    twins: wp.array[int],
    normals: wp.array[wp.vec3],
    cosine: float,
    error: wp.array[int],
    flags: wp.array[int],
):
    e = wp.tid()
    flags[e] = 0
    other = twins[e]
    if error[0] == 0 and other > e:
        if wp.dot(normals[e // 3], normals[other // 3]) < cosine:
            flags[e] = 1


@wp.kernel
def compact_features(
    indices: wp.array[int],
    flags: wp.array[int],
    prefix: wp.array[int],
    edges: wp.array[wp.vec2i],
    count: wp.array[int],
    degree: wp.array[int],
    direction: wp.array[wp.vec3d],
    points: wp.array[wp.vec3],
):
    e = wp.tid()
    if e == flags.shape[0] - 1:
        count[0] = prefix[e]
    if flags[e] != 0:
        a, b = indices[e], indices[next_edge(e)]
        edges[prefix[e] - 1] = wp.vec2i(a, b)
        d = wp.normalize(wp.vec3d(points[b]) - wp.vec3d(points[a]))
        wp.atomic_add(degree, a, 1)
        wp.atomic_add(degree, b, 1)
        wp.atomic_add(direction, a, d)
        wp.atomic_add(direction, b, -d)


@wp.kernel
def corner_flags(
    degree: wp.array[int],
    direction: wp.array[wp.vec3d],
    cosine: float,
    flags: wp.array[int],
):
    i = wp.tid()
    flags[i] = 0
    if degree[i] > 0:
        # For two outward unit edge directions, |u+v|^2 = 2 + 2 dot(u,v).
        if degree[i] != 2 or wp.dot(direction[i], direction[i]) > wp.float64(2.0 * (1.0 - cosine)):
            flags[i] = 1


@wp.kernel
def compact_corners(
    points: wp.array[wp.vec3],
    flags: wp.array[int],
    prefix: wp.array[int],
    ids: wp.array[int],
    corners: wp.array[wp.vec3],
    count: wp.array[int],
):
    i = wp.tid()
    if i == flags.shape[0] - 1:
        count[0] = prefix[i]
    if flags[i] != 0:
        ids[prefix[i] - 1] = i
        corners[prefix[i] - 1] = points[i]


@wp.kernel
def surface_quadrature(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    areas: wp.array[wp.float64],
    subdivisions: int,
    error: wp.array[int],
    samples: wp.array[wp.vec3],
    weights: wp.array[wp.float64],
):
    i = wp.tid()
    weights[i] = wp.float64(0.0)
    samples[i] = wp.vec3(0.0)
    if error[0] != 0:
        return
    per_face = 3 * subdivisions * subdivisions
    f, local = i // per_face, i % per_face
    k, node = local // 3, local % 3
    row = int(wp.sqrt(float(k)))
    # Correct float sqrt rounding for large subdivision counts.
    if (row + 1) * (row + 1) <= k:
        row += 1
    if row * row > k:
        row -= 1
    offset = k - row * row
    u, v = subdivisions - 1 - row, offset // 2
    a, b, c = (
        wp.vec2d(wp.float64(u), wp.float64(v)),
        wp.vec2d(wp.float64(u + 1), wp.float64(v)),
        wp.vec2d(wp.float64(u), wp.float64(v + 1)),
    )
    if offset % 2 == 1:
        a, b, c = b, wp.vec2d(wp.float64(u + 1), wp.float64(v + 1)), c
    uv = (a + b + c) / wp.float64(6.0)
    if node == 0:
        uv += a * wp.float64(0.5)
    elif node == 1:
        uv += b * wp.float64(0.5)
    else:
        uv += c * wp.float64(0.5)
    uv /= wp.float64(subdivisions)
    p, q, r = (
        wp.vec3d(points[indices[3 * f]]),
        wp.vec3d(points[indices[3 * f + 1]]),
        wp.vec3d(points[indices[3 * f + 2]]),
    )
    samples[i] = wp.vec3(p + uv[0] * (q - p) + uv[1] * (r - p))
    weights[i] = areas[f] / wp.float64(per_face)


@wp.kernel
def crease_quadrature(
    points: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    count: wp.array[int],
    subdivisions: int,
    samples: wp.array[wp.vec3],
    weights: wp.array[wp.float64],
):
    i = wp.tid()
    weights[i] = wp.float64(0.0)
    samples[i] = wp.vec3(0.0)
    e, local = i // (2 * subdivisions), i % (2 * subdivisions)
    if e >= count[0]:
        return
    ids = edges[e]
    p, q = wp.vec3d(points[ids[0]]), wp.vec3d(points[ids[1]])
    node = wp.float64(0.5) + wp.float64(0.28867513459481288225) * wp.float64(2 * (local % 2) - 1)
    t = (wp.float64(local // 2) + node) / wp.float64(subdivisions)
    samples[i] = wp.vec3(p + t * (q - p))
    weights[i] = wp.length(q - p) / wp.float64(2 * subdivisions)


@wp.kernel
def domain_queries(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    domain_error: wp.array[int],
    max_distance: float,
    closest: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    distance: wp.array[float],
    inside: wp.array[int],
    faces: wp.array[int],
    valid: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    closest[i] = wp.vec3(0.0)
    normal[i] = wp.vec3(0.0)
    distance[i] = max_distance
    inside[i] = 0
    faces[i] = -1
    valid[i] = 0
    if domain_error[0] != 0:
        wp.atomic_or(error, 0, domain_error[0])
        return
    p = wp.vec3d(points[i])
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 64)
        return
    query = closest_point(tree, p, wp.float64(max_distance))
    if query.face >= 0:
        side = material_side(tree, p)
        if side != 0 and side != 1:
            wp.atomic_or(error, 0, 128)
            return
        a = wp.vec3d(tree.points[tree.indices[3 * query.face]])
        b = wp.vec3d(tree.points[tree.indices[3 * query.face + 1]])
        c = wp.vec3d(tree.points[tree.indices[3 * query.face + 2]])
        closest[i] = wp.vec3(query.point)
        normal[i] = wp.vec3(wp.normalize(wp.cross(b - a, c - a)))
        distance[i] = float(wp.sqrt(query.distance2)) * float(1 - 2 * side)
        inside[i] = side
        faces[i] = query.face
        valid[i] = 1


class _Scan:
    """Preallocated inclusive scan: no CUB scratch allocation in capture."""

    def __init__(self, n, device):
        self.levels = []
        while True:
            blocks = (n + 255) // 256
            self.levels.append(
                (
                    wp.empty(blocks, dtype=int, device=device),
                    wp.empty(blocks, dtype=int, device=device),
                )
            )
            if blocks == 1:
                break
            n = blocks

    def run(self, values, output, level=0):
        sums, scanned = self.levels[level]
        wp.launch_tiled(
            scan_blocks,
            dim=len(sums),
            inputs=[values, output, sums],
            block_dim=256,
            device=values.device,
        )
        if len(sums) > 1:
            self.run(sums, scanned, level + 1)
            wp.launch(scan_add, len(output), inputs=[output, scanned], device=values.device)


def _contiguous_points(points):
    if hasattr(points, "is_contiguous"):
        contiguous = points.is_contiguous
        if callable(contiguous):
            contiguous = contiguous()
        if not contiguous:
            raise ValueError("Domain workspace points must be contiguous")
    result, _ = _points_array(points)
    if not result.device.is_cuda:
        raise ValueError("Domain workspaces require CUDA points")
    return result


class SurfaceDomain:
    """Closed, oriented triangle boundary with capturable geometry updates.

    Topology is copied and validated on the device at setup. Vertex positions can
    be produced in-place by another CUDA process; call ``refit()`` after changes.
    Material must be on the back side of every face (cavity shells face inward).
    """

    def __init__(
        self,
        points,
        triangles,
        *,
        surface_subdivisions=1,
        crease_subdivisions=4,
        feature_angle=45.0,
        corner_angle=45.0,
        validate_geometry=True,
        query_backend="reference",
    ):
        if query_backend not in ("reference", "native"):
            raise ValueError("query_backend must be 'reference' or 'native'")
        self.query_backend = query_backend
        self._native_mesh = None
        self._native_queries = None
        self.points = _contiguous_points(points)
        self.device = self.points.device
        self._owners = (points, triangles)
        self.surface_subdivisions = operator.index(surface_subdivisions)
        self.crease_subdivisions = operator.index(crease_subdivisions)
        if min(self.surface_subdivisions, self.crease_subdivisions) < 1:
            raise ValueError("Quadrature subdivisions must be positive")
        if not 0 < feature_angle < 180 or not 0 < corner_angle < 180:
            raise ValueError("Feature and corner angles must lie between 0 and 180 degrees")
        self._feature_cos = math.cos(math.radians(feature_angle))
        self._corner_cos = math.cos(math.radians(corner_angle))
        self.validate_geometry = bool(validate_geometry)
        if isinstance(triangles, np.ndarray):
            triangles = wp.array(triangles, device=self.device)
        elif type(triangles).__module__.startswith("torch"):
            triangles = wp.from_torch(triangles)
        if not isinstance(triangles, wp.array) or triangles.dtype != wp.int32:
            raise ValueError("Triangles must have dtype int32")
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or triangles.device != self.device
            or not triangles.is_contiguous
        ):
            raise ValueError("Triangles must be contiguous (F, 3) on the points device")
        self.face_count = len(triangles)
        n, m = len(self.points), 3 * self.face_count
        if n < 4 or self.face_count < 4:
            raise ValueError("A closed domain needs at least four points and four triangles")
        samples = m * self.surface_subdivisions**2
        crease_samples = 2 * m * self.crease_subdivisions
        if max(n, 8 * m, samples, crease_samples) >= 2**31:
            raise ValueError("Domain capacities exceed int32 indexing")
        with wp.ScopedDevice(self.device):
            self._topology_error = wp.zeros(1, dtype=int)
            self._query_epoch = wp.zeros(1, dtype=wp.int64)
            self.error = wp.zeros(1, dtype=int)
            self._points = wp.empty_like(self.points)
            self._indices = wp.empty(m, dtype=int)
            self._twins = wp.empty(m, dtype=int)
            self._normals = wp.empty(self.face_count, dtype=wp.vec3)
            self._areas = wp.empty(self.face_count, dtype=wp.float64)
            table_size = 1 << (2 * m - 1).bit_length()
            table = wp.full(table_size, -1, dtype=int)
            slots = wp.empty(m, dtype=int)
            counts = wp.zeros(table_size, dtype=int)
            balance = wp.zeros_like(counts)
            minimum = wp.full(table_size, 2**31 - 1, dtype=int)
            maximum = wp.full(table_size, -1, dtype=int)
            first = wp.full(n, 2**31 - 1, dtype=int)
            degree = wp.zeros(n, dtype=int)
            wp.launch(
                safe_faces,
                m,
                inputs=[triangles.reshape((m,)), self._indices, n, self._topology_error],
            )
            wp.launch(
                pair_edges,
                m,
                inputs=[
                    self._indices,
                    table,
                    slots,
                    counts,
                    balance,
                    minimum,
                    maximum,
                    first,
                    degree,
                ],
            )
            wp.launch(
                validate_edges,
                m,
                inputs=[
                    slots,
                    counts,
                    balance,
                    minimum,
                    maximum,
                    self._twins,
                    self._topology_error,
                ],
            )
            wp.launch(
                validate_vertex_links, n, inputs=[first, degree, self._twins, self._topology_error]
            )
            wp.copy(self.error, self._topology_error)
            wp.launch(safe_points, n, inputs=[self.points, self._points, self.error])
            self._hierarchy = SurfaceHierarchy(self._points, self._indices)
            self.tree = self._hierarchy.tree
            if query_backend == "native":
                from . import _native_surface_queries

                self._native_queries = _native_surface_queries
                self._native_mesh = wp.Mesh(self._points, self._indices, bvh_constructor="lbvh")
                wp.load_module(module=_native_surface_queries, device=self.device)
                wp.load_module(module=_native_surface_queries, device=self.device, block_dim=32)
            self.surface_points = wp.empty(samples, dtype=wp.vec3)
            self.surface_weights = wp.empty(samples, dtype=wp.float64)
            self.feature_edges = wp.empty(m, dtype=wp.vec2i)
            self.feature_count = wp.zeros(1, dtype=int)
            self.crease_points = wp.empty(crease_samples, dtype=wp.vec3)
            self.crease_weights = wp.empty(crease_samples, dtype=wp.float64)
            self.corner_ids = wp.empty(n, dtype=int)
            self.corner_points = wp.empty(n, dtype=wp.vec3)
            self.corner_count = wp.zeros(1, dtype=int)
            self.corner_mask = wp.empty(n, dtype=int)
            self._flags = wp.empty(m, dtype=int)
            self._prefix = wp.empty(m, dtype=int)
            self._corner_prefix = wp.empty(n, dtype=int)
            self._feature_degree = wp.zeros(n, dtype=int)
            self._feature_direction = wp.zeros(n, dtype=wp.vec3d)
            self._scan_edges, self._scan_corners = _Scan(m, self.device), _Scan(n, self.device)
            wp.load_module(module=__name__, device=self.device)
            wp.load_module(module="warpower._triangle_contact", device=self.device)
            wp.load_module(module="warpower._graph", device=self.device)
            self.refit()
        wp.synchronize_device(self.device)

    def refit(self, stream=None):
        """Refit the boundary, validate geometry, and update quadrature on device."""
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream):
            wp.launch(
                advance_query_epoch, 1, inputs=[self._query_epoch], device=self.device, block_dim=32
            )
            wp.copy(self.error, self._topology_error)
            wp.launch(safe_points, len(self.points), inputs=[self.points, self._points, self.error])
            self._hierarchy.refit()
            if self._native_mesh is not None:
                self._native_mesh.refit()
            wp.launch(
                face_geometry,
                self.face_count,
                inputs=[self._points, self._indices, self._normals, self._areas, self.error],
            )
            if self.validate_geometry:
                wp.launch(
                    check_intersections,
                    self.face_count,
                    inputs=[self.tree, self.error],
                )
                wp.launch(
                    check_orientation,
                    self.face_count,
                    inputs=[self.tree, self.error],
                )
            self._feature_degree.zero_()
            self._feature_direction.zero_()
            wp.launch(
                feature_flags,
                len(self._flags),
                inputs=[self._twins, self._normals, self._feature_cos, self.error, self._flags],
            )
            self._scan_edges.run(self._flags, self._prefix)
            wp.launch(
                compact_features,
                len(self._flags),
                inputs=[
                    self._indices,
                    self._flags,
                    self._prefix,
                    self.feature_edges,
                    self.feature_count,
                    self._feature_degree,
                    self._feature_direction,
                    self._points,
                ],
            )
            wp.launch(
                corner_flags,
                len(self.points),
                inputs=[
                    self._feature_degree,
                    self._feature_direction,
                    self._corner_cos,
                    self.corner_mask,
                ],
            )
            self._scan_corners.run(self.corner_mask, self._corner_prefix)
            wp.launch(
                compact_corners,
                len(self.points),
                inputs=[
                    self._points,
                    self.corner_mask,
                    self._corner_prefix,
                    self.corner_ids,
                    self.corner_points,
                    self.corner_count,
                ],
            )
            wp.launch(
                surface_quadrature,
                len(self.surface_points),
                inputs=[
                    self._points,
                    self._indices,
                    self._areas,
                    self.surface_subdivisions,
                    self.error,
                    self.surface_points,
                    self.surface_weights,
                ],
            )
            wp.launch(
                crease_quadrature,
                len(self.crease_points),
                inputs=[
                    self._points,
                    self.feature_edges,
                    self.feature_count,
                    self.crease_subdivisions,
                    self.crease_points,
                    self.crease_weights,
                ],
            )
        return self


class DomainQueryWorkspace:
    """Fixed query buffers; ``run()`` is capturable and reads current input points."""

    def __init__(self, domain, points, *, max_distance=1.0e30):
        self.domain, self._owner = domain, points
        self.points = _contiguous_points(points)
        self.device = self.points.device
        if self.device != domain.device:
            raise ValueError("Query points must be on the domain device")
        if not math.isfinite(max_distance) or not 0 < max_distance <= 1.0e30:
            raise ValueError("max_distance must be positive, finite, and at most 1e30")
        self.max_distance = float(max_distance)
        with wp.ScopedDevice(self.device):
            self.closest = wp.empty_like(self.points)
            self.normal = wp.empty_like(self.points)
            self.signed_distance = wp.empty(len(self.points), dtype=float)
            self.inside = wp.empty(len(self.points), dtype=int)
            self.face = wp.empty(len(self.points), dtype=int)
            self.valid = wp.empty(len(self.points), dtype=int)
            self.error = wp.zeros(1, dtype=int)

    def run(self, stream=None):
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream):
            wp.copy(self.error, self.domain.error)
            wp.launch(
                domain_queries
                if self.domain._native_queries is None
                else self.domain._native_queries.domain_queries,
                len(self.points),
                inputs=[
                    self.domain.tree
                    if self.domain._native_mesh is None
                    else self.domain._native_mesh.id,
                    self.points,
                    self.domain.error,
                    self.max_distance,
                    self.closest,
                    self.normal,
                    self.signed_distance,
                    self.inside,
                    self.face,
                    self.valid,
                    self.error,
                ],
            )
        return self
