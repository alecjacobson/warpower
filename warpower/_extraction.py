"""Device-only circumcenter extraction of a boundary-approximating tet mesh."""

import math

import numpy as np
import warp as wp

from ._domain import _contiguous_points, _Scan
from ._mesh_topology import MeshTopology
from ._surface_bvh import SurfaceTree, closest_point, closest_triangle, material_side
from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def input_ready(
    domain_error: wp.array[int],
    source_error: wp.array[int],
    count: wp.array[int],
    capacity: int,
    error: wp.array[int],
):
    error[0] = int(domain_error[0] != 0)
    if source_error[0] != 0 or count[0] < 0 or count[0] > capacity:
        error[0] = error[0] | 2


@wp.kernel
def classify(
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
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a, b, c = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
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
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    hit = closest_point(tree, center, wp.float64(1.0e150))
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    if hit.distance2 > wp.float64(0.0):
        side = material_side(tree, center)
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


@wp.struct
class SelectionCenter:
    retained: int
    distance_sign: wp.float64
    error: int


@wp.func(inline=True)
def selection_center(
    tree: SurfaceTree, center: wp.vec3d, radius: wp.float64, tolerance: wp.float64
):
    """Private retention and distance sign; magnitude is not a distance output.

    Radius/geometry are validated by the caller. Unproved or non-inside centers
    keep the complete query, including exact-zero and strict threshold handling.
    """
    result = SelectionCenter()
    proved = False
    if (
        tree.n > 0
        and wp.abs(center[0]) < wp.float64(1.0e140)
        and wp.abs(center[1]) < wp.float64(1.0e140)
        and wp.abs(center[2]) < wp.float64(1.0e140)
    ):
        f = tree.ids[0]
        x = wp.vec3d(tree.points[tree.indices[3 * f]])
        y = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
        z = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
        q = closest_triangle(center, x, y, z)
        witness_distance = wp.length_sq(center - q)
        limit = wp.float64(1.0e150)
        # Current finite float32 boxes have squared distances < 4e280
        # from a center within these bounds. Before a finite hit, the full
        # traversal cannot prune this witnessed leaf against its ~1e300
        # initial limit. A different earlier hit also proves query success.
        proved = wp.isfinite(witness_distance) and witness_distance < limit * limit
    side = int(-1)
    if proved:
        side = material_side(tree, center)
    if proved and side == 1:
        # Private sign-only contract. Full distances remain unavailable here.
        result.distance_sign = wp.float64(-1.0)
        result.retained = 1
    else:
        hit = closest_point(tree, center, wp.float64(1.0e150))
        if hit.face < 0 or not wp.isfinite(hit.distance2):
            result.error = 16
            return result
        if hit.distance2 == wp.float64(0.0):
            side = 1
        elif not proved:
            side = material_side(tree, center)
        if side != 0 and side != 1:
            result.error = 16
            return result
        distance = wp.sqrt(hit.distance2)
        result.distance_sign = wp.where(side == 1, -distance, distance)
        result.retained = int(side == 1 or distance / radius < tolerance)
    return result


@wp.kernel
def classify_selection(
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
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    # Keep physical geometry arithmetic identical to classify. This private
    # kernel changes only the surface-query decision, not public geometry.
    p0 = wp.vec3d(points[ids[0]])
    a, b, c = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
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
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    status = selection_center(tree, center, radius, tolerance)
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
def compact(
    source: wp.array2d[int],
    keep: wp.array[int],
    prefix: wp.array[int],
    input_volume: wp.array[wp.float64],
    input_quality: wp.array[wp.float64],
    error: wp.array[int],
    output: wp.array2d[int],
    count: wp.array[int],
    source_indices: wp.array[int],
    quality: wp.array[wp.float64],
    volume: wp.array[wp.float64],
    minimum: wp.array[wp.float64],
):
    i = wp.tid()
    if error[0] != 0:
        return
    if i == keep.shape[0] - 1:
        count[0] = prefix[i]
    if keep[i] == 0:
        return
    at = prefix[i] - 1
    for k in range(4):
        output[at, k] = source[i, k]
    source_indices[at] = i
    quality[at] = input_quality[i]
    wp.atomic_add(volume, 0, input_volume[i])
    wp.atomic_min(minimum, 0, input_quality[i])


@wp.kernel
def surface_error(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    count: wp.array[int],
    integral: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    error: wp.array[int],
):
    i = wp.tid()
    if i >= count[0] or error[0] != 0:
        return
    a = wp.vec3d(points[faces[i, 0]])
    b = wp.vec3d(points[faces[i, 1]])
    c = wp.vec3d(points[faces[i, 2]])
    weight = wp.length(wp.cross(b - a, c - a)) / wp.float64(6.0)
    for k in range(3):
        dominant = a
        if k == 1:
            dominant = b
        elif k == 2:
            dominant = c
        sample = (a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0)
        hit = closest_point(tree, sample, wp.float64(1.0e150))
        if hit.face < 0 or not wp.isfinite(hit.distance2):
            wp.atomic_or(error, 0, 16)
        else:
            wp.atomic_add(integral, 0, weight * hit.distance2)
            wp.atomic_max(maximum, 0, wp.sqrt(hit.distance2))


@wp.kernel
def selection_surface_status(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
):
    """Preserve surface-query errors when internal selection needs no distances."""
    i = wp.tid()
    if i >= count[0] or error[0] != 0:
        return
    a = wp.vec3d(points[faces[i, 0]])
    b = wp.vec3d(points[faces[i, 1]])
    c = wp.vec3d(points[faces[i, 2]])
    for k in range(3):
        dominant = a
        if k == 1:
            dominant = b
        elif k == 2:
            dominant = c
        sample = (a + b + c) / wp.float64(6.0) + dominant / wp.float64(2.0)
        proved = False
        if tree.n > 0:
            f = tree.ids[0]
            x = wp.vec3d(tree.points[tree.indices[3 * f]])
            y = wp.vec3d(tree.points[tree.indices[3 * f + 1]])
            z = wp.vec3d(tree.points[tree.indices[3 * f + 2]])
            q = closest_triangle(sample, x, y, z)
            distance = wp.length_sq(sample - q)
            limit = wp.float64(1.0e150)
            # Current finite float32 boxes and these boundary samples have
            # squared box distances < 1.5e78. Before any finite hit, the full
            # traversal cannot prune this leaf against its ~1e300 limit.
            # This identical leaf computation therefore witnesses a finite
            # hit, even if another triangle is visited first. No nearest
            # distance or geometric accuracy is inferred from the witness.
            proved = wp.isfinite(distance) and distance < limit * limit
        if not proved:
            hit = closest_point(tree, sample, wp.float64(1.0e150))
            if hit.face < 0 or not wp.isfinite(hit.distance2):
                wp.atomic_or(error, 0, 16)


@wp.kernel
def domain_volume(tree: SurfaceTree, volume: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    if error[0] != 0:
        return
    origin = wp.vec3d(tree.points[0])
    a = wp.vec3d(tree.points[tree.indices[3 * i]]) - origin
    b = wp.vec3d(tree.points[tree.indices[3 * i + 1]]) - origin
    c = wp.vec3d(tree.points[tree.indices[3 * i + 2]]) - origin
    wp.atomic_add(volume, 0, wp.dot(a, wp.cross(b, c)) / wp.float64(6.0))


@wp.kernel
def finish(
    error: wp.array[int],
    diagnostics: wp.array[int],
    count: wp.array[int],
    boundary_count: wp.array[int],
    valid: wp.array[int],
    volume: wp.array[wp.float64],
    target_volume: wp.array[wp.float64],
    difference: wp.array[wp.float64],
    area: wp.array[wp.float64],
    integral: wp.array[wp.float64],
    rms: wp.array[wp.float64],
):
    valid[0] = int(error[0] == 0 and diagnostics[0] == 0)
    difference[0] = volume[0] - target_volume[0]
    rms[0] = wp.float64(0.0)
    if area[0] > wp.float64(0.0):
        rms[0] = wp.sqrt(integral[0] / area[0])
    if error[0] != 0:
        count[0] = 0
        boundary_count[0] = 0


def _int_array(value, device, shape=None):
    if isinstance(value, np.ndarray):
        value = wp.array(value, device=device)
    elif type(value).__module__.startswith("torch"):
        value = wp.from_torch(value)
    if (
        not isinstance(value, wp.array)
        or value.dtype != wp.int32
        or value.device != device
        or not value.is_contiguous
        or (shape is not None and value.shape != shape)
    ):
        raise ValueError(
            "Connectivity/count/error inputs must be contiguous int32 on the domain device"
        )
    return value


class ExtractionWorkspace:
    """Extract the Alliez thickened-domain subcomplex without changing input tets.

    Buffers and graph are allocated at setup. ``run`` uses current positions,
    connectivity and device counts. See docs/extraction.md for validation limits.
    """

    def __init__(
        self,
        domain,
        points,
        tetrahedra,
        tetrahedra_count=None,
        *,
        source_error=None,
        tolerance=0.4,
        containment_recovery=False,
        exterior_recovery=False,
        feature_roles=None,
        exterior_enabled=None,
        proven_exterior_recovery=False,
        proven_exterior_enabled=None,
        _cache_surface_queries=False,
        _rejected_link_witness=None,
    ):
        self.domain = domain
        self.points = _contiguous_points(points)
        self.device = self.points.device
        if self.device != domain.device or not 0 < len(self.points) < 2**29:
            raise ValueError("Nonempty points must be on the domain device and fit int32 indexing")
        self._owners = points, tetrahedra, tetrahedra_count, source_error
        self.input_tetrahedra = _int_array(tetrahedra, self.device)
        if self.input_tetrahedra.ndim != 2 or self.input_tetrahedra.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (capacity, 4)")
        self.capacity = len(self.input_tetrahedra)
        if not 0 < self.capacity < 2**25:
            raise ValueError("Tetrahedron capacity must be positive and fit int32 topology tables")
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("Extraction tolerance must be finite and in [0, 1)")
        self.tolerance = float(tolerance)
        self.containment_recovery = bool(containment_recovery)
        self.exterior_recovery = bool(exterior_recovery)
        self.proven_exterior_recovery = bool(proven_exterior_recovery)
        self.feature_roles = (
            None
            if feature_roles is None
            else _int_array(feature_roles, self.device, (len(self.points),))
        )
        self.exterior_enabled = (
            None if exterior_enabled is None else _int_array(exterior_enabled, self.device, (1,))
        )
        self.proven_exterior_enabled = (
            None
            if proven_exterior_enabled is None
            else _int_array(proven_exterior_enabled, self.device, (1,))
        )
        self._rejected_link_witness = (
            None
            if _rejected_link_witness is None
            else _int_array(_rejected_link_witness, self.device, (1,))
        )
        self._query_cache = None
        if _cache_surface_queries:
            from ._surface_query_cache import SurfaceQueryCache

            self._query_cache = SurfaceQueryCache(domain, self.capacity)
        with wp.ScopedDevice(self.device):
            self.input_count = (
                wp.full(1, self.capacity, dtype=int)
                if tetrahedra_count is None
                else _int_array(tetrahedra_count, self.device, (1,))
            )
            self.source_error = (
                wp.zeros(1, dtype=int)
                if source_error is None
                else _int_array(source_error, self.device, (1,))
            )
            self.error = wp.zeros(1, dtype=int)
            self.valid = wp.zeros(1, dtype=int)
            self.selected = wp.empty(self.capacity, dtype=int)
            self.circumcenters = wp.empty(self.capacity, dtype=wp.vec3d)
            self.circumradii = wp.empty(self.capacity, dtype=wp.float64)
            self.signed_distance = wp.empty(self.capacity, dtype=wp.float64)
            self.tetrahedra = wp.empty_like(self.input_tetrahedra)
            self.tetrahedra_count = wp.zeros(1, dtype=int)
            self.source_indices = wp.empty(self.capacity, dtype=int)
            self.radius_ratio = wp.empty(self.capacity, dtype=wp.float64)
            self.volume = wp.zeros(1, dtype=wp.float64)
            self.domain_volume = wp.zeros_like(self.volume)
            self.volume_difference = wp.zeros_like(self.volume)
            self.minimum_radius_ratio = wp.ones_like(self.volume)
            self.surface_rms_distance = wp.zeros_like(self.volume)
            self.surface_max_sample_distance = wp.zeros_like(self.volume)
            self._surface_integral = wp.zeros_like(self.volume)
            self._input_volume = wp.empty_like(self.radius_ratio)
            self._input_quality = wp.empty_like(self.radius_ratio)
            self._prefix = wp.empty_like(self.selected)
            self._scan = _Scan(self.capacity, self.device)
            self._topology = MeshTopology(self.points, self.tetrahedra, self.tetrahedra_count)
            for name in (
                "diagnostics",
                "boundary_faces",
                "boundary_count",
                "boundary_owners",
                "component_labels",
                "component_count",
                "shell_labels",
                "shell_count",
                "cavity_count",
                "shell_euler",
                "shell_volume",
                "boundary_area",
            ):
                setattr(self, name, getattr(self._topology, name))
            if self.containment_recovery:
                from . import _containment

                self.repaired_count = wp.zeros(1, dtype=int)
                self.repair_rounds = wp.zeros(1, dtype=int)
                self._repair_additions = wp.zeros(1, dtype=int)
                self._repair_ready = wp.zeros(1, dtype=int)
                wp.load_module(module=_containment, device=self.device)
            self._exterior = None
            if self.exterior_recovery:
                from ._exterior_band import ExteriorBandRecovery

                self._exterior = ExteriorBandRecovery(
                    self,
                    self.feature_roles,
                    self.exterior_enabled,
                    pinched_site=self._rejected_link_witness,
                )
                self.removed_count = self._exterior.removed
                self.exterior_rounds = self._exterior.rounds
                self.exterior_rejected = self._exterior.rejected
            self._proven_exterior = None
            if self.proven_exterior_recovery:
                from ._proven_exterior import ProvenExteriorRecovery

                self._proven_exterior = ProvenExteriorRecovery(self, self.proven_exterior_enabled)
                self.proven_removed_count = self._proven_exterior.removed
                self.proven_exterior_rejected = self._proven_exterior.rejected
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args, *, block_dim=256):
        wp.launch(kernel, dim, inputs=args, device=self.device, block_dim=block_dim)

    def _record(self):
        for a in (
            self.tetrahedra_count,
            self.volume,
            self.domain_volume,
            self._surface_integral,
            self.surface_max_sample_distance,
        ):
            a.zero_()
        self.minimum_radius_ratio.fill_(1.0)
        self._launch(
            input_ready,
            1,
            [self.domain.error, self.source_error, self.input_count, self.capacity, self.error],
        )
        self._classify()
        self._compact_topology()
        if self.containment_recovery:
            from . import _containment

            self._launch(
                _containment.begin,
                1,
                [
                    self.error,
                    self.diagnostics,
                    self.repair_rounds,
                    self.repaired_count,
                    self._repair_ready,
                ],
            )
            wp.capture_while(self._repair_ready, self._repair)
        if self._exterior is not None:
            self._exterior.run()
        if self._proven_exterior is not None:
            self._proven_exterior.run()
        self._surface_metrics()
        self._launch(
            domain_volume, self.domain.tree.n, [self.domain.tree, self.domain_volume, self.error]
        )
        self._launch(
            finish,
            1,
            [
                self.error,
                self.diagnostics,
                self.tetrahedra_count,
                self.boundary_count,
                self.valid,
                self.volume,
                self.domain_volume,
                self.volume_difference,
                self.boundary_area,
                self._surface_integral,
                self.surface_rms_distance,
            ],
        )

    def _classify(self):
        self._classify_with(classify)

    def _classify_with(self, kernel):
        args = [
            self.domain.tree,
            self.points,
            self.input_tetrahedra,
            self.input_count,
            self.tolerance,
            self.error,
            self.circumcenters,
            self.circumradii,
            self.signed_distance,
            self.selected,
            self._input_volume,
            self._input_quality,
        ]
        if self._query_cache is not None:
            self._query_cache.extraction(args, selection=kernel is classify_selection)
        else:
            # Register-heavy surface queries need enough blocks for small sets.
            self._launch(kernel, self.capacity, args, block_dim=32)

    def _surface_metrics(self):
        self._launch(
            surface_error,
            len(self.boundary_faces),
            [
                self.domain.tree,
                self.points,
                self.boundary_faces,
                self.boundary_count,
                self._surface_integral,
                self.surface_max_sample_distance,
                self.error,
            ],
            block_dim=32,
        )

    def _compact_topology(self):
        self._scan.run(self.selected, self._prefix)
        self._launch(
            compact,
            self.capacity,
            [
                self.input_tetrahedra,
                self.selected,
                self._prefix,
                self._input_volume,
                self._input_quality,
                self.error,
                self.tetrahedra,
                self.tetrahedra_count,
                self.source_indices,
                self.radius_ratio,
                self.volume,
                self.minimum_radius_ratio,
            ],
        )
        self._topology.run()

    def _repair(self):
        from . import _containment

        t = self._topology
        self._repair_additions.zero_()
        self._launch(
            _containment.add_contained,
            self.capacity,
            [
                self.domain.tree,
                self.points,
                self.input_tetrahedra,
                self.input_count,
                self.selected,
                t.boundary_indices,
                t._edges,
                t._edge_counts,
                t._edge_balance,
                self._repair_additions,
                self.repaired_count,
            ],
        )
        wp.capture_if(self._repair_additions, self._recompact)
        self._launch(
            _containment.next_round,
            1,
            [
                self.error,
                self.diagnostics,
                self._repair_additions,
                self.repair_rounds,
                self._repair_ready,
            ],
        )

    def _recompact(self):
        self.tetrahedra_count.zero_()
        self.volume.zero_()
        self.minimum_radius_ratio.fill_(1.0)
        self._compact_topology()

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self


class _SelectionExtractionWorkspace(ExtractionWorkspace):
    """Private selection/recovery and validity, without surface-distance outputs.

    Geometry, topology, recovery and error checks have the full extractor's
    semantics. signed_distance is only a nonpositive/positive sign marker;
    its magnitude is unavailable. Unused RMS/max/integral fields stay zero.
    Callers needing distances must use the public ExtractionWorkspace.
    """

    def _classify(self):
        self._classify_with(classify_selection)

    def _surface_metrics(self):
        self._launch(
            selection_surface_status,
            len(self.boundary_faces),
            [
                self.domain.tree,
                self.points,
                self.boundary_faces,
                self.boundary_count,
                self.error,
            ],
            block_dim=32,
        )
