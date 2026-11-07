"""Device-only circumcenter extraction of a boundary-approximating tet mesh."""

import math

import numpy as np
import warp as wp

from ._domain import _contiguous_points, _Scan
from ._mesh_topology import MeshTopology
from ._surface_bvh import SurfaceTree, closest_point, material_side
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
    center, radius = wp.vec3d(), wp.float64(0.0)
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
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
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
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
        self, domain, points, tetrahedra, tetrahedra_count=None, *, source_error=None, tolerance=0.4
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
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

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
        self._launch(
            classify,
            self.capacity,
            [
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
            ],
        )
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
        )
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
