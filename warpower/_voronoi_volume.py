"""Signed circumcentric volumes of bounded cells in a certified Delaunay mesh."""

import warp as wp

from ._mesh_topology import face_vertices, sorted_face
from ._sizing import SizingData, sizing_value
from ._stable_volume import determinant as physical_determinant
from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def begin(
    active: wp.array[int],
    count: wp.array[int],
    core_error: wp.array[int],
    mesh_error: wp.array[int],
    field: SizingData,
    graded: bool,
    n: int,
    capacity: int,
    error: wp.array[int],
    ready: wp.array[int],
):
    error[0] = 0
    if active[0] < 4 or active[0] > n or count[0] <= 0 or count[0] > capacity:
        error[0] = 1
    if core_error[0] != 0 or mesh_error[0] != 0:
        error[0] = 1
    if graded and field.error[0] != 0:
        error[0] = 1
    ready[0] = int(error[0] == 0)


@wp.kernel
def validate(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    active: wp.array[int],
    incidence: wp.array[int],
    error: wp.array[int],
):
    t = wp.tid()
    if t >= count[0]:
        return
    ids = wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= active[0]:
            wp.atomic_or(error, 0, 2)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 2)
            return
    a = wp.vec3d(points[ids[0]])
    b = wp.vec3d(points[ids[1]]) - a
    c = wp.vec3d(points[ids[2]]) - a
    d = wp.vec3d(points[ids[3]]) - a
    determinant = wp.dot(b, wp.cross(c, d))
    determinant = physical_determinant(
        points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], b, c, d, determinant
    )
    if determinant <= wp.float64(0.0) or not wp.isfinite(determinant):
        wp.atomic_or(error, 0, 2)
        return
    for k in range(4):
        wp.atomic_add(incidence, ids[k], 1)


@wp.kernel
def ready(error: wp.array[int], gate: wp.array[int]):
    gate[0] = int(error[0] == 0)


@wp.kernel
def index_faces(
    tets: wp.array2d[int],
    count: wp.array[int],
    table: wp.array[int],
    occurrences: wp.array[int],
):
    f = wp.tid()
    if f >= 4 * count[0]:
        return
    key = sorted_face(face_vertices(tets, f))
    slot = int(
        (
            wp.uint32(key[0]) * wp.uint32(73856093)
            ^ wp.uint32(key[1]) * wp.uint32(19349663)
            ^ wp.uint32(key[2]) * wp.uint32(83492791)
        )
        & wp.uint32(table.shape[0] - 1)
    )
    while True:
        other = wp.atomic_cas(table, slot, -1, f)
        if other == -1 or sorted_face(face_vertices(tets, other)) == key:
            wp.atomic_add(occurrences, slot, 1)
            break
        slot = (slot + 1) & (table.shape[0] - 1)


@wp.kernel
def mark_hull(
    tets: wp.array2d[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    hull: wp.array[int],
    error: wp.array[int],
):
    slot = wp.tid()
    n = occurrences[slot]
    if n == 1:
        v = face_vertices(tets, table[slot])
        for k in range(3):
            wp.atomic_or(hull, v[k], 1)
    elif n > 2:
        wp.atomic_or(error, 0, 4)


@wp.kernel
def accumulate(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    hull: wp.array[int],
    volume: wp.array[wp.float64],
    magnitude: wp.array[wp.float64],
    error: wp.array[int],
):
    t = wp.tid()
    if t >= count[0]:
        return
    ids = wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3])
    # This cell contributes no bounded dual volume. Orientation was already
    # checked, so no circumcenter metric is needed for these four hull sites.
    if hull[ids[0]] != 0 and hull[ids[1]] != 0 and hull[ids[2]] != 0 and hull[ids[3]] != 0:
        return
    origin = wp.vec3d(points[ids[0]])
    a = wp.vec3d(points[ids[1]]) - origin
    b = wp.vec3d(points[ids[2]]) - origin
    c = wp.vec3d(points[ids[3]]) - origin
    determinant = wp.dot(a, wp.cross(b, c))
    offset = wp.vec3d()
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        offset = wp.vec3d(geometry.offset)
    else:
        offset = (
            wp.dot(a, a) * wp.cross(b, c)
            + wp.dot(b, b) * wp.cross(c, a)
            + wp.dot(c, c) * wp.cross(a, b)
        ) / (wp.float64(2.0) * determinant)
    for i in range(4):
        if hull[ids[i]] != 0:
            continue
        p = wp.vec3d(points[ids[i]])
        center = offset - (p - origin)
        total, absolute = wp.float64(0.0), wp.float64(0.0)
        for j in range(4):
            if j == i:
                continue
            u = wp.vec3d(points[ids[j]]) - p
            for k in range(4):
                if k == i or k == j:
                    continue
                last = 6 - i - j - k
                v = wp.vec3d(points[ids[k]]) - p
                normal = wp.cross(u, v)
                denominator = wp.float64(2.0) * wp.dot(normal, normal)
                face = (
                    wp.dot(u, u) * wp.cross(v, normal) + wp.dot(v, v) * wp.cross(normal, u)
                ) / denominator
                inversions = (
                    int(i > j)
                    + int(i > k)
                    + int(i > last)
                    + int(j > k)
                    + int(j > last)
                    + int(k > last)
                )
                sign = wp.float64(1 - 2 * (inversions % 2))
                value = sign * wp.dot(u, wp.cross(face, center)) / wp.float64(12.0)
                if not wp.isfinite(value) or denominator <= wp.float64(0.0):
                    wp.atomic_or(error, 0, 8)
                total += value
                absolute += wp.abs(value)
        wp.atomic_add(volume, ids[i], total)
        wp.atomic_add(magnitude, ids[i], absolute)


@wp.kernel
def normalize(
    points: wp.array[wp.vec3],
    active: wp.array[int],
    incidence: wp.array[int],
    hull: wp.array[int],
    volume: wp.array[wp.float64],
    magnitude: wp.array[wp.float64],
    field: SizingData,
    graded: bool,
    normalized: wp.array[wp.float64],
    valid: wp.array[int],
    invalid: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if i >= active[0] or incidence[i] == 0 or hull[i] != 0:
        return
    mu = wp.float64(1.0)
    if graded:
        mu = sizing_value(field, wp.vec3d(points[i]))
    value = volume[i] / (mu * mu * mu)
    # A conservative cancellation diagnostic, not an exact interval bound.
    if (
        not wp.isfinite(value)
        or not wp.isfinite(magnitude[i])
        or mu <= wp.float64(0.0)
        or value <= wp.float64(0.0)
        or volume[i] <= wp.float64(1.0e-12) * magnitude[i]
    ):
        wp.atomic_add(invalid, 0, 1)
        wp.atomic_or(error, 0, 8)
        return
    normalized[i] = value
    valid[i] = 1


@wp.kernel
def publish(
    error: wp.array[int],
    volume: wp.array[wp.float64],
    normalized: wp.array[wp.float64],
    valid: wp.array[int],
    count: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        volume[i] = wp.float64(0.0)
        normalized[i] = wp.float64(0.0)
        valid[i] = 0
    elif valid[i] != 0:
        wp.atomic_add(count, 0, 1)


class BoundedVoronoiVolumes:
    """Private fixed-buffer dual reduction; requires the core's current certified mesh."""

    def __init__(self, core):
        self.core, self.device = core, core.device
        self.n, self.m = core.n, len(core.full_tetrahedra)
        self.field = SizingData() if core.sizing is None else core.sizing.data
        self.graded = core.sizing is not None
        with wp.ScopedDevice(self.device):
            self.active = (
                core.active_count
                if core.active_count is not None
                else wp.full(1, self.n, dtype=int)
            )
            self.volume = wp.zeros(self.n, dtype=wp.float64)
            self.normalized = wp.zeros(self.n, dtype=wp.float64)
            self.valid = wp.zeros(self.n, dtype=int)
            self.hull = wp.zeros(self.n, dtype=int)
            self.incidence = wp.zeros(self.n, dtype=int)
            self.error = wp.zeros(1, dtype=int)
            self.invalid_cells = wp.zeros(1, dtype=int)
            self.count = wp.zeros(1, dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            self._magnitude = wp.zeros(self.n, dtype=wp.float64)
            size = 1 << (8 * self.m - 1).bit_length()
            self._table = wp.full(size, -1, dtype=int)
            self._occurrences = wp.zeros(size, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as cap:
                self._record()
            self.graph = cap.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _record(self):
        c = self.core
        for a in (
            self.volume,
            self.normalized,
            self.valid,
            self.hull,
            self.incidence,
            self.invalid_cells,
            self.count,
            self._magnitude,
        ):
            a.zero_()
        self._launch(
            begin,
            1,
            [
                self.active,
                c.full_tetrahedra_count,
                c.error,
                c.mesh.error,
                self.field,
                self.graded,
                self.n,
                self.m,
                self.error,
                self._ready,
            ],
        )
        wp.capture_if(self._ready, self._validate)
        self._launch(
            publish, self.n, [self.error, self.volume, self.normalized, self.valid, self.count]
        )

    def _validate(self):
        c = self.core
        self._launch(
            validate,
            self.m,
            [
                c.points,
                c.full_tetrahedra,
                c.full_tetrahedra_count,
                self.active,
                self.incidence,
                self.error,
            ],
        )
        self._launch(ready, 1, [self.error, self._ready])
        wp.capture_if(self._ready, self._compute)

    def _compute(self):
        c = self.core
        self._table.fill_(-1)
        self._occurrences.zero_()
        self._launch(
            index_faces,
            4 * self.m,
            [c.full_tetrahedra, c.full_tetrahedra_count, self._table, self._occurrences],
        )
        self._launch(
            mark_hull,
            len(self._table),
            [c.full_tetrahedra, self._table, self._occurrences, self.hull, self.error],
        )
        self._launch(
            accumulate,
            self.m,
            [
                c.points,
                c.full_tetrahedra,
                c.full_tetrahedra_count,
                self.hull,
                self.volume,
                self._magnitude,
                self.error,
            ],
        )
        self._launch(
            normalize,
            self.n,
            [
                c.points,
                self.active,
                self.incidence,
                self.hull,
                self.volume,
                self._magnitude,
                self.field,
                self.graded,
                self.normalized,
                self.valid,
                self.invalid_cells,
                self.error,
            ],
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Voronoi-volume stream must use the core device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
