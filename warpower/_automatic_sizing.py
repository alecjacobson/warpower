"""Pole-based boundary feature-size estimates for sampled graded sizing."""

import operator

import warp as wp

from . import _poles
from ._delaunay_recovery import DelaunayWorkspace
from ._domain import _Scan, surface_quadrature
from ._site_bvh import SiteHierarchy
from ._sizing import SizingField

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def begin(domain_error: wp.array[int], error: wp.array[int], ready: wp.array[int]):
    error[0] = int(domain_error[0] != 0)
    ready[0] = int(error[0] == 0)


@wp.kernel
def mesh_ready(mesh_error: wp.array[int], error: wp.array[int], ready: wp.array[int]):
    if error[0] == 0 and mesh_error[0] != 0:
        error[0] = 2
    ready[0] = int(error[0] == 0)


@wp.kernel
def poles_ready(count: wp.array[int], error: wp.array[int], ready: wp.array[int]):
    if error[0] == 0 and count[0] == 0:
        error[0] = 8
    ready[0] = int(error[0] == 0)


@wp.kernel
def feature_sizes(
    points: wp.array[wp.vec3],
    tree: _poles.PoleTree,
    distance: wp.array[wp.float64],
    values: wp.array[float],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        return
    d = _poles.nearest_pole(tree, wp.vec3d(points[i]))
    if not wp.isfinite(d) or d <= wp.float64(0.0) or d >= wp.float64(1e149):
        wp.atomic_or(error, 0, 8)
    else:
        distance[i] = d
        # Values above every supported upper clamp have the same field effect.
        # The lower floor only protects float32 underflow, below field.minimum.
        values[i] = float(wp.clamp(d, wp.float64(1.401298464324817e-45), wp.float64(1e30)))


@wp.kernel
def finish(
    distance: wp.array[wp.float64],
    values: wp.array[float],
    valid: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        valid[i] = 0
        if i < values.shape[0]:
            distance[i] = wp.float64(0.0)
            values[i] = 0.0
        if i == 0:
            count[0] = 0


class AutomaticSizingWorkspace:
    """Approximate boundary feature size from finite Voronoi poles, on device.

    ``field`` is a SizingField for downstream queries or VTM. Original boundary
    vertices and configurable triangle quadrature form the control samples.
    The approximation depends on their density; it is not an exact medial axis.
    """

    def __init__(
        self,
        domain,
        *,
        control_subdivisions=1,
        gradation=1.0,
        minimum=1e-6,
        maximum=1e30,
        tetrahedra_capacity=None,
        tetrahedra_cell_capacity=1024,
        chunk_size=None,
    ):
        self.domain, self.device = domain, domain.device
        self.control_subdivisions = operator.index(control_subdivisions)
        if self.control_subdivisions < 0:
            raise ValueError("control_subdivisions must be nonnegative")
        self.extra = 3 * domain.face_count * self.control_subdivisions**2
        self.n = len(domain.points) + self.extra
        if self.n >= 2**28:
            raise ValueError("Control/pole capacity exceeds hierarchy indexing")
        with wp.ScopedDevice(self.device):
            self.control_points = wp.zeros(self.n, dtype=wp.vec3)
            self._samples = self.control_points[len(domain.points) :] if self.extra else None
            self._sample_weights = wp.zeros(self.extra, dtype=wp.float64)
            self._controls()
            self.mesh = DelaunayWorkspace(
                self.control_points,
                tetrahedra_capacity=tetrahedra_capacity,
                tetrahedra_cell_capacity=tetrahedra_cell_capacity,
                chunk_size=chunk_size,
            )
            self.field = SizingField(
                self.control_points,
                wp.ones(self.n, dtype=float),
                gradation=gradation,
                minimum=minimum,
                maximum=maximum,
            )
            self.local_feature_size = wp.zeros(self.n, dtype=wp.float64)
            self.error = wp.zeros(1, dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            self._centers = wp.zeros(len(self.mesh.tetrahedra), dtype=wp.vec3d)
            self._hull_counts = wp.zeros(self.n + 1, dtype=int)
            self._hull_offsets = wp.empty(self.n + 1, dtype=int)
            self._hull_cursor = wp.empty(self.n, dtype=int)
            # A triangulated convex-hull boundary has at most 2*N-4 faces.
            self._hull_faces = wp.empty(6 * self.n, dtype=int)
            self._hull_scan = _Scan(self.n + 1, self.device)
            self._pole_keys = wp.empty((self.n, 4), dtype=int)
            self.hull_face = wp.full(self.n, 2**31 - 1, dtype=int)
            self._maximum = wp.zeros(self.n, dtype=wp.float64)
            self._first = wp.full(self.n, 2**31 - 1, dtype=int)
            self._second = wp.full(self.n, 2**31 - 1, dtype=int)
            self._normals = wp.zeros(self.n, dtype=wp.vec3d)
            self.poles = wp.zeros(2 * self.n, dtype=wp.vec3d)
            self.pole_valid = wp.zeros(2 * self.n, dtype=int)
            self.pole_source_tets = wp.full(2 * self.n, -1, dtype=int)
            self.pole_count = wp.zeros(1, dtype=int)
            self._sortable = wp.zeros(2 * self.n, dtype=wp.vec3)
            self._sort = SiteHierarchy(self._sortable)
            self._tree = _poles.PoleTree()
            self._tree.poles, self._tree.valid, self._tree.ids = (
                self.poles,
                self.pole_valid,
                self._sort.tree.ids,
            )
            self._tree.n = 2 * self.n
            self._tree.base = 1 << (((2 * self.n + 3) // 4) - 1).bit_length()
            self._tree.lower = wp.empty(2 * self._tree.base, dtype=wp.vec3d)
            self._tree.upper = wp.empty_like(self._tree.lower)
            wp.load_module(module=_poles, device=self.device)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def _launch(self, kernel, n, args):
        wp.launch(kernel, n, inputs=args, device=self.device)

    def _controls(self):
        wp.copy(self.control_points, self.domain.points, count=len(self.domain.points))
        if self.extra:
            self._launch(
                surface_quadrature,
                self.extra,
                [
                    self.domain._points,
                    self.domain._indices,
                    self.domain._areas,
                    self.control_subdivisions,
                    self.domain.error,
                    self._samples,
                    self._sample_weights,
                ],
            )

    def _record(self):
        self._launch(begin, 1, [self.domain.error, self.error, self._ready])
        self.field.values.zero_()
        self.local_feature_size.zero_()
        self.pole_count.zero_()
        self.pole_valid.zero_()
        self.pole_source_tets.fill_(-1)
        self._controls()
        wp.capture_if(self._ready, self.mesh.run)
        self._launch(mesh_ready, 1, [self.mesh.error, self.error, self._ready])
        wp.capture_if(self._ready, self._extract)
        self._launch(poles_ready, 1, [self.pole_count, self.error, self._ready])
        wp.capture_if(self._ready, self._distances)
        self._launch(
            finish,
            2 * self.n,
            [
                self.local_feature_size,
                self.field.values,
                self.pole_valid,
                self.pole_count,
                self.error,
            ],
        )
        self.field.refit()

    def _extract(self):
        tets, count = self.mesh.tetrahedra, self.mesh.tetrahedra_count
        cert = self.mesh._certificate
        self._hull_counts.zero_()
        self.hull_face.fill_(2**31 - 1)
        self._first.fill_(2**31 - 1)
        self._second.fill_(2**31 - 1)
        self._normals.zero_()
        self._launch(
            _poles.centers, len(tets), [self.control_points, tets, count, self._centers, self.error]
        )
        self._launch(
            _poles.hull_counts,
            len(cert._table),
            [tets, cert._table, cert._occurrences, self._hull_counts, self.error],
        )
        self._hull_scan.run(self._hull_counts, self._hull_offsets)
        wp.copy(self._hull_cursor, self._hull_offsets, count=self.n)
        self._launch(
            _poles.hull_scatter,
            len(cert._table),
            [tets, cert._table, cert._occurrences, self._hull_cursor, self._hull_faces, self.error],
        )
        self._launch(
            _poles.hull_directions,
            self.n,
            [
                self.control_points,
                tets,
                self._hull_offsets,
                self._hull_faces,
                self.hull_face,
                self._normals,
                self.error,
            ],
        )
        for opposite, selected in ((False, self._first), (True, self._second)):
            self._maximum.zero_()
            self._launch(
                _poles.farthest_distance,
                len(tets),
                [
                    self.control_points,
                    tets,
                    count,
                    self._centers,
                    self.hull_face,
                    self._normals,
                    self._maximum,
                    opposite,
                    self.error,
                ],
            )
            self._pole_keys.fill_(2**31 - 1)
            for prefix in range(5):
                self._launch(
                    _poles.farthest_select,
                    len(tets),
                    [
                        self.control_points,
                        tets,
                        count,
                        self._centers,
                        self.hull_face,
                        self._normals,
                        self._maximum,
                        self._pole_keys,
                        selected,
                        prefix,
                        opposite,
                        self.error,
                    ],
                )
            if not opposite:
                self._launch(
                    _poles.pole_directions,
                    self.n,
                    [
                        self.control_points,
                        tets,
                        self._centers,
                        self.hull_face,
                        self._first,
                        self._normals,
                        self.error,
                    ],
                )
        self._launch(
            _poles.publish_poles,
            2 * self.n,
            [
                self._centers,
                self._first,
                self._second,
                self.poles,
                self.pole_valid,
                self.pole_source_tets,
                self._sortable,
                self.pole_count,
                self.error,
            ],
        )

    def _distances(self):
        self._sort.rebuild()
        self._launch(_poles.pole_leaves, self._tree.base, [self._tree])
        level = self._tree.base // 2
        while level:
            self._launch(_poles.pole_merge, level, [self._tree, level])
            level //= 2
        self._launch(
            feature_sizes,
            self.n,
            [
                self.control_points,
                self._tree,
                self.local_feature_size,
                self.field.values,
                self.error,
            ],
        )

    def refit(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Automatic sizing stream must use the domain device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
