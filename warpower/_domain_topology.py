"""Device topology invariants and signed shell volumes of an input domain."""

import warp as wp

from ._mesh_topology import join, root

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def setup_ready(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 0)


@wp.kernel
def initialize(parents: wp.array[int]):
    i = wp.tid()
    parents[i] = i


@wp.kernel
def connect(twins: wp.array[int], parents: wp.array[int]):
    edge = wp.tid()
    join(parents, edge // 3, twins[edge] // 3)


@wp.kernel
def label(
    parents: wp.array[int],
    indices: wp.array[int],
    labels: wp.array[int],
    used: wp.array[int],
):
    i = wp.tid()
    labels[i] = root(parents, i)
    for k in range(3):
        wp.atomic_or(used, indices[3 * i + k], 1)


@wp.kernel
def count_vertices(used: wp.array[int], chi: wp.array[int]):
    if used[wp.tid()] != 0:
        wp.atomic_add(chi, 0, 1)


@wp.kernel
def begin(
    domain_error: wp.array[int],
    chi: wp.array[int],
    counts: wp.array[int],
    volume: wp.array[wp.float64],
    error: wp.array[int],
    ready: wp.array[int],
):
    error[0] = int(domain_error[0] != 0)
    ready[0] = int(error[0] == 0)
    volume[0] = wp.float64(0.0)
    for k in range(4):
        counts[k] = 0
    counts[3] = chi[0]


@wp.kernel
def accumulate(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    labels: wp.array[int],
    volumes: wp.array[wp.float64],
):
    i = wp.tid()
    owner = labels[i]
    origin = wp.vec3d(points[indices[3 * owner]])
    a = wp.vec3d(points[indices[3 * i]]) - origin
    b = wp.vec3d(points[indices[3 * i + 1]]) - origin
    c = wp.vec3d(points[indices[3 * i + 2]]) - origin
    wp.atomic_add(volumes, owner, wp.dot(a, wp.cross(b, c)) / wp.float64(6.0))


@wp.kernel
def summarize(
    labels: wp.array[int],
    volumes: wp.array[wp.float64],
    counts: wp.array[int],
    volume: wp.array[wp.float64],
    error: wp.array[int],
):
    i = wp.tid()
    if labels[i] == i:
        v = volumes[i]
        if not wp.isfinite(v) or v == wp.float64(0.0):
            wp.atomic_or(error, 0, 2)
        else:
            wp.atomic_add(counts, 1, 1)
            if v > wp.float64(0.0):
                wp.atomic_add(counts, 0, 1)
            else:
                wp.atomic_add(counts, 2, 1)
            wp.atomic_add(volume, 0, v)


@wp.kernel
def publish(error: wp.array[int], counts: wp.array[int], volume: wp.array[wp.float64]):
    if error[0] == 0 and (not wp.isfinite(volume[0]) or volume[0] <= wp.float64(0.0)):
        error[0] = 2
    if error[0] != 0:
        for k in range(4):
            counts[k] = 0
        volume[0] = wp.float64(0.0)


class DomainTopology:
    """Private input-boundary invariants; topology is immutable, geometry can refit."""

    def __init__(self, domain):
        self.domain, self.device = domain, domain.device
        with wp.ScopedDevice(self.device):
            self.counts = wp.zeros(4, dtype=int)
            self.volume = wp.zeros(1, dtype=wp.float64)
            self.error = wp.zeros(1, dtype=int)
            self.labels = wp.full(domain.face_count, -1, dtype=int)
            self.shell_volumes = wp.zeros(domain.face_count, dtype=wp.float64)
            self._parents = wp.empty(domain.face_count, dtype=int)
            self._used = wp.zeros(len(domain.points), dtype=int)
            self._chi = wp.full(1, -(domain.face_count // 2), dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as static:
                self._launch(setup_ready, 1, [domain._topology_error, self._ready])
                wp.capture_if(self._ready, self._topology)
            self._static_graph = static.graph
            wp.capture_launch(self._static_graph)
            with wp.ScopedCapture() as cap:
                self._record()
            self.graph = cap.graph
        wp.synchronize_device(self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def _topology(self):
        d = self.domain
        self._launch(initialize, d.face_count, [self._parents])
        self._launch(connect, 3 * d.face_count, [d._twins, self._parents])
        self._launch(label, d.face_count, [self._parents, d._indices, self.labels, self._used])
        self._launch(count_vertices, len(d.points), [self._used, self._chi])

    def _record(self):
        self._launch(
            begin,
            1,
            [self.domain.error, self._chi, self.counts, self.volume, self.error, self._ready],
        )
        wp.capture_if(self._ready, self._geometry)
        self._launch(publish, 1, [self.error, self.counts, self.volume])

    def _geometry(self):
        d = self.domain
        self.shell_volumes.zero_()
        self._launch(
            accumulate, d.face_count, [d.tree.points, d._indices, self.labels, self.shell_volumes]
        )
        self._launch(
            summarize,
            d.face_count,
            [self.labels, self.shell_volumes, self.counts, self.volume, self.error],
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Domain topology stream must use the domain device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self
