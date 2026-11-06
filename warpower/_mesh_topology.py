"""Fixed-buffer topology and quality diagnostics for a tetrahedral subcomplex."""

import warp as wp

from ._domain import next_edge, previous_edge

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def face_vertices(tets: wp.array2d[int], face: int):
    t, opposite = face // 4, face % 4
    v = wp.vec3i(0)
    at = int(0)
    for k in range(4):
        if k != opposite:
            v[at] = tets[t, k]
            at += 1
    # Alternating simplicial boundary of a positively oriented tetrahedron.
    if opposite % 2 == 1:
        a = int(v[0])
        v[0] = v[1]
        v[1] = a
    return v


@wp.func
def sorted_face(v: wp.vec3i):
    a, b, c = v[0], v[1], v[2]
    return wp.vec3i(
        wp.min(a, wp.min(b, c)),
        a + b + c - wp.min(a, wp.min(b, c)) - wp.max(a, wp.max(b, c)),
        wp.max(a, wp.max(b, c)),
    )


@wp.func
def root(parent: wp.array[int], index: int):
    i = index
    next_i = wp.atomic_add(parent, i, 0)
    while next_i != i:
        i = next_i
        next_i = wp.atomic_add(parent, i, 0)
    return i


@wp.func
def join(parent: wp.array[int], left: int, right: int):
    a, b = root(parent, left), root(parent, right)
    while a != b:
        lo, hi = wp.min(a, b), wp.max(a, b)
        old = wp.atomic_min(parent, hi, lo)
        if old == hi:
            break
        a, b = root(parent, lo), root(parent, old)


@wp.kernel
def identity(parent: wp.array[int]):
    i = wp.tid()
    parent[i] = i


@wp.kernel
def flatten(parent: wp.array[int], count: wp.array[int], components: wp.array[int]):
    i = wp.tid()
    if i < count[0]:
        r = root(parent, i)
        wp.atomic_min(parent, i, r)
        if r == i:
            wp.atomic_add(components, 0, 1)


@wp.kernel
def index_faces(
    tets: wp.array2d[int],
    count: wp.array[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    parents: wp.array[int],
):
    f = wp.tid()
    if f >= 4 * count[0]:
        return
    oriented = face_vertices(tets, f)
    key = sorted_face(oriented)
    parity = (
        int(oriented[0] > oriented[1])
        + int(oriented[0] > oriented[2])
        + int(oriented[1] > oriented[2])
    )
    sign = 1 - 2 * (parity % 2)
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
            wp.atomic_add(balance, slot, sign)
            if other >= 0:
                join(parents, f // 4, other // 4)
            break
        slot = (slot + 1) & (table.shape[0] - 1)


@wp.kernel
def boundary_faces(
    tets: wp.array2d[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    output: wp.array[int],
    count: wp.array[int],
    owners: wp.array[int],
    diagnostics: wp.array[int],
):
    slot = wp.tid()
    n = occurrences[slot]
    if n == 1:
        f = table[slot]
        v = face_vertices(tets, f)
        at = wp.atomic_add(count, 0, 1)
        for k in range(3):
            output[3 * at + k] = v[k]
        owners[at] = f // 4
    elif n > 0 and (n != 2 or balance[slot] != 0):
        wp.atomic_or(diagnostics, 0, 1)


@wp.kernel
def index_edges(
    indices: wp.array[int],
    count: wp.array[int],
    table: wp.array[int],
    slots: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    minimum: wp.array[int],
    maximum: wp.array[int],
    first: wp.array[int],
    degree: wp.array[int],
    parents: wp.array[int],
):
    e = wp.tid()
    if e >= 3 * count[0]:
        return
    a, b = indices[e], indices[next_edge(e)]
    lo, hi = wp.min(a, b), wp.max(a, b)
    slot = int(
        (wp.uint32(lo) * wp.uint32(73856093) ^ wp.uint32(hi) * wp.uint32(19349663))
        & wp.uint32(table.shape[0] - 1)
    )
    while True:
        other = wp.atomic_cas(table, slot, -1, e)
        if other == -1 or (
            wp.min(indices[other], indices[next_edge(other)]) == lo
            and wp.max(indices[other], indices[next_edge(other)]) == hi
        ):
            if other >= 0:
                join(parents, e // 3, other // 3)
            break
        slot = (slot + 1) & (table.shape[0] - 1)
    slots[e] = slot
    wp.atomic_add(occurrences, slot, 1)
    wp.atomic_add(balance, slot, wp.where(a < b, 1, -1))
    wp.atomic_min(minimum, slot, e)
    wp.atomic_max(maximum, slot, e)
    wp.atomic_min(first, a, e)
    wp.atomic_add(degree, a, 1)


@wp.kernel
def validate_edges(
    count: wp.array[int],
    slots: wp.array[int],
    occurrences: wp.array[int],
    balance: wp.array[int],
    minimum: wp.array[int],
    maximum: wp.array[int],
    twins: wp.array[int],
    diagnostics: wp.array[int],
):
    e = wp.tid()
    if e >= 3 * count[0]:
        return
    slot = slots[e]
    twins[e] = -1
    if occurrences[slot] != 2 or balance[slot] != 0:
        wp.atomic_or(diagnostics, 0, 2)
    else:
        twins[e] = wp.where(e == minimum[slot], maximum[slot], minimum[slot])


@wp.kernel
def vertex_links(
    first: wp.array[int],
    degree: wp.array[int],
    twins: wp.array[int],
    parents: wp.array[int],
    chi: wp.array[int],
    diagnostics: wp.array[int],
):
    i = wp.tid()
    if degree[i] == 0:
        return
    wp.atomic_add(chi, parents[first[i] // 3], 1)
    if (diagnostics[0] & 2) != 0:
        return
    e = int(first[i])
    visited = int(0)
    for _ in range(degree[i]):
        visited += 1
        e = twins[previous_edge(e)]
        if e == first[i]:
            break
    if visited != degree[i] or e != first[i]:
        wp.atomic_or(diagnostics, 0, 4)


@wp.kernel
def euler_edges(table: wp.array[int], parents: wp.array[int], chi: wp.array[int]):
    i = wp.tid()
    if table[i] >= 0:
        wp.atomic_sub(chi, parents[table[i] // 3], 1)


@wp.kernel
def shell_geometry(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    count: wp.array[int],
    parents: wp.array[int],
    chi: wp.array[int],
    volumes: wp.array[wp.float64],
    area: wp.array[wp.float64],
):
    f = wp.tid()
    if f >= count[0]:
        return
    # A shared origin reduces translation cancellation in closed-shell sums.
    origin = wp.vec3d(points[0])
    a = wp.vec3d(points[indices[3 * f]]) - origin
    b = wp.vec3d(points[indices[3 * f + 1]]) - origin
    c = wp.vec3d(points[indices[3 * f + 2]]) - origin
    r = parents[f]
    wp.atomic_add(chi, r, 1)
    wp.atomic_add(volumes, r, wp.dot(a, wp.cross(b, c)) / wp.float64(6.0))
    wp.atomic_add(area, 0, wp.length(wp.cross(b - a, c - a)) * wp.float64(0.5))


@wp.kernel
def count_cavities(
    count: wp.array[int],
    parents: wp.array[int],
    volumes: wp.array[wp.float64],
    cavities: wp.array[int],
    diagnostics: wp.array[int],
):
    i = wp.tid()
    if i < count[0] and parents[i] == i and diagnostics[0] == 0:
        if volumes[i] < wp.float64(0.0):
            wp.atomic_add(cavities, 0, 1)


class MeshTopology:
    """Internal diagnostics; inputs must already be index/orientation validated.

    Face incidence and closed boundary edge/vertex links are checked. This is
    not an intersection test or a certificate for arbitrary input complexes.
    """

    def __init__(self, points, tets, count):
        self.points, self.tets, self.count = points, tets, count
        self.device = points.device
        m, n = len(tets), len(points)
        f = 4 * m
        face_slots = 1 << (2 * f - 1).bit_length()
        edge_slots = 1 << (6 * f - 1).bit_length()
        with wp.ScopedDevice(self.device):
            self.diagnostics = wp.zeros(1, dtype=int)
            self.boundary_indices = wp.empty(3 * f, dtype=int)
            self.boundary_faces = self.boundary_indices.reshape((f, 3))
            self.boundary_count = wp.zeros(1, dtype=int)
            self.boundary_owners = wp.empty(f, dtype=int)
            self.component_labels = wp.empty(m, dtype=int)
            self.component_count = wp.zeros(1, dtype=int)
            self.shell_labels = wp.empty(f, dtype=int)
            self.shell_count = wp.zeros(1, dtype=int)
            self.cavity_count = wp.zeros(1, dtype=int)
            self.shell_euler = wp.zeros(f, dtype=int)
            self.shell_volume = wp.zeros(f, dtype=wp.float64)
            self.boundary_area = wp.zeros(1, dtype=wp.float64)
            self._faces = wp.empty(face_slots, dtype=int)
            self._face_counts = wp.empty_like(self._faces)
            self._face_balance = wp.empty_like(self._faces)
            self._edges = wp.empty(edge_slots, dtype=int)
            self._edge_counts = wp.empty_like(self._edges)
            self._edge_balance = wp.empty_like(self._edges)
            self._edge_minimum = wp.empty_like(self._edges)
            self._edge_maximum = wp.empty_like(self._edges)
            self._edge_slots = wp.empty(3 * f, dtype=int)
            self._twins = wp.empty_like(self._edge_slots)
            self._first = wp.empty(n, dtype=int)
            self._degree = wp.empty(n, dtype=int)
            wp.load_module(module=__name__, device=self.device)

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim, inputs=args, device=self.device)

    def run(self):
        for array in (
            self.diagnostics,
            self.boundary_count,
            self.component_count,
            self.shell_count,
            self.cavity_count,
            self.shell_euler,
            self.shell_volume,
            self.boundary_area,
            self._face_counts,
            self._face_balance,
            self._edge_counts,
            self._edge_balance,
            self._degree,
        ):
            array.zero_()
        for array in (self._faces, self._edges, self._edge_maximum):
            array.fill_(-1)
        self._edge_minimum.fill_(2147483647)
        self._first.fill_(2147483647)
        self._launch(identity, len(self.component_labels), [self.component_labels])
        self._launch(identity, len(self.shell_labels), [self.shell_labels])
        self._launch(
            index_faces,
            4 * len(self.tets),
            [
                self.tets,
                self.count,
                self._faces,
                self._face_counts,
                self._face_balance,
                self.component_labels,
            ],
        )
        self._launch(
            flatten, len(self.tets), [self.component_labels, self.count, self.component_count]
        )
        self._launch(
            boundary_faces,
            len(self._faces),
            [
                self.tets,
                self._faces,
                self._face_counts,
                self._face_balance,
                self.boundary_indices,
                self.boundary_count,
                self.boundary_owners,
                self.diagnostics,
            ],
        )
        self._launch(
            index_edges,
            len(self.boundary_indices),
            [
                self.boundary_indices,
                self.boundary_count,
                self._edges,
                self._edge_slots,
                self._edge_counts,
                self._edge_balance,
                self._edge_minimum,
                self._edge_maximum,
                self._first,
                self._degree,
                self.shell_labels,
            ],
        )
        self._launch(
            flatten,
            len(self.shell_labels),
            [self.shell_labels, self.boundary_count, self.shell_count],
        )
        self._launch(
            validate_edges,
            len(self.boundary_indices),
            [
                self.boundary_count,
                self._edge_slots,
                self._edge_counts,
                self._edge_balance,
                self._edge_minimum,
                self._edge_maximum,
                self._twins,
                self.diagnostics,
            ],
        )
        self._launch(
            vertex_links,
            len(self.points),
            [
                self._first,
                self._degree,
                self._twins,
                self.shell_labels,
                self.shell_euler,
                self.diagnostics,
            ],
        )
        self._launch(
            euler_edges, len(self._edges), [self._edges, self.shell_labels, self.shell_euler]
        )
        self._launch(
            shell_geometry,
            len(self.shell_labels),
            [
                self.points,
                self.boundary_indices,
                self.boundary_count,
                self.shell_labels,
                self.shell_euler,
                self.shell_volume,
                self.boundary_area,
            ],
        )
        self._launch(
            count_cavities,
            len(self.shell_labels),
            [
                self.boundary_count,
                self.shell_labels,
                self.shell_volume,
                self.cavity_count,
                self.diagnostics,
            ],
        )
