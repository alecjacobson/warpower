"""Transactional tangential boundary jitter for the private VTM core."""

import math
import operator

import warp as wp

from ._quality_moves import MoveState, QualityMove
from ._sampling import random64
from ._surface_bvh import SurfaceTree, closest_point
from ._tetrahedra import sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def find_quality(
    state: MoveState,
    tets: wp.array2d[int],
    count: wp.array[int],
    quality: wp.array[wp.float64],
    roles: wp.array[int],
    eligible: wp.array[int],
    minimum: wp.array[wp.float64],
):
    i = wp.tid()
    eligible[i] = 0
    if i < count[0] and quality[i] < state.threshold:
        boundary = False
        for k in range(4):
            boundary = boundary or roles[tets[i, k]] == 1
        if boundary:
            eligible[i] = 1
            wp.atomic_min(minimum, 0, quality[i])


@wp.kernel
def high_key(
    tets: wp.array2d[int],
    quality: wp.array[wp.float64],
    eligible: wp.array[int],
    minimum: wp.array[wp.float64],
    high: wp.array[wp.uint64],
):
    i = wp.tid()
    if eligible[i] != 0 and quality[i] == minimum[0]:
        ids = sorted4(wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
        key = (wp.uint64(ids[0]) << wp.uint64(32)) | wp.uint64(ids[1])
        wp.atomic_min(high, 0, key)


@wp.kernel
def low_key(
    tets: wp.array2d[int],
    quality: wp.array[wp.float64],
    eligible: wp.array[int],
    minimum: wp.array[wp.float64],
    high: wp.array[wp.uint64],
    low: wp.array[wp.uint64],
):
    i = wp.tid()
    if eligible[i] != 0 and quality[i] == minimum[0]:
        ids = sorted4(wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
        hi = (wp.uint64(ids[0]) << wp.uint64(32)) | wp.uint64(ids[1])
        if hi == high[0]:
            lo = (wp.uint64(ids[2]) << wp.uint64(32)) | wp.uint64(ids[3])
            wp.atomic_min(low, 0, lo)


@wp.kernel
def select_tet(
    tets: wp.array2d[int],
    quality: wp.array[wp.float64],
    eligible: wp.array[int],
    minimum: wp.array[wp.float64],
    high: wp.array[wp.uint64],
    low: wp.array[wp.uint64],
    row: wp.array[int],
    selected: wp.array[int],
):
    i = wp.tid()
    if eligible[i] != 0 and quality[i] == minimum[0]:
        ids = sorted4(wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3]))
        hi = (wp.uint64(ids[0]) << wp.uint64(32)) | wp.uint64(ids[1])
        lo = (wp.uint64(ids[2]) << wp.uint64(32)) | wp.uint64(ids[3])
        if hi == high[0] and lo == low[0]:
            row[0] = i
            for k in range(4):
                selected[k] = ids[k]


@wp.kernel
def prepare_vertices(
    state: MoveState,
    tree: SurfaceTree,
    original: wp.array[wp.vec3],
    roles: wp.array[int],
    edge: wp.array[wp.float64],
    row: wp.array[int],
    selected: wp.array[int],
    fraction: wp.float64,
    sites: wp.array[int],
    normals: wp.array[wp.vec3d],
    u: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    limits: wp.array[wp.float64],
    count: wp.array[int],
):
    count[0] = 0
    state.running[0] = 0
    if row[0] < 0:
        return
    for k in range(4):
        site = selected[k]
        if roles[site] == 1:
            hit = closest_point(tree, wp.vec3d(original[site]), wp.float64(1.0e150))
            if (
                hit.face < 0
                or not wp.isfinite(edge[site])
                or edge[site] <= wp.float64(0.0)
                or edge[site] >= wp.float64(1.0e299)
            ):
                state.error[0] |= 4
                continue
            face = hit.face
            a = wp.vec3d(tree.points[tree.indices[3 * face]])
            b = wp.vec3d(tree.points[tree.indices[3 * face + 1]]) - a
            c = wp.vec3d(tree.points[tree.indices[3 * face + 2]]) - a
            normal = wp.cross(b, c)
            length = wp.length(normal)
            if not wp.isfinite(length) or length <= wp.float64(0.0):
                state.error[0] |= 4
                continue
            normal /= length
            axis = wp.vec3d(1.0, 0.0, 0.0)
            if wp.abs(normal[0]) > wp.float64(0.8):
                axis = wp.vec3d(0.0, 1.0, 0.0)
            tangent = wp.normalize(wp.cross(normal, axis))
            at = count[0]
            sites[at] = site
            normals[at] = normal
            u[at] = tangent
            v[at] = wp.cross(normal, tangent)
            limits[at] = fraction * edge[site]
            count[0] = at + 1
    state.running[0] = int(count[0] > 0 and state.error[0] == 0)


@wp.kernel
def propose(
    state: MoveState,
    seed: wp.array[int],
    original: wp.array[wp.vec3],
    sites: wp.array[int],
    count: wp.array[int],
    normals: wp.array[wp.vec3d],
    u: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    limits: wp.array[wp.float64],
    proposed_normal: wp.array[wp.vec3d],
    distance: wp.array[wp.float64],
    limit: wp.array[wp.float64],
):
    # Each finite attempt owns a disjoint 2^32-state stream. The operation salt
    # separates jitter variates from the initial sampler's site streams.
    bits = (wp.uint64(wp.uint32(seed[0])) ^ wp.uint64(0xD1B54A32D192ED03)) + (
        wp.uint64(state.attempts[0] - 1) << wp.uint64(32)
    ) * wp.uint64(0x9E3779B97F4A7C15)
    bits, choice = random64(bits)
    bits, angle = random64(bits)
    bits, radius = random64(bits)
    at = wp.min(count[0] - 1, int(choice * wp.float64(count[0])))
    site = sites[at]
    angle *= wp.float64(6.2831853071795864769)
    radius = wp.sqrt(radius) * limits[at]
    offset = radius * (wp.cos(angle) * u[at] + wp.sin(angle) * v[at])
    p = wp.vec3(wp.vec3d(original[site]) + offset)
    moved = wp.length(wp.vec3d(p) - wp.vec3d(original[site]))
    state.site[0] = site
    state.position[0] = p
    proposed_normal[0] = normals[at]
    distance[0] = moved
    limit[0] = limits[at]
    state.proposal_valid[0] = int(
        wp.isfinite(p[0])
        and wp.isfinite(p[1])
        and wp.isfinite(p[2])
        and moved > wp.float64(0.0)
        and moved <= limits[at]
    )


class BoundaryJitterWorkspace(QualityMove):
    """One bounded search for an improving tangential move of a surface site."""

    def __init__(self, core, *, step_fraction=0.1, seed=2026, **quality_options):
        if not math.isfinite(step_fraction) or not 0 < step_fraction <= 1:
            raise ValueError("step_fraction must be in (0,1]")
        super().__init__(core, **quality_options)
        self.step_fraction = step_fraction
        with wp.ScopedDevice(self.device):
            if isinstance(seed, wp.array):
                if (
                    seed.dtype != wp.int32
                    or seed.shape != (1,)
                    or seed.device != self.device
                    or not seed.is_contiguous
                ):
                    raise ValueError("Seed must be one contiguous device int32")
                self.seed = seed
            else:
                seed = operator.index(seed)
                if not -(2**31) <= seed < 2**31:
                    raise ValueError("Seed must fit int32")
                self.seed = wp.full(1, seed, dtype=int)
            self.source_tetrahedron = wp.full(1, -1, dtype=int)
            self.canonical_tetrahedron = wp.full(4, -1, dtype=int)
            self._eligible = wp.zeros(self._m, dtype=int)
            self._minimum = wp.ones(1, dtype=wp.float64)
            self._high = wp.zeros(1, dtype=wp.uint64)
            self._low = wp.zeros(1, dtype=wp.uint64)
            self._sites = wp.full(4, -1, dtype=int)
            self._normals = wp.zeros(4, dtype=wp.vec3d)
            self._u = wp.zeros(4, dtype=wp.vec3d)
            self._v = wp.zeros(4, dtype=wp.vec3d)
            self._limits = wp.zeros(4, dtype=wp.float64)
            self.eligible_vertices = wp.zeros(1, dtype=int)
            self.proposed_normal = wp.zeros(1, dtype=wp.vec3d)
            self.proposed_distance = wp.zeros(1, dtype=wp.float64)
            self.proposed_limit = wp.zeros(1, dtype=wp.float64)
            wp.load_module(module=__name__, device=self.device)
        self._finish_setup()

    def _prepare_proposal(self):
        e = self.extraction
        self.source_tetrahedron.fill_(-1)
        self.canonical_tetrahedron.fill_(-1)
        self._minimum.fill_(1)
        self._high.fill_(0xFFFFFFFFFFFFFFFF)
        self._low.fill_(0xFFFFFFFFFFFFFFFF)
        self._sites.fill_(-1)
        self.proposed_distance.zero_()
        self.proposed_limit.zero_()
        self.proposed_normal.zero_()
        self._launch(
            find_quality,
            self._m,
            [
                self._state,
                e.tetrahedra,
                e.tetrahedra_count,
                e.radius_ratio,
                self.core.boundary.roles,
                self._eligible,
                self._minimum,
            ],
        )
        self._launch(
            high_key,
            self._m,
            [e.tetrahedra, e.radius_ratio, self._eligible, self._minimum, self._high],
        )
        self._launch(
            low_key,
            self._m,
            [e.tetrahedra, e.radius_ratio, self._eligible, self._minimum, self._high, self._low],
        )
        self._launch(
            select_tet,
            self._m,
            [
                e.tetrahedra,
                e.radius_ratio,
                self._eligible,
                self._minimum,
                self._high,
                self._low,
                self.source_tetrahedron,
                self.canonical_tetrahedron,
            ],
        )
        self._launch(
            prepare_vertices,
            1,
            [
                self._state,
                self.core.domain.tree,
                self._old_points,
                self.core.boundary.roles,
                self.core._edge,
                self.source_tetrahedron,
                self.canonical_tetrahedron,
                self.step_fraction,
                self._sites,
                self._normals,
                self._u,
                self._v,
                self._limits,
                self.eligible_vertices,
            ],
        )

    def _propose(self):
        self._launch(
            propose,
            1,
            [
                self._state,
                self.seed,
                self._old_points,
                self._sites,
                self.eligible_vertices,
                self._normals,
                self._u,
                self._v,
                self._limits,
                self.proposed_normal,
                self.proposed_distance,
                self.proposed_limit,
            ],
        )
