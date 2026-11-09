"""Sampled boundary fitting with evolving roles and distinct corner anchors."""

import math

import numpy as np
import warp as wp

from . import _bvh
from ._domain import _contiguous_points
from ._site_bvh import SiteHierarchy, nearest_site

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def prepare_sites(
    points: wp.array[wp.vec3],
    safe: wp.array[wp.vec3],
    active: wp.array[int],
    domain_error: wp.array[int],
    error: wp.array[int],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
):
    i = wp.tid()
    p = wp.vec3(0.0)
    targets[i] = points[i]
    roles[i] = 0
    if domain_error[0] != 0:
        wp.atomic_or(error, 0, 1)
    if active[0] < 1 or active[0] > points.shape[0]:
        wp.atomic_or(error, 0, 2)
    elif i < active[0]:
        p = points[i]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 2)
            p = wp.vec3(0.0)
    safe[i] = p


@wp.kernel
def sites_ready(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 0)


@wp.kernel
def begin_corners(
    count: wp.array[int],
    active: wp.array[int],
    error: wp.array[int],
    pending: wp.array[int],
):
    if count[0] > active[0]:
        wp.atomic_or(error, 0, 4)
    pending[0] = wp.where(error[0] == 0, count[0], 0)


@wp.kernel
def propose_corners(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    active: wp.array[int],
    corners: wp.array[wp.vec3],
    count: wp.array[int],
    assigned: wp.array[int],
    occupied: wp.array[int],
    proposals: wp.array[int],
    winners: wp.array[int],
    error: wp.array[int],
):
    c = wp.tid()
    if c >= count[0] or assigned[c] >= 0 or error[0] != 0:
        return
    hit = nearest_site(tree, points, wp.vec3d(corners[c]), active, occupied, True)
    proposals[c] = hit.index
    if hit.index < 0:
        wp.atomic_or(error, 0, 4)
    else:
        wp.atomic_min(winners, hit.index, c)


@wp.kernel
def claim_corners(
    count: wp.array[int],
    assigned: wp.array[int],
    occupied: wp.array[int],
    proposals: wp.array[int],
    winners: wp.array[int],
    remaining: wp.array[int],
    error: wp.array[int],
):
    c = wp.tid()
    if c >= count[0] or assigned[c] >= 0 or error[0] != 0:
        return
    site = proposals[c]
    if site >= 0 and winners[site] == c:
        assigned[c] = site
        occupied[site] = c
    else:
        wp.atomic_add(remaining, 0, 1)


@wp.kernel
def continue_corners(
    remaining: wp.array[int],
    count: wp.array[int],
    rounds: wp.array[int],
    error: wp.array[int],
    pending: wp.array[int],
):
    rounds[0] += 1
    if rounds[0] >= count[0] and remaining[0] > 0:
        wp.atomic_or(error, 0, 4)
    pending[0] = wp.where(error[0] == 0, remaining[0], 0)


@wp.kernel
def accumulate_samples(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    samples: wp.array[wp.vec3],
    base_weights: wp.array[wp.float64],
    sizing: wp.array[float],
    dimension: int,
    owners: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    energies: wp.array[wp.float64],
    total: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    component: int,
    error: wp.array[int],
):
    s = wp.tid()
    owners[s] = -1
    if error[0] != 0 or base_weights[s] == wp.float64(0.0):
        return
    h = wp.float64(sizing[s])
    if h <= wp.float64(0.0) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 8)
        return
    denominator = h * h * h
    if dimension == 2:
        denominator *= h
    weight = base_weights[s] / denominator
    sample = wp.vec3d(samples[s])
    hit = nearest_site(tree, points, sample, active, occupied, False)
    if hit.index < 0 or not wp.isfinite(weight):
        wp.atomic_or(error, 0, 16)
        return
    owners[s] = hit.index
    delta = sample - wp.vec3d(points[hit.index])
    wp.atomic_add(mass, hit.index, weight)
    wp.atomic_add(moment, hit.index, weight * delta)
    wp.atomic_add(energies, component, weight * hit.distance2)
    wp.atomic_add(total, component, weight)
    wp.atomic_max(maximum, component, wp.sqrt(hit.distance2))


@wp.kernel
def make_targets(
    points: wp.array[wp.vec3],
    active: wp.array[int],
    surface_mass: wp.array[wp.float64],
    surface_moment: wp.array[wp.vec3d],
    crease_mass: wp.array[wp.float64],
    crease_moment: wp.array[wp.vec3d],
    occupied: wp.array[int],
    corners: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    role_count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if i >= active[0] or error[0] != 0:
        return
    p = wp.vec3d(points[i])
    role = int(0)
    if surface_mass[i] > wp.float64(0.0):
        role = 1
        p += surface_moment[i] / surface_mass[i]
    if crease_mass[i] > wp.float64(0.0):
        role = 2
        p = wp.vec3d(points[i]) + crease_moment[i] / crease_mass[i]
    if occupied[i] >= 0:
        role = 3
        p = wp.vec3d(corners[occupied[i]])
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 16)
        return
    targets[i] = wp.vec3(p)
    roles[i] = role
    wp.atomic_add(role_count, role, 1)


@wp.kernel
def finish_targets(
    points: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    role_count: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        targets[i] = points[i]
        roles[i] = 0
        if i == 0:
            for k in range(4):
                role_count[k] = 0


@wp.kernel
def apply_targets(
    points: wp.array[wp.vec3],
    active: wp.array[int],
    targets: wp.array[wp.vec3],
    roles: wp.array[int],
    error: wp.array[int],
    damping: float,
    max_step: float,
):
    i = wp.tid()
    if i >= active[0] or error[0] != 0 or roles[i] == 0:
        return
    if roles[i] == 3:
        points[i] = targets[i]
    else:
        p = wp.vec3d(points[i])
        delta = (wp.vec3d(targets[i]) - p) * wp.float64(damping)
        length = wp.length(delta)
        if length > wp.float64(max_step):
            delta *= wp.float64(max_step) / length
        points[i] = wp.vec3(p + delta)


def _sizing(value, n, device):
    if value is None:
        return wp.ones(n, dtype=float, device=device)
    if isinstance(value, np.ndarray):
        value = wp.array(value, device=device)
    elif type(value).__module__.startswith("torch"):
        value = wp.from_torch(value)
    if (
        not isinstance(value, wp.array)
        or value.dtype != wp.float32
        or value.shape != (n,)
        or value.device != device
        or not value.is_contiguous
    ):
        raise ValueError("Sizing must be contiguous float32, with one value per quadrature sample")
    return value


class BoundaryWorkspace:
    """Compute sampled surface/crease centroids and distinct corner targets.

    ``run()`` computes targets without changing input sites. ``apply()`` applies
    damped centroid updates and exact corner anchors. Roles are recomputed from
    current nearest-site assignments at every run; interior sites stay unchanged.
    """

    def __init__(
        self,
        domain,
        points,
        *,
        active_count=None,
        surface_sizing=None,
        crease_sizing=None,
        query_precision="float64",
    ):
        if query_precision not in ("float32", "float64", "mixed"):
            raise ValueError("query_precision must be 'float32', 'float64' or 'mixed'")
        self.query_precision = query_precision
        self._accumulate_samples = accumulate_samples
        if query_precision == "float32":
            from ._boundary_precision import accumulate_samples32

            self._accumulate_samples = accumulate_samples32
        elif query_precision == "mixed":
            from ._mixed_boundary import accumulate_samples_mixed

            self._accumulate_samples = accumulate_samples_mixed
        self.domain = domain
        self.points = _contiguous_points(points)
        self.n, self.device = len(self.points), self.points.device
        if self.device != domain.device or self.n < 1:
            raise ValueError("Boundary sites must be nonempty and on the domain device")
        if self.n >= 2**29:
            raise ValueError("Site capacity exceeds int32 hierarchy indexing")
        self._owners = points, active_count, surface_sizing, crease_sizing
        self.surface_sizing = _sizing(surface_sizing, len(domain.surface_points), self.device)
        self.crease_sizing = _sizing(crease_sizing, len(domain.crease_points), self.device)
        if active_count is None:
            active_count = wp.array([self.n], dtype=int, device=self.device)
        if (
            not isinstance(active_count, wp.array)
            or active_count.dtype != wp.int32
            or active_count.shape != (1,)
            or active_count.device != self.device
            or not active_count.is_contiguous
        ):
            raise ValueError("active_count must be a contiguous device int32 scalar")
        self.active_count = active_count
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self.targets = wp.empty_like(self.points)
            self.roles = wp.zeros(self.n, dtype=int)
            self.role_counts = wp.zeros(4, dtype=int)
            self.corner_sites = wp.full(len(domain.points), -1, dtype=int)
            self.corner_owners = wp.full(self.n, -1, dtype=int)
            self.surface_owners = wp.empty(len(domain.surface_points), dtype=int)
            self.crease_owners = wp.empty(len(domain.crease_points), dtype=int)
            self.surface_mass = wp.zeros(self.n, dtype=wp.float64)
            self.crease_mass = wp.zeros(self.n, dtype=wp.float64)
            self._surface_moment = wp.zeros(self.n, dtype=wp.vec3d)
            self._crease_moment = wp.zeros(self.n, dtype=wp.vec3d)
            self.energy = wp.zeros(2, dtype=wp.float64)
            self.total_weight = wp.zeros(2, dtype=wp.float64)
            self.max_distance = wp.zeros(2, dtype=wp.float64)
            self._proposals = wp.empty(len(domain.points), dtype=int)
            self._winners = wp.empty(self.n, dtype=int)
            self._pending = wp.zeros(1, dtype=int)
            self._remaining = wp.zeros(1, dtype=int)
            self.corner_rounds = wp.zeros(1, dtype=int)
            self._safe = wp.empty_like(self.points)
            self._site_ready = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            self._prepare()
            self._hierarchy = SiteHierarchy(self._safe)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def _prepare(self):
        self.error.zero_()
        wp.launch(
            prepare_sites,
            self.n,
            inputs=[
                self.points,
                self._safe,
                self.active_count,
                self.domain.error,
                self.error,
                self.targets,
                self.roles,
            ],
            device=self.device,
        )

    def _corner_round(self):
        self._winners.fill_(2**31 - 1)
        self._remaining.zero_()
        wp.launch(
            propose_corners,
            len(self.corner_sites),
            inputs=[
                self._hierarchy.tree,
                self._safe,
                self.active_count,
                self.domain.corner_points,
                self.domain.corner_count,
                self.corner_sites,
                self.corner_owners,
                self._proposals,
                self._winners,
                self.error,
            ],
            device=self.device,
        )
        wp.launch(
            claim_corners,
            len(self.corner_sites),
            inputs=[
                self.domain.corner_count,
                self.corner_sites,
                self.corner_owners,
                self._proposals,
                self._winners,
                self._remaining,
                self.error,
            ],
            device=self.device,
        )
        wp.launch(
            continue_corners,
            1,
            inputs=[
                self._remaining,
                self.domain.corner_count,
                self.corner_rounds,
                self.error,
                self._pending,
            ],
            device=self.device,
        )

    def _record(self):
        self._prepare()
        wp.launch(sites_ready, 1, inputs=[self.error, self._site_ready], device=self.device)
        # Warp gives a child graph its own sort scratch during capture setup.
        # A top-level sort would instead add allocation/free nodes to replay.
        wp.capture_if(self._site_ready, self._hierarchy.rebuild)
        self.corner_sites.fill_(-1)
        self.corner_owners.fill_(-1)
        self.corner_rounds.zero_()
        self.role_counts.zero_()
        wp.launch(
            begin_corners,
            1,
            inputs=[self.domain.corner_count, self.active_count, self.error, self._pending],
            device=self.device,
        )
        wp.capture_while(self._pending, self._corner_round)
        self.energy.zero_()
        self.total_weight.zero_()
        self.max_distance.zero_()
        for samples, weights, sizing, owners, mass, moment, dimension, component in (
            (
                self.domain.surface_points,
                self.domain.surface_weights,
                self.surface_sizing,
                self.surface_owners,
                self.surface_mass,
                self._surface_moment,
                2,
                0,
            ),
            (
                self.domain.crease_points,
                self.domain.crease_weights,
                self.crease_sizing,
                self.crease_owners,
                self.crease_mass,
                self._crease_moment,
                1,
                1,
            ),
        ):
            mass.zero_()
            moment.zero_()
            wp.launch(
                self._accumulate_samples,
                len(samples),
                inputs=[
                    self._hierarchy.tree,
                    self._safe,
                    self.active_count,
                    self.corner_owners,
                    samples,
                    weights,
                    sizing,
                    dimension,
                    owners,
                    mass,
                    moment,
                    self.energy,
                    self.total_weight,
                    self.max_distance,
                    component,
                    self.error,
                ],
                block_dim=32,
                device=self.device,
            )
        wp.launch(
            make_targets,
            self.n,
            inputs=[
                self.points,
                self.active_count,
                self.surface_mass,
                self._surface_moment,
                self.crease_mass,
                self._crease_moment,
                self.corner_owners,
                self.domain.corner_points,
                self.targets,
                self.roles,
                self.role_counts,
                self.error,
            ],
            device=self.device,
        )
        wp.launch(
            finish_targets,
            self.n,
            inputs=[self.points, self.targets, self.roles, self.role_counts, self.error],
            device=self.device,
        )

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self

    def apply(self, *, damping=1.0, max_step=1.0e30, stream=None):
        if not math.isfinite(damping) or not 0 < damping <= 1:
            raise ValueError("damping must be finite and in (0, 1]")
        if not math.isfinite(max_step) or not 0 < max_step <= 1.0e30:
            raise ValueError("max_step must be finite and in (0, 1e30]")
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            wp.launch(
                apply_targets,
                self.n,
                inputs=[
                    self.points,
                    self.active_count,
                    self.targets,
                    self.roles,
                    self.error,
                    float(damping),
                    float(max_step),
                ],
                device=self.device,
            )
        return self
