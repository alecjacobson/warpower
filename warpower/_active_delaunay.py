"""Internal VTM routing for an already padded, device-controlled active prefix."""

import operator

import warp as wp

from ._delaunay_recovery import DelaunayWorkspace
from ._wall_reuse import WallReuseDelaunayWorkspace as ReuseDelaunayWorkspace

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def choose(
    active: wp.array[int],
    capacities: wp.array[int],
    flags: wp.array[int],
    calls: wp.array[int],
    selected: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
):
    selected[0] = 0
    count[0] = 0
    error[0] = 8
    for i in range(capacities.shape[0]):
        flags[i] = 0
    if active[0] < 4 or active[0] > capacities[capacities.shape[0] - 1]:
        return
    error[0] = 0
    for i in range(capacities.shape[0]):
        if active[0] <= capacities[i]:
            flags[i] = 1
            calls[i] += 1
            selected[0] = capacities[i]
            break


class ActiveDelaunayWorkspace:
    """Private mesh-output interface, deliberately not a DelaunayWorkspace.

    The caller MUST prepare every inactive point as an exact copy of point zero
    before run(). VTM owns that scratch buffer separately from its input sites.
    Then the smallest fitting prefix contains every distinct site, and its full
    original-coordinate certificate proves the active hull. Original site IDs
    and canonical first duplicate owners are unchanged by taking a prefix.

    Only connectivity, counts and scalar diagnostics are published. Certificate
    face indices belong to the selected private constructor and are not exposed.
    All storage, views and conditional graphs are allocated during setup.
    """

    _diagnostics = (
        "certificate_error",
        "base_error",
        "candidate_error",
        "attempts",
        "recovered",
        "base_attempted",
        "recovery_first",
        "auxiliary_level",
        "symbolic_attempted",
        "symbolic_recovered",
        "symbolic_error",
        "symbolic_expansions",
        "facet_attempted",
        "facet_recovered",
        "facet_error",
        "facet_rounds",
        "facet_seeds",
        "fast_attempted",
        "fast_accepted",
        "fast_error",
        "fast_repaired_cells",
    )

    def __init__(self, points, *, active_count, prefix_capacities=(1024, 2048), **options):
        levels = tuple(operator.index(n) for n in prefix_capacities)
        if not levels or any(n < 4 or n > 8192 for n in levels):
            raise ValueError("Prefix capacities must be in [4, 8192]")
        if tuple(sorted(set(levels))) != levels or levels[-1] >= len(points):
            raise ValueError("Prefix capacities must be increasing and smaller than full capacity")
        if (
            not isinstance(active_count, wp.array)
            or active_count.dtype != wp.int32
            or active_count.shape != (1,)
            or active_count.device != points.device
            or not active_count.is_contiguous
        ):
            raise ValueError("active_count must be one contiguous device int32")
        self.active_count = active_count
        self._full_reuses_cells = 2048 < len(points) <= 16384
        full_constructor = ReuseDelaunayWorkspace if self._full_reuses_cells else DelaunayWorkspace
        self._full = full_constructor(points, **options)
        self.points, self.n, self.device = self._full.points, self._full.n, self._full.device
        # Reuse the full output allocation, but never expose its stale face index.
        self.tetrahedra = self._full.tetrahedra
        self.tetrahedra_count = self._full.tetrahedra_count
        self.error = self._full.error
        for name in self._diagnostics:
            setattr(self, name, getattr(self._full, name))
        # Existing mesher fresh-solve reset controls for the full constructor.
        self._preferred = self._full._preferred
        self._preferred_level = self._full._preferred_level
        self.fast_candidates = self._full.fast_candidates
        self._fast_available = self._full._fast_available
        self.facet_recovery = True
        with wp.ScopedDevice(self.device):
            children = []
            for level in levels:
                child_options = dict(options)
                child_options["tetrahedra_capacity"] = min(len(self.tetrahedra), 8 * level)
                children.append(ReuseDelaunayWorkspace(self.points[:level], **child_options))
            self._children = tuple(children)
            self.capacities = (*levels, self.n)
            self._capacities = wp.array(self.capacities, dtype=int)
            self._flags = wp.zeros(len(self.capacities), dtype=int)
            self._conditions = tuple(self._flags[i : i + 1] for i in range(len(self.capacities)))
            self.route_calls = wp.zeros(len(self.capacities), dtype=int)
            self.selected_capacity = wp.zeros(1, dtype=int)
            self._fallback = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

    def clear_reuse(self):
        """Start a fresh trajectory; all operations are device graph nodes."""
        self.route_calls.zero_()
        if self._full_reuses_cells:
            self._full.clear_reuse()
        for child in self._children:
            child.clear_reuse()
            child._preferred.zero_()
            child._preferred_level.zero_()

    def _publish(self, child):
        child.run()
        wp.copy(self.tetrahedra, child.tetrahedra, count=child.tetrahedra.size)
        wp.copy(self.tetrahedra_count, child.tetrahedra_count)
        wp.copy(self.error, child.error)
        for name in self._diagnostics:
            wp.copy(getattr(self, name), getattr(child, name))

    def _record(self):
        wp.launch(
            choose,
            1,
            inputs=[
                self.active_count,
                self._capacities,
                self._flags,
                self.route_calls,
                self.selected_capacity,
                self.tetrahedra_count,
                self.error,
            ],
            device=self.device,
        )
        for condition, child in zip(self._conditions, self._children):
            wp.capture_if(condition, lambda child=child: self._publish(child))
        wp.capture_if(self._conditions[-1], self._full.run)
        # A prefix has a smaller finite output/strategy budget. Preserve the
        # original full constructor as fallback instead of introducing failures.
        wp.launch(
            full_fallback,
            1,
            inputs=[self.error, self.selected_capacity, self.n, self.route_calls, self._fallback],
            device=self.device,
        )
        wp.capture_if(self._fallback, self._full.run)

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError(
                        "Use Warp ScopedCapture on the shared stream for conditional graphs"
                    )
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self


@wp.kernel
def full_fallback(
    error: wp.array[int],
    selected: wp.array[int],
    capacity: int,
    calls: wp.array[int],
    ready: wp.array[int],
):
    ready[0] = int(error[0] != 0 and selected[0] > 0 and selected[0] < capacity)
    if ready[0] != 0:
        selected[0] = capacity
        calls[calls.shape[0] - 1] += 1
