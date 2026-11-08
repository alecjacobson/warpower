"""Private setup-only prototype: existing constructors plus larger prefix choices.

The production plan keeps the direct private router's old default; VTM requests
these explicit levels. This prototype installs that policy before VTM setup.
No device kernel is changed. The full constructor remains output owner/fallback.
"""
import operator
import warp as wp
from warpower._active_delaunay import ActiveDelaunayWorkspace as Base
from warpower._delaunay_recovery import DelaunayWorkspace
from warpower._wall_reuse import WallReuseDelaunayWorkspace as ReuseDelaunayWorkspace

class LargerPrefixDelaunayWorkspace(Base):
    def __init__(self, points, *, active_count, prefix_capacities=None, **options):
        if prefix_capacities is None:
            prefix_capacities = tuple(n for n in (1024, 2048, 4096, 8192) if n < len(points))
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
            wp.load_module(module="warpower._active_delaunay", device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
        wp.synchronize_device(self.device)

def install():
    import warpower._active_delaunay as active
    assert active.ActiveDelaunayWorkspace is Base
    active.ActiveDelaunayWorkspace = LargerPrefixDelaunayWorkspace
