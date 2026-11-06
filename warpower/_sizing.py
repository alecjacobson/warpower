"""Sampled positive sizing fields with device-resident Lipschitz extension."""

import math

import warp as wp

from . import _bvh
from ._domain import _contiguous_points
from ._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.struct
class SizingData:
    tree: _bvh.Tree
    points: wp.array[wp.vec3]
    values: wp.array[float]
    error: wp.array[int]
    gradation: wp.float64
    minimum: wp.float64
    maximum: wp.float64


@wp.func(inline=True)
def sizing_value(field: SizingData, p: wp.vec3d):
    """Evaluate the sampled cone envelope in float64; zero means invalid field."""
    if field.error[0] != 0:
        return wp.float64(0.0)
    if field.gradation == wp.float64(0.0):
        return wp.clamp(-wp.float64(field.tree.weight[1]), field.minimum, field.maximum)
    result = field.maximum
    stack = wp.zeros(shape=128, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        delta = wp.max(
            wp.max(wp.vec3d(field.tree.lower[node]) - p, p - wp.vec3d(field.tree.upper[node])),
            wp.vec3d(0.0),
        )
        bound = -wp.float64(field.tree.weight[node]) + field.gradation * wp.length(delta)
        # Leave a margin for arithmetic roundoff in lower bounds. This affects
        # pruning only; final values use the original sample values and positions.
        if bound > result * wp.float64(1.000000000001):
            continue
        if field.tree.count[node] > 0:
            for k in range(field.tree.start[node], field.tree.start[node] + field.tree.count[node]):
                i = field.tree.ids[k]
                value = wp.float64(field.values[i]) + field.gradation * wp.length(
                    wp.vec3d(field.points[i]) - p
                )
                result = wp.min(result, value)
        else:
            left, right = int(field.tree.left[node]), int(field.tree.right[node])
            # Visit the lower estimated cone first to tighten later pruning.
            dl = wp.max(
                wp.max(wp.vec3d(field.tree.lower[left]) - p, p - wp.vec3d(field.tree.upper[left])),
                wp.vec3d(0.0),
            )
            dr = wp.max(
                wp.max(
                    wp.vec3d(field.tree.lower[right]) - p, p - wp.vec3d(field.tree.upper[right])
                ),
                wp.vec3d(0.0),
            )
            bl = -wp.float64(field.tree.weight[left]) + field.gradation * wp.length(dl)
            br = -wp.float64(field.tree.weight[right]) + field.gradation * wp.length(dr)
            if bl < br:
                left, right = right, left
            # A radix key has 63 spatial bits plus at most 29 index tie bits;
            # this stack exceeds the maximum supported hierarchy depth.
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return wp.clamp(result, field.minimum, field.maximum)


@wp.kernel
def prepare_field(
    points: wp.array[wp.vec3],
    values: wp.array[float],
    safe: wp.array[wp.vec3],
    sizes: wp.array[float],
    negative: wp.array[float],
    error: wp.array[int],
):
    i = wp.tid()
    p = wp.vec3(points[i])
    value = float(values[i])
    if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 1)
        p = wp.vec3(0.0)
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_or(error, 0, 2)
        value = 1.0
    safe[i] = p
    sizes[i] = value
    negative[i] = -value


@wp.kernel
def ready_field(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] == 0)


@wp.kernel
def evaluate_field(
    field: SizingData,
    points: wp.array[wp.vec3],
    values: wp.array[float],
    error: wp.array[int],
):
    i = wp.tid()
    p = points[i]
    value = wp.float64(0.0)
    if field.error[0] != 0:
        wp.atomic_or(error, 0, 1)
    elif not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
        wp.atomic_or(error, 0, 2)
    else:
        value = sizing_value(field, wp.vec3d(p))
    values[i] = float(value)


def _device_values(values, n, device):
    if (
        not isinstance(values, wp.array)
        or values.dtype != wp.float32
        or values.shape != (n,)
        or values.device != device
        or not values.is_contiguous
    ):
        raise ValueError("Sizing samples must be contiguous device float32, one per position")
    return values


class SizingField:
    """Positive relative sizing from sampled upper constraints and a gradation bound.

    ``mu(x) = clamp(min_i(values[i] + gradation * distance(x, points[i])),
    minimum, maximum)``. Samples are upper constraints, not interpolation knots.
    A common positive scale changes relative weights but not their ratios.
    Call ``refit()`` after a device producer changes sample positions or values.
    """

    def __init__(self, points, values, *, gradation=1.0, minimum=1e-6, maximum=1e30):
        self.points = _contiguous_points(points)
        self.n, self.device = len(self.points), self.points.device
        if self.n < 1 or self.n >= 2**29:
            raise ValueError("Sizing sample count must lie in [1, 2**29)")
        self.values = _device_values(values, self.n, self.device)
        if not math.isfinite(gradation) or not 0 <= gradation <= 1e30:
            raise ValueError("gradation must be finite and in [0, 1e30]")
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or not 1e-30 <= minimum <= maximum <= 1e30
        ):
            raise ValueError("Sizing bounds must satisfy 1e-30 <= minimum <= maximum <= 1e30")
        self._owners = points, values
        with wp.ScopedDevice(self.device):
            self.error = wp.zeros(1, dtype=int)
            self._ready = wp.zeros(1, dtype=int)
            self._safe = wp.zeros_like(self.points)
            self._sizes = wp.ones(self.n, dtype=float)
            self._hierarchy = SiteHierarchy(self._safe)
            self.data = SizingData()
            self.data.tree = self._hierarchy.tree
            self.data.points, self.data.values, self.data.error = (
                self._safe,
                self._sizes,
                self.error,
            )
            self.data.gradation = wp.float64(gradation)
            self.data.minimum, self.data.maximum = wp.float64(minimum), wp.float64(maximum)
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record()
            self.graph = capture.graph
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def _record(self):
        self.error.zero_()
        wp.launch(
            prepare_field,
            self.n,
            inputs=[
                self.points,
                self.values,
                self._safe,
                self._sizes,
                self._hierarchy._weights,
                self.error,
            ],
            device=self.device,
        )
        wp.launch(ready_field, 1, inputs=[self.error, self._ready], device=self.device)
        wp.capture_if(self._ready, self._hierarchy.rebuild)

    def refit(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Sizing stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError("Use Warp ScopedCapture for conditional graphs")
                self._record()
            else:
                wp.capture_launch(self.graph, stream=stream)
        return self


class SizingQueryWorkspace:
    """Preallocated queries into a SizingField; ``run()`` performs no refit."""

    def __init__(self, field, points):
        self.field = field
        self.points = _contiguous_points(points)
        self.device, self.n = self.points.device, len(self.points)
        if self.device != field.device or self.n < 1:
            raise ValueError("Queries must be nonempty and on the field device")
        self._owner = points
        with wp.ScopedDevice(self.device):
            self.values = wp.empty(self.n, dtype=float)
            self.error = wp.zeros(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Query stream must use the input device")
        with wp.ScopedStream(stream):
            self.error.zero_()
            wp.launch(
                evaluate_field,
                self.n,
                inputs=[self.field.data, self.points, self.values, self.error],
                device=self.device,
            )
        return self
