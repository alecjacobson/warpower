"""Common, counted version of the reference's nonfinite-gradient masking.

The original CUDA wrapper silently replaces all nonfinite parameter gradients
with zero. Both fitting backends use this same Warp pass, which also records
what was replaced. The reference geometry and ray kernels stay unchanged.
"""

import torch
import warp as wp

_counts = None
LABELS = [
    "points_gradient",
    "diffuse_gradient",
    "specular_gradient",
    "density_gradient",
    "rgba",
    "distortion",
    "capped_rays",
]


def counters():
    global _counts
    if _counts is None:
        _counts = torch.zeros(len(LABELS), dtype=torch.int64, device="cuda")
    return _counts


@wp.kernel
def clean(values: wp.array[float], counts: wp.array[wp.int64], category: int):
    i = wp.tid()
    if not wp.isfinite(values[i]):
        wp.atomic_add(counts, category, wp.int64(1))
        values[i] = 0.0


@wp.kernel
def check_forward(
    rgba: wp.array[wp.vec4],
    dist: wp.array[float],
    intersections: wp.array[int],
    cap: int,
    counts: wp.array[wp.int64],
):
    i = wp.tid()
    for j in range(4):
        if not wp.isfinite(rgba[i][j]):
            wp.atomic_add(counts, 4, wp.int64(1))
    if not wp.isfinite(dist[i]):
        wp.atomic_add(counts, 5, wp.int64(1))
    if intersections[i] >= cap:
        wp.atomic_add(counts, 6, wp.int64(1))


def sanitize(gradient, category):
    wp.launch(
        clean,
        gradient.numel(),
        inputs=[wp.from_torch(gradient.detach().view(-1)), wp.from_torch(counters()), category],
        stream=wp.stream_from_torch(torch.cuda.current_stream()),
    )
    return gradient


def audit_forward(rgba, dist, intersections, cap):
    wp.launch(
        check_forward,
        dist.numel(),
        inputs=[
            wp.from_torch(rgba.detach().reshape(-1, 4), dtype=wp.vec4),
            wp.from_torch(dist.detach().reshape(-1)),
            wp.from_torch(intersections.view(torch.int32).reshape(-1)),
            cap,
            wp.from_torch(counters()),
        ],
        stream=wp.stream_from_torch(torch.cuda.current_stream()),
    )


def reset():
    counters().zero_()


def snapshot():
    return dict(zip(LABELS, counters().cpu().tolist()))


def instrument_native(wrapper):
    class CountedTorch:
        index = 0

        def __getattr__(self, name):
            return getattr(torch, name)

        def nan_to_num(self, value, *, nan, posinf, neginf, out):
            assert nan == posinf == neginf == 0 and out is value
            category = self.index % 4
            self.index += 1
            return sanitize(value, category)

    wrapper.torch = CountedTorch()


def instrument_forward(ops):
    original = ops.TraceVoroTracing

    class CountedTrace:
        @staticmethod
        def apply(*args):
            result = original.apply(*args)
            audit_forward(result[0], result[5], result[3], args[10])
            return result

    ops.TraceVoroTracing = CountedTrace
