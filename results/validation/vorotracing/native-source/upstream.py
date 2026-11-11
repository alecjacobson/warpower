"""Load the pinned recipe with either original CUDA or Warp GPU operations."""

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import warp as wp

SOURCE = Path("/home/horde/references/vorotracing")


def configure():
    sys.path[:0] = [
        "/opt/warpower-generated-cache/voro-deps",
        str(SOURCE),
        "/home/horde/references/RoMaV2/src",
    ]
    os.environ.setdefault("TORCH_HOME", "/opt/warpower-generated-cache/torch-models")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", "/opt/warpower-generated-cache/vorotracing-extensions"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)


def load(backend, audit=False):
    configure()
    if backend == "warp":
        from warp_renderer import Nearest, Trace, infer, prepared_edges

        import warpower

        ops = ModuleType("vorotracing.cuda")
        ops.__path__ = []
        ops.NNBVH = Nearest
        ops.prefetch_octmap_adj = prepared_edges
        ops.trace_octmap_infer = infer

        def adjacency(points, **kwargs):
            result = warpower.voronoi_diagram(points.detach())
            if bool(torch.any(result.status != 0)):
                raise RuntimeError(
                    f"WarPower adjacency failure: {torch.unique(result.status, return_counts=True)}"
                )
            return (
                result.adjacency.view(torch.uint32),
                result.offsets.view(torch.uint32),
                result.status,
            )

        class Tracing:
            @staticmethod
            def apply(
                points,
                diffuse,
                specular,
                density,
                adj,
                offsets,
                rays,
                starts,
                depth_quantiles,
                threshold,
                max_steps,
                return_contribution,
            ):
                if depth_quantiles is not None or return_contribution:
                    raise NotImplementedError(
                        "The fixed-budget paper recipe does not request quantiles or contribution maps"
                    )
                rgba, dist, counts = Trace.apply(
                    points,
                    diffuse,
                    specular,
                    density,
                    adj,
                    offsets,
                    rays,
                    starts,
                    threshold,
                    max_steps,
                )
                return (
                    rgba,
                    None,
                    None,
                    counts,
                    SimpleNamespace(ray_error=None, point_error=None),
                    dist,
                )

        ops.build_voronoi_adjacency = adjacency
        ops.TraceVoroTracing = Tracing
        sys.modules["vorotracing.cuda"] = ops
    elif backend != "native":
        raise ValueError(backend)
    import vorotracing

    if audit:
        from numerics import instrument_forward, instrument_native
        from vorotracing import cuda

        if backend == "native":
            from vorotracing.cuda import _wrapper

            instrument_native(_wrapper)
        instrument_forward(cuda)

    return vorotracing
