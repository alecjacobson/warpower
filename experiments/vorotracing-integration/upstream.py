"""Load the pinned recipe with either original CUDA or Warp GPU operations."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import warp as wp

SOURCE = Path(os.environ.get("VOROTRACING_SOURCE_DIR", "/home/horde/references/vorotracing"))
ROMA = Path(os.environ.get("ROMAV2_SOURCE_DIR", "/home/horde/references/RoMaV2"))
COMMIT = "75910484d6d46af25e180044ca6b9091204fe6fa"


def configure():
    sys.path[:0] = [
        os.environ.get("VOROTRACING_PYCOLMAP_PATH", "/opt/warpower-generated-cache/voro-deps"),
        str(SOURCE),
        str(ROMA / "src"),
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
    actual_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != COMMIT:
        raise RuntimeError(f"Expected VoroTracing {COMMIT}, found {actual_commit}")
    subprocess.run(
        ["git", "-C", str(SOURCE), "diff", "--quiet", "HEAD", "--", "vorotracing"], check=True
    )
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
                # The reference training tracer reads four edges at a time.
                # Retain its harmless tail padding so saved Warp checkpoints
                # can also be consumed by the native forward/backward kernels.
                torch.nn.functional.pad(result.adjacency, (0, 32)).view(torch.uint32),
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

    if backend == "warp":

        @torch.no_grad()
        def get_starting_point(self, rays):
            if rays.ndim == 1:
                rays = rays[None, :]
            origins = rays[..., :3].reshape(-1, 3).contiguous()
            return self.bvh.query(self.primal_points, origins).reshape(rays.shape[:-1])

        vorotracing.VoroTracing.get_starting_point = get_starting_point

    if audit:
        from numerics import instrument_forward, instrument_native
        from vorotracing import cuda

        if backend == "native":
            from vorotracing.cuda import _wrapper

            instrument_native(_wrapper)
        instrument_forward(cuda)

    return vorotracing
