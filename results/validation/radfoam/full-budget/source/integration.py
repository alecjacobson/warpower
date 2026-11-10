"""RadFoam scene subclass with native, Warpower and Paragram geometry backends."""

import importlib
import os
import sys
import time
from pathlib import Path

import torch
import warp as wp

UPSTREAM = Path(os.environ.get("RADFOAM_SOURCE_DIR", "/home/horde/references/radfoam")).resolve()
NATIVE = Path(
    os.environ.get("RADFOAM_BUILD_DIR", ".work/radfoam-fitting-20261110/native")
).resolve()
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(UPSTREAM))
native = importlib.import_module("radfoam_headless")

sys.modules["radfoam"] = native
RadFoamScene = importlib.import_module("radfoam_model.scene").RadFoamScene
COLMAPDataset = importlib.import_module("data_loader.colmap").COLMAPDataset


class AuditedPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.counts = dict(
            forward_calls=0,
            backward_calls=0,
            nonfinite_rgba=0,
            nonfinite_position_gradients=0,
            nonfinite_attribute_gradients=0,
            traversal_limit_rays=0,
        )

    def trace_forward(self, *args, **kwargs):
        result = self.pipeline.trace_forward(*args, **kwargs)
        self.counts["forward_calls"] += 1
        self.counts["nonfinite_rgba"] += int((~torch.isfinite(result["rgba"])).sum())
        self.counts["traversal_limit_rays"] += int(
            (result["num_intersections"].long() > 1024).sum()
        )
        return result

    def trace_backward(self, *args, **kwargs):
        result = self.pipeline.trace_backward(*args, **kwargs)
        self.counts["backward_calls"] += 1
        self.counts["nonfinite_position_gradients"] += int(
            (~torch.isfinite(result["points_grad"])).sum()
        )
        self.counts["nonfinite_attribute_gradients"] += int(
            (~torch.isfinite(result["attr_grad"])).sum()
        )
        return result


class Scene(RadFoamScene):
    def __init__(
        self, initial, model_args, backend, optimizer_args, steps, warmup, defer_geometry=False
    ):
        torch.nn.Module.__init__(self)
        self.backend = backend
        self.device = torch.device("cuda:0")
        self.attr_dtype = torch.float32
        self.cameras = None
        self.sh_degree = model_args.sh_degree
        self.num_init_points = len(initial["primal_points"])
        self.num_final_points = model_args.final_points
        self.activation_scale = model_args.activation_scale
        self.faces = None
        self.triangulation = None
        self.diagram = None
        self.geometry_records = []
        self.frame_callback = None
        self.pipeline = AuditedPipeline(native.create_pipeline(self.sh_degree, self.attr_dtype))
        for name in ["primal_points", "density", "att_dc", "att_sh"]:
            setattr(self, name, torch.nn.Parameter(initial[name].cuda().clone()))
        self.declare_optimizer(optimizer_args, warmup=warmup, max_iterations=steps)
        if not defer_geometry:
            self.update_triangulation(incremental=False)

    def update_triangulation(self, rebuild=True, incremental=False):
        if self.frame_callback is not None:
            self.frame_callback(self.primal_points.detach(), incremental)
        torch.cuda.synchronize()
        start = time.perf_counter()
        retries, permuted, statuses = 0, False, {}
        self.last_permutation = None
        if self.backend == "warpower-morton" and not incremental:
            from spatial import morton_permutation

            self.last_permutation = morton_permutation(self.primal_points)
            self.permute_points(self.last_permutation)
            self.diagram = None
            permuted = True
        if self.backend == "native":
            if self.triangulation is None:
                self.triangulation = native.Triangulation(self.primal_points.detach())
                permuted = True
            elif rebuild:
                candidate = self.primal_points.detach()
                perturbation = 1e-6
                while True:
                    try:
                        permuted = self.triangulation.rebuild(candidate, incremental=incremental)
                        break
                    except native.TriangulationFailedError:
                        retries += 1
                        if retries > 25:
                            raise
                        perturbation *= 2
                        incremental = False
                        candidate = self.primal_points.detach() + perturbation * torch.randn_like(
                            self.primal_points
                        )
                if retries > 5:
                    with torch.no_grad():
                        self.primal_points.copy_(candidate)
            if permuted:
                self.last_permutation = self.triangulation.permutation().long().clone()
                self.permute_points(self.last_permutation)
            self.point_adjacency = self.triangulation.point_adjacency()
            self.point_adjacency_offsets = self.triangulation.point_adjacency_offsets()
        else:
            kwargs = {}
            if self.backend.startswith("warpower"):
                import warpower as library

                if self.backend == "warpower-hints" and incremental and self.diagram is not None:
                    kwargs = {
                        "initial_guesses": self.diagram.adjacency,
                        "initial_guesses_offsets": self.diagram.offsets,
                    }
            else:
                import paragram as library
            with torch.no_grad():
                self.diagram = library.voronoi_diagram(self.primal_points.detach(), **kwargs)
            values, counts = torch.unique(self.diagram.status, return_counts=True)
            statuses = dict(zip(map(str, values.cpu().tolist()), counts.cpu().tolist()))
            if any(k != "0" for k in statuses):
                raise RuntimeError(f"{self.backend} non-success cells: {statuses}")
            self.point_adjacency = self.diagram.adjacency.view(torch.uint32)
            self.point_adjacency_offsets = self.diagram.offsets.view(torch.uint32)
        torch.cuda.synchronize()
        built = time.perf_counter()
        self.aabb_tree = native.build_aabb_tree(self.primal_points.detach())
        torch.cuda.synchronize()
        end = time.perf_counter()
        self.geometry_records.append(
            {
                "n": len(self.primal_points),
                "incremental_requested": incremental,
                "backend_seconds": built - start,
                "aabb_seconds": end - built,
                "total_seconds": end - start,
                "retries": retries,
                "permuted": permuted,
                "status_counts": statuses,
                "directed_edges": self.point_adjacency.numel(),
            }
        )


def initialize_runtime():
    torch.cuda.set_device(0)
    wp.init()
    # Leave room for Torch model state and rendering. This is deliberately lower
    # than the standalone construction benchmark's 32 GiB retention threshold.
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)


def checkpoint(model, optimizer, step, path, config):
    torch.save(
        {
            "parameters": {
                name: getattr(model, name).detach().cpu()
                for name in ["primal_points", "density", "att_dc", "att_sh"]
            },
            "optimizer": optimizer.state_dict(),
            "num_init_points": model.num_init_points,
            "step": step,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        path,
    )
