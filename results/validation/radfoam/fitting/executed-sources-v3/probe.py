"""Small fitting probe using Warpower and RadFoam's unchanged CUDA renderer.

The C ABI bridge supports SH degree zero on the default CUDA stream. This is
not the complete RadFoam trainer, data loader, viewer, or densification loop.
"""

import argparse
import ctypes as ct
import hashlib
import importlib.util
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from scipy.spatial import Delaunay

import warpower

POINTERS = (
    "points attributes adjacency offsets ray_data start depth_quantiles rgba depth "
    "depth_indices intersections contribution rgba_grad depth_grad point_grad "
    "attribute_grad ray_grad"
).split()


class Buffers(ct.Structure):
    _fields_ = [(k, ct.c_uint32) for k in ("sites", "edges", "rays", "quantiles")] + [
        (k, ct.c_void_p) for k in POINTERS
    ]


class Pipeline:
    def __init__(self, binary):
        self.library = ct.CDLL(str(binary.resolve()))
        for name in ("rf_forward", "rf_backward"):
            function = getattr(self.library, name)
            function.argtypes = [ct.POINTER(Buffers)]
            function.restype = ct.c_int
        self.library.rf_error.restype = ct.c_char_p
        self.backward_calls = 0

    def call(self, name, points, attributes, adjacency, offsets, rays, start, **kw):
        assert torch.cuda.current_stream() == torch.cuda.default_stream()
        assert adjacency.dtype == offsets.dtype == start.dtype == torch.uint32
        tensors = dict(
            points=points,
            attributes=attributes,
            adjacency=adjacency,
            offsets=offsets,
            ray_data=rays,
            start=start,
            **kw,
        )
        tensors = {k: v.contiguous() for k, v in tensors.items() if v is not None}
        assert all(v.is_cuda for v in tensors.values())
        buffers = Buffers(
            sites=len(points),
            edges=len(adjacency),
            rays=len(rays),
            quantiles=tensors["depth_quantiles"].shape[-1] if "depth_quantiles" in tensors else 0,
        )
        for key, tensor in tensors.items():
            setattr(buffers, key, tensor.data_ptr())
        if getattr(self.library, name)(ct.byref(buffers)):
            raise RuntimeError(self.library.rf_error().decode())

    def trace_forward(
        self,
        points,
        attributes,
        adjacency,
        offsets,
        rays,
        start,
        depth_quantiles=None,
        return_contribution=False,
    ):
        result = {
            "rgba": torch.empty((len(rays), 4), device="cuda"),
            "num_intersections": torch.empty((len(rays), 1), device="cuda", dtype=torch.uint32),
        }
        if depth_quantiles is not None:
            result["depth"] = torch.zeros_like(depth_quantiles)
            result["depth_indices"] = torch.zeros_like(depth_quantiles, dtype=torch.uint32)
        if return_contribution:
            result["contribution"] = torch.zeros((len(points), 1), device="cuda")
        self.call(
            "rf_forward",
            points,
            attributes,
            adjacency,
            offsets,
            rays,
            start,
            depth_quantiles=depth_quantiles,
            rgba=result["rgba"],
            intersections=result["num_intersections"],
            depth=result.get("depth"),
            depth_indices=result.get("depth_indices"),
            contribution=result.get("contribution"),
        )
        assert torch.isfinite(result["rgba"]).all()
        return result

    def trace_backward(
        self,
        points,
        attributes,
        adjacency,
        offsets,
        rays,
        start,
        rgba,
        rgba_grad,
        depth_quantiles=None,
        depth_indices=None,
        depth_grad=None,
        ray_error=None,
    ):
        assert ray_error is None, "error-map accumulation is outside this probe"
        point_grad, attr_grad = torch.zeros_like(points), torch.zeros_like(attributes)
        ray_grad = torch.empty_like(rays)
        self.call(
            "rf_backward",
            points,
            attributes,
            adjacency,
            offsets,
            rays,
            start,
            depth_quantiles=depth_quantiles,
            depth_indices=depth_indices,
            rgba=rgba,
            rgba_grad=rgba_grad,
            depth_grad=depth_grad,
            point_grad=point_grad,
            attribute_grad=attr_grad,
            ray_grad=ray_grad,
        )
        # Check before upstream TraceRays masks nonfinite gradients.
        assert torch.isfinite(point_grad).all() and torch.isfinite(attr_grad).all()
        self.backward_calls += 1
        return {"points_grad": point_grad, "attr_grad": attr_grad, "ray_grad": ray_grad}


def scipy_graph(points):
    simplices = Delaunay(points.detach().cpu().numpy().astype(np.float64)).simplices
    pairs = np.concatenate([simplices[:, ij] for ij in itertools.combinations(range(4), 2)])
    pairs = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(pairs[:, 0], minlength=len(points)))])
    return (
        torch.from_numpy(pairs[:, 1].astype(np.uint32)).cuda(),
        torch.from_numpy(offsets.astype(np.uint32)).cuda(),
    )


def warpower_graph(points):
    with torch.no_grad():
        result = warpower.voronoi_diagram(points.detach())
    assert torch.all(result.status == 0), result.status
    adjacency, offsets = result.adjacency.view(torch.uint32), result.offsets.view(torch.uint32)
    assert adjacency.data_ptr() == result.adjacency.data_ptr()
    assert offsets.data_ptr() == result.offsets.data_ptr()
    return adjacency, offsets


def graph_hash(graph):
    adjacency, offsets = (x.cpu().numpy() for x in graph)
    rows = np.repeat(np.arange(len(offsets) - 1, dtype=np.uint64), np.diff(offsets))
    keys = (rows << np.uint64(32)) | adjacency.astype(np.uint64)
    return hashlib.sha256(np.sort(keys).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radfoam", type=Path, default=Path("/home/horde/references/radfoam"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    wp.init()
    spec = importlib.util.spec_from_file_location(
        "upstream_render", args.radfoam / "radfoam_model/render.py"
    )
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    pipeline = Pipeline(args.binary)
    torch.manual_seed(2026)
    torch.cuda.set_device(0)
    rng = np.random.default_rng(2026)
    inner = torch.tensor(rng.uniform(-1, 1, (64, 3)), dtype=torch.float32, device="cuda")
    guards = torch.tensor(
        list(itertools.product([-4, 4], repeat=3)), dtype=torch.float32, device="cuda"
    )
    teacher_points = torch.cat([inner, guards])
    teacher_attributes = torch.cat(
        [
            torch.cat([2 * torch.sin(inner * 2), torch.full((64, 1), 1.5, device="cuda")], dim=1),
            torch.zeros((8, 4), device="cuda"),
        ]
    )
    cameras = torch.tensor(
        [[3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0], [0, 0, 3], [0, 0, -3]],
        dtype=torch.float32,
        device="cuda",
    ).repeat_interleave(256, dim=0)
    targets = torch.tensor(
        rng.uniform(-1, 1, (len(cameras), 3)), dtype=torch.float32, device="cuda"
    )
    rays = torch.cat([cameras, targets - cameras], dim=1)

    def trace(points, attributes, graph, quantiles=None):
        with torch.no_grad():
            start = torch.cdist(rays[:, :3], points).argmin(dim=1).to(torch.uint32)
        return render.TraceRays.apply(
            pipeline, points, attributes, *graph, rays, start, quantiles, False
        )

    target = trace(teacher_points, teacher_attributes, scipy_graph(teacher_points))[0].detach()
    full_graph, clipped_graph = scipy_graph(teacher_points), warpower_graph(teacher_points)
    outputs, gradients = [], []
    for graph in (full_graph, clipped_graph):
        p, a = teacher_points.clone().requires_grad_(), teacher_attributes.clone().requires_grad_()
        out = trace(p, a, graph)[0]
        gradients.append(torch.autograd.grad(out.square().mean(), (p, a)))
        outputs.append(out.detach())
    control = {
        "rgba_max_abs_difference": float((outputs[0] - outputs[1]).abs().max()),
        "position_gradient_max_abs_difference": float(
            (gradients[0][0] - gradients[1][0]).abs().max()
        ),
        "attribute_gradient_max_abs_difference": float(
            (gradients[0][1] - gradients[1][1]).abs().max()
        ),
        "full_hull_directed_edges": len(full_graph[0]),
        "clipped_directed_edges": len(clipped_graph[0]),
    }
    assert control["rgba_max_abs_difference"] < 1e-5, control
    assert control["position_gradient_max_abs_difference"] < 1e-5, control
    assert control["attribute_gradient_max_abs_difference"] < 1e-5, control
    print(json.dumps({"control": control}), flush=True)

    points = torch.nn.Parameter(inner + 0.03 * torch.randn_like(inner))
    colors = torch.nn.Parameter(torch.zeros((64, 3), device="cuda"))
    density = torch.nn.Parameter(torch.full((64, 1), 0.05, device="cuda"))
    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": 0.002},
            {"params": [colors], "lr": 0.03},
            {"params": [density], "lr": 0.01},
        ]
    )
    initial_positions = points.detach().clone()
    records, graph_hashes = [], []
    started = time.perf_counter()
    for step in range(args.steps):
        p = torch.cat([points, guards])
        attributes = torch.cat(
            [
                torch.cat([colors, F.softplus(density, beta=10)], dim=1),
                torch.zeros((8, 4), device="cuda"),
            ]
        )
        graph = warpower_graph(p)
        graph_hashes.append(graph_hash(graph))
        quantiles = torch.tensor([0.5, 0.1], device="cuda").expand(len(rays), -1).contiguous()
        rgba, depth, _, intersections, _ = trace(p, attributes, graph, quantiles)
        valid_depth = (depth > 0).all(dim=1)
        photometric = (rgba - target).square().mean()
        depth_loss = ((depth[:, 0] - depth[:, 1]).abs() * valid_depth).mean()
        loss = photometric + 1e-4 * depth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(
            x.grad is not None and torch.isfinite(x.grad).all() for x in [points, colors, density]
        )
        row = {
            "step": step,
            "photometric_mse": float(photometric.detach()),
            "loss": float(loss.detach()),
            "position_gradient_norm": float(points.grad.norm()),
            "max_intersections": int(intersections.to(torch.int64).max()),
        }
        records.append(row)
        optimizer.step()
        if step % 30 == 0 or step == args.steps - 1:
            print(json.dumps(row), flush=True)
    assert records[-1]["photometric_mse"] < 0.25 * records[0]["photometric_mse"]
    assert any(x["position_gradient_norm"] > 0 for x in records)
    report = {
        "scope": "Synthetic SH0 teacher fitting with unmodified upstream RadFoam CUDA forward/backward and Python TraceRays autograd; C ABI headless bridge; default stream; 64 trainable sites plus 8 fixed zero-density guard sites; 1536 rays; graph rebuilt every step. Not a full dataset run or upstream train.py/densification/resume test.",
        "radfoam_commit": subprocess.check_output(
            ["git", "-C", str(args.radfoam), "rev-parse", "HEAD"], text=True
        ).strip(),
        "warpower_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(),
        "control": control,
        "steps": args.steps,
        "records": records,
        "backward_calls_finite_before_masking": pipeline.backward_calls,
        "graph_changes": sum(a != b for a, b in zip(graph_hashes, graph_hashes[1:])),
        "graph_change_definition": "Change in sorted directed edge set, independent of CSR row ordering",
        "distinct_topologies": len(set(graph_hashes)),
        "position_displacement_norm": float((points.detach() - initial_positions).norm()),
        "wall_seconds_including_checks": time.perf_counter() - started,
        "zero_copy_uint32_views": True,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("renderer_bridge.cpp"),
                args.radfoam / "src/tracing/pipeline.cu",
                args.radfoam / "radfoam_model/render.py",
            ]
        },
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "graph_changes": report["graph_changes"],
                "mse_reduction": records[0]["photometric_mse"] / records[-1]["photometric_mse"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
